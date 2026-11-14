"""
store and forward sync from a field unit to the command center
detections wait in the local database until the command center confirms each one
when the link is down nothing is lost, the unit keeps detecting and sends the backlog once it is back

newest detections go first, so coordinators see the current picture as soon as a link returns
resending is safe, the command center ignores detection ids it already has
a detection the command center refuses is held back with the reason, counted and shown, never quietly dropped

usage
    python -m src.sync --url https://command.example.org
    python -m src.sync --once
    python -m src.sync --retry-refused
"""
import argparse
import gzip
import json
import logging
import random
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import closing
from datetime import datetime

from src.config import config
from src.logger import connect, data_logger, now, prune, read_status

log = logging.getLogger(__name__)

# fields sent for each detection
columns = ("uid", "unit_id", "timestamp", "latitude", "longitude", "confidence",
           "thermal_peak", "bbox", "detector", "latency_ms", "position_source")

# rows that can be sent, not yet confirmed and not refused
waiting = "synced_at is null and sync_error is null"
# rows the command center refused, held until someone retries them
refused = "synced_at is null and sync_error is not null"

# the retry wait stops doubling after this many failures in a row
# without a cap the sum overflows after about a thousand failures, a few days offline
max_doublings = 16


class send_error(Exception):
    # the batch did not reach the command center, so it stays queued
    pass


def pending(limit):
    # the newest detections the command center has not confirmed yet
    with closing(connect()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(f"""
            select {", ".join(columns)} from detections
            where {waiting}
            order by id desc limit ?
        """, (limit,)).fetchall()

    detections = []
    for row in rows:
        detection = dict(row)
        # boxes are saved as text like [x, y, w, h]
        try:
            detection["bbox"] = json.loads(detection["bbox"]) if detection["bbox"] else None
        except ValueError:
            detection["bbox"] = None
        detections.append(detection)
    return detections


def pending_count():
    # detections still waiting to be sent
    with closing(connect()) as conn:
        return conn.execute(f"select count(*) from detections where {waiting}").fetchone()[0]


def refused_count():
    # detections the command center refused, they need someone to look at them
    with closing(connect()) as conn:
        return conn.execute(f"select count(*) from detections where {refused}").fetchone()[0]


def refused_reasons():
    # each reason the command center gave and how many detections it applies to
    with closing(connect()) as conn:
        return conn.execute(f"select sync_error, count(*) from detections where {refused} group by sync_error").fetchall()


def retry_refused():
    # puts refused detections back in the queue, for after the cause has been fixed
    with closing(connect()) as conn, conn:
        return conn.execute(f"update detections set sync_error = null where {refused}").rowcount


def mark(accepted, rejected):
    # accepted detections are done
    # refused ones keep their reason and stay unconfirmed, so they are counted and can be retried
    # they are left out of the queue so one bad row cannot block the rest
    when = now()
    with closing(connect()) as conn, conn:
        conn.executemany("update detections set synced_at = ? where uid = ? and synced_at is null",
                         [(when, uid) for uid in accepted])
        conn.executemany("update detections set sync_error = ? where uid = ? and synced_at is null",
                         [(reason, uid) for uid, reason in rejected.items()])


def poster(url, path, token=None, timeout=15, insecure=None):
    # returns a function that posts json to one command center address and gives back its reply
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"{url} is not an http or https address")
    # a token sent over plain http can be read by anyone on the way, along with where people are
    local = parts.hostname in ("127.0.0.1", "localhost", "::1")
    allowed = config.sync_insecure if insecure is None else insecure
    if token and parts.scheme == "http" and not local and not allowed:
        raise ValueError(f"refusing to send the token to {url} over plain http, use https, "
                         "or set NEUSE_SYNC_INSECURE=1 on a network you trust")

    endpoint = url.rstrip("/") + path
    headers = {"Content-Type": "application/json", "Content-Encoding": "gzip"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    def post(payload):
        # compressed, since field links can be slow and paid by the megabyte
        body = gzip.compress(json.dumps(payload).encode("utf-8"))
        request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as err:
            raise send_error(f"command center answered {err.code}") from err
        except (urllib.error.URLError, OSError) as err:
            raise send_error(f"could not reach the command center, {err}") from err
        except ValueError as err:
            raise send_error("command center reply was not json") from err

    return post


def http_sender(url, token=None, timeout=15, insecure=None):
    # returns a function that posts one batch of detections
    # sent_at is this unit's clock at the moment of sending
    # the command center compares it with its own clock and corrects detection times from a unit whose clock is wrong
    post = poster(url, "/api/v1/detections", token, timeout, insecure)
    return lambda detections: post({"sent_at": now(), "detections": detections})


def http_coverage_sender(url, token=None, timeout=15, insecure=None):
    # returns a function that posts ground squares the unit has covered
    # sent_at lets the command center correct the times from a unit whose clock is wrong, as for detections
    post = poster(url, "/api/v1/coverage", token, timeout, insecure)
    return lambda cells: post({"unit_id": config.unit_id, "sent_at": now(), "cells": cells})


def pending_coverage(limit=2000):
    # covered ground squares the command center does not have yet, newest first
    with closing(connect()) as conn:
        return [list(row) for row in conn.execute("""
            select cell_row, cell_col, last_seen from coverage where synced = 0
            order by last_seen desc limit ?
        """, (limit,))]


def sync_coverage(send_coverage, batch=2000):
    # sends covered squares a batch at a time, returns how many were sent or None when the link failed
    sent = 0
    while True:
        cells = pending_coverage(batch)
        if not cells:
            return sent
        try:
            reply = send_coverage(cells)
        except send_error:
            return None
        if not isinstance(reply, dict) or reply.get("saved") != len(cells):
            return None
        with closing(connect()) as conn, conn:
            # only squares not seen again while they were being sent are marked as sent
            conn.executemany("update coverage set synced = 1 where cell_row = ? and cell_col = ? and last_seen = ?",
                             [tuple(cell) for cell in cells])
        sent += len(cells)


def http_status_sender(url, token=None, timeout=15, insecure=None):
    # returns a function that posts the unit's status note
    return poster(url, "/api/v1/status", token, timeout, insecure)


def status_note():
    # what the command center hears from a unit even when it has nothing new
    # where the unit is, what it runs, how many detections are waiting or refused
    # and how long ago the detector last handled a frame, so a unit that syncs but no longer detects shows up
    status = read_status() or {}
    idle = None
    if status.get("updated_at"):
        # both readings come from this unit's clock, so a wrong clock does not change the difference
        idle = max(0.0, (datetime.now().astimezone() - datetime.fromisoformat(status["updated_at"])).total_seconds())
    return {
        "unit_id": config.unit_id,
        "queued": pending_count(),
        "refused": refused_count(),
        "latitude": status.get("latitude", config.default_lat),
        "longitude": status.get("longitude", config.default_lon),
        "detector": status.get("detector"),
        "frames": status.get("frames"),
        "detector_idle_s": round(idle, 1) if idle is not None else None,
    }


def sync_once(send, batch=None):
    # sends queued detections a batch at a time until the queue is empty or the link fails
    # returns how many were confirmed, how many were refused this time, how many still wait and any error
    batch = batch or config.sync_batch
    synced = turned_away = 0

    def result(error=None, left=None):
        return {"synced": synced, "refused": turned_away, "pending": pending_count() if left is None else left,
                "error": error}

    while True:
        rows = pending(batch)
        if not rows:
            return result(left=0)

        try:
            reply = send(rows)
        except send_error as err:
            return result(str(err))
        if not isinstance(reply, dict):
            return result("command center reply was not understood")

        # only trust ids that were in this batch
        sent = {row["uid"] for row in rows}
        accepted = reply.get("accepted")
        accepted = [uid for uid in accepted if isinstance(uid, str) and uid in sent] if isinstance(accepted, list) else []
        rejected = reply.get("rejected")
        rejected = {uid: str(why) for uid, why in rejected.items()
                    if uid in sent and uid not in accepted} if isinstance(rejected, dict) else {}
        mark(accepted, rejected)
        synced += len(accepted)
        turned_away += len(rejected)

        # nothing taken means the same batch would come straight back, so stop and wait instead
        if not accepted and not rejected:
            return result("command center took nothing from the batch")


def retry_wait(failures, interval, max_wait, jitter):
    # seconds to wait after this many failures in a row
    # doubles up to max_wait, the random part spreads retries so many units do not all try at once
    return min(max_wait, interval * 2 ** min(failures, max_doublings)) * (0.5 + jitter / 2)


def run(send, interval=None, max_wait=None, stop=lambda: False, sleep=time.sleep, jitter=random.random,
        send_status=None, send_coverage=None):
    # keeps syncing until stop() is true
    # send_status, when given, also posts the unit's status note each round
    # send_coverage, when given, also posts the ground squares the camera has covered
    # after a failure the wait doubles up to max_wait, and goes back to interval once a sync works
    interval = interval or config.sync_interval
    max_wait = max_wait or config.sync_max_wait
    failures = 0
    last_prune = float("-inf")
    while not stop():
        result = sync_once(send)
        if send_status is not None:
            try:
                send_status(status_note())
            except send_error:
                # the detections are what matter, a missed status note is simply sent next round
                pass
        if send_coverage is not None and not result["error"]:
            sync_coverage(send_coverage)
        if result["refused"]:
            log.warning("the command center refused %s detections, %s are now held on this unit, "
                        "fix the cause then run python -m src.sync --retry-refused", result["refused"], refused_count())
        if result["error"]:
            failures += 1
            wait = retry_wait(failures, interval, max_wait, jitter())
            log.warning("sync failed, %s queued, trying again in %.0f s, %s", result["pending"], wait, result["error"])
        else:
            failures = 0
            wait = interval
            if result["synced"]:
                log.info("sent %s detections", result["synced"])

        # clear out old confirmed detections about once an hour, when the unit is set to
        if config.keep_days and time.monotonic() - last_prune >= 3600:
            removed = prune(config.keep_days)
            if removed:
                log.info("removed %s detections confirmed more than %s days ago", removed, config.keep_days)
            last_prune = time.monotonic()
        sleep(wait)


def main():
    parser = argparse.ArgumentParser(description="send logged detections to the command center")
    parser.add_argument("--url", default=config.sync_url, help="command center address, or set NEUSE_SYNC_URL")
    parser.add_argument("--token", default=config.sync_token, help="secret for the command center, or set NEUSE_SYNC_TOKEN")
    parser.add_argument("--once", action="store_true", help="send what is queued and stop")
    parser.add_argument("--retry-refused", action="store_true",
                        help="put detections the command center refused back in the queue, then sync")
    parser.add_argument("--interval", type=float, default=config.sync_interval, help="seconds between checks")
    args = parser.parse_args()
    if not args.url:
        parser.error("no command center address, use --url or set NEUSE_SYNC_URL")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # make sure the table exists before the first read
    data_logger()
    try:
        send = http_sender(args.url, args.token)
        send_status = http_status_sender(args.url, args.token)
        send_coverage = http_coverage_sender(args.url, args.token)
    except ValueError as err:
        parser.error(str(err))

    if args.retry_refused:
        print(f"put {retry_refused()} refused detections back in the queue")
    for reason, count in refused_reasons():
        log.warning("%s detections are held because the command center refused them, %s", count, reason)

    if args.once or args.retry_refused:
        result = sync_once(send)
        try:
            send_status(status_note())
        except send_error:
            pass
        if not result["error"]:
            sync_coverage(send_coverage)
        print(f"sent {result['synced']}, refused {result['refused']}, still queued {result['pending']}" +
              (f", {result['error']}" if result["error"] else ""))
        return
    log.info("unit %s syncing to %s", config.unit_id, args.url)
    run(send, interval=args.interval, send_status=send_status, send_coverage=send_coverage)


if __name__ == "__main__":
    main()
