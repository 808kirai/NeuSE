"""
situation reports for rescue coordinators
the facts come straight from the database, then a template or a language model writes them up in plain words

a sighting is one detection in one frame, so the same person seen in ten frames is ten sightings
reports count sightings and areas, never people, so they do not overstate how many were found

the language model is optional and reached through an openai style chat api, such as huawei cloud's model service
set NEUSE_LLM_URL, NEUSE_LLM_KEY and NEUSE_LLM_MODEL to use one, the template is used whenever it cannot be reached
"""
import json
import math
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

from command_center import store

# sightings closer than this in metres count as the same area
area_radius = 100.0
# areas listed in a report, the rest are counted
max_areas = 8
# areas handed to a language model
model_areas = 20
metres_per_degree = 111320


def distance(a, b):
    # metres between two latitude and longitude pairs, close enough for areas a few km wide
    mid = math.radians((a[0] + b[0]) / 2)
    north = (a[0] - b[0]) * metres_per_degree
    east = (a[1] - b[1]) * metres_per_degree * math.cos(mid)
    return math.hypot(north, east)


def group_areas(sightings, radius=area_radius):
    # puts each sighting into the first area whose centre is within radius, or starts a new area
    # newest sightings go first so each area is centred on where people were seen most recently
    areas = []
    for s in sorted(sightings, key=lambda s: s["timestamp_utc"], reverse=True):
        point = (s["latitude"], s["longitude"])
        area = next((a for a in areas if distance(a["centre"], point) <= radius), None)
        if area is None:
            area = {"centre": point, "points": []}
            areas.append(area)
        area["points"].append(s)
        # move the centre to the average of its sightings
        area["centre"] = (sum(p["latitude"] for p in area["points"]) / len(area["points"]),
                          sum(p["longitude"] for p in area["points"]) / len(area["points"]))
    return areas


def count(number, word):
    # "1 sighting" and "2 sightings"
    return f"{number} {word}" if number == 1 else f"{number} {word}s"


def ago(minutes):
    return "just now" if minutes == 0 else f"{minutes} min ago"


def minutes_since(iso, now):
    return max(0, round((now - datetime.fromisoformat(iso)).total_seconds() / 60))


def gather(db, minutes=30, now=None, online_window=60, detecting_window=30):
    # the facts a report is written from, all measured, nothing guessed
    now = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    start = (now - timedelta(minutes=minutes)).isoformat()
    sightings = store.sightings_between(db, start, now.isoformat())

    units = []
    for unit in store.units(db):
        state = store.unit_state(unit, now, online_window, detecting_window)
        quiet = state["detector_idle_s"]
        units.append({
            "unit_id": unit["unit_id"],
            "online": state["online"],
            "minutes_since_contact": minutes_since(unit["last_contact"], now),
            # false when the unit is in contact but its detector has stopped handling frames, None when unknown
            "detecting": state["detecting"],
            "detector_quiet_minutes": None if quiet is None else round(quiet / 60),
            # sightings still on the unit, the command center cannot see them yet
            "waiting_on_unit": unit["queued"] or 0,
            # sightings the command center turned away, someone has to look at that unit
            "refused_on_unit": unit["refused"] or 0,
            "sightings_in_window": sum(1 for s in sightings if s["unit_id"] == unit["unit_id"]),
            "latitude": unit["latitude"],
            "longitude": unit["longitude"],
        })

    areas = []
    for area in group_areas(sightings):
        points = area["points"]
        confidences = [p["confidence"] for p in points if p["confidence"] is not None]
        nearest = min(units, key=lambda u: distance(area["centre"], (u["latitude"], u["longitude"])), default=None)
        areas.append({
            "latitude": round(area["centre"][0], 5),
            "longitude": round(area["centre"][1], 5),
            "sightings": len(points),
            "units": sorted({p["unit_id"] for p in points}),
            "minutes_since_latest": minutes_since(max(p["timestamp_utc"] for p in points), now),
            "highest_confidence": round(max(confidences), 2) if confidences else None,
            "nearest_unit": nearest["unit_id"] if nearest else None,
            "nearest_unit_metres": round(distance(area["centre"], (nearest["latitude"], nearest["longitude"])))
            if nearest else None,
        })
    # most recent first, then the busiest
    areas.sort(key=lambda a: (a["minutes_since_latest"], -a["sightings"]))

    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "window_minutes": minutes,
        "sightings": len(sightings),
        # sightings placed at the unit itself because it gave no height, heading and camera tilt
        # the person can be a few hundred metres from such a position
        "sightings_placed_at_unit": sum(1 for s in sightings if s["position_source"] != "projected"),
        "areas": areas,
        "units": units,
    }


def write_template(facts):
    # a plain report that needs no network, used by default and whenever the language model fails
    lines = []
    clock = datetime.fromisoformat(facts["generated_at"]).strftime("%H:%M UTC")
    lines.append(f"Situation at {clock}, covering the last {facts['window_minutes']} minutes.")

    units = facts["units"]
    online = [u for u in units if u["online"]]
    lines.append(f"{len(online)} of {count(len(units), 'unit')} online.")
    for unit in units:
        if not unit["online"]:
            note = f"{unit['unit_id']} has been silent for {unit['minutes_since_contact']} min"
            if unit["waiting_on_unit"]:
                note += f" and had {count(unit['waiting_on_unit'], 'sighting')} still waiting to send, so more may arrive"
            lines.append(note + ".")
        elif unit["waiting_on_unit"]:
            lines.append(f"{unit['unit_id']} is online and still sending {count(unit['waiting_on_unit'], 'sighting')}.")
        if unit["online"] and unit["detecting"] is False:
            lines.append(f"{unit['unit_id']} is in contact but its detector has been quiet for "
                         f"{unit['detector_quiet_minutes']} min, so it is not searching.")
        if unit["refused_on_unit"]:
            lines.append(f"{unit['unit_id']} holds {count(unit['refused_on_unit'], 'sighting')} the command center "
                         "refused, that unit needs checking.")

    if not facts["areas"]:
        lines.append("No sightings of people in this window.")
        return "\n".join(lines)

    lines.append(f"{count(facts['sightings'], 'sighting')} in {count(len(facts['areas']), 'area')}. "
                 "One person can be seen in many frames, so sightings are not a head count.")
    for number, area in enumerate(facts["areas"][:max_areas], start=1):
        detail = (f"Area {number} at {area['latitude']}, {area['longitude']} - {count(area['sightings'], 'sighting')} "
                  f"by {', '.join(area['units'])}, latest {ago(area['minutes_since_latest'])}")
        if area["highest_confidence"] is not None:
            detail += f", highest confidence {area['highest_confidence']}"
        if area["nearest_unit"]:
            detail += f", {area['nearest_unit']} is {area['nearest_unit_metres']} m away"
        lines.append(detail + ".")
    hidden = len(facts["areas"]) - max_areas
    if hidden > 0:
        lines.append(f"{count(hidden, 'more area')} on the map.")
    at_unit = facts["sightings_placed_at_unit"]
    if at_unit:
        lines.append(f"{count(at_unit, 'sighting')} could only be placed at the unit's own position, "
                     "the person can be a few hundred metres from it.")
    return "\n".join(lines)


instructions = (
    "You write short situation reports for search and rescue coordinators. "
    "Use only the facts in the JSON you are given and never add numbers, places or events that are not there. "
    "A sighting is one detection in one camera frame, the same person can appear in many, "
    "so never turn sightings into a count of people. "
    "Start with what needs attention first, such as recent areas with many sightings, silent units with "
    "sightings still waiting to send, units in contact whose detector has stopped and units holding refused "
    "sightings. Say so when sightings are only placed at the unit's own position. "
    "Give coordinates as they appear. Use plain sentences, at most 150 words."
)


def for_model(facts):
    # the facts handed to the language model, with the list of areas kept short so the request stays small
    listed = facts["areas"][:model_areas]
    return {**facts, "areas": listed, "areas_not_listed": len(facts["areas"]) - len(listed)}


def write_with_model(facts, url, key, model, timeout=20):
    # asks an openai style chat api to write the report, raises on any failure so the caller can fall back
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": instructions},
                     {"role": "user", "content": json.dumps(for_model(facts))}],
        "temperature": 0.2,
        "max_tokens": 400,
    }).encode("utf-8")
    request = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        reply = json.loads(response.read())
    text = reply["choices"][0]["message"]["content"].strip()
    if not text:
        raise ValueError("empty reply")
    return text


def report(db, minutes=30, llm=None, now=None, online_window=60, detecting_window=30):
    # the full report, the text plus the facts it came from
    # llm is (url, key, model) or None, the facts always come with the text so coordinators can check them
    facts = gather(db, minutes, now, online_window, detecting_window)
    if llm and all(llm):
        try:
            return {"source": "model", "text": write_with_model(facts, *llm), "facts": facts}
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError, TypeError) as err:
            text = write_template(facts)
            return {"source": "template", "text": text, "facts": facts,
                    "note": f"the language model could not be used, {type(err).__name__}, so this came from the template"}
    return {"source": "template", "text": write_template(facts), "facts": facts}
