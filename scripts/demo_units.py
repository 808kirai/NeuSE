"""
runs several field units at once against a command center, for demos
each unit is its own process with its own database, like a real drone
each one flies a search pattern, sends real thermal frames through the person check and syncs when its link is up
links drop at random, so the map shows units going offline, queueing detections and catching up

the frames are real hit-uav images but the flight paths and link drops are made up

usage, with the command center running in another terminal
    python scripts/demo_units.py --frames-dir ../datasets/hit-uav/normal_json/test --units 3
"""
import argparse
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# metres in one degree of latitude
metres_per_degree = 111320


def flight_path(index, units, base, step, area=300.0, spacing=40.0, distance=600.0):
    # position of unit index after step frames
    # units start around the base and each sweeps its own square in back and forth lines, like a search
    angle = 2 * math.pi * index / max(units, 1)
    centre_x, centre_y = distance * math.cos(angle), distance * math.sin(angle)

    # how far along the back and forth lines the unit is, 10 m per frame
    travelled = (step * 10.0) % (area * (area / spacing))
    line = int(travelled // area)
    along = travelled % area
    x = along if line % 2 == 0 else area - along
    y = line * spacing

    east, north = centre_x - area / 2 + x, centre_y - area / 2 + y
    lat = base[0] + north / metres_per_degree
    lon = base[1] + east / (metres_per_degree * math.cos(math.radians(base[0])))
    return lat, lon


class link:
    # a radio link that drops now and then and comes back after a while
    def __init__(self, rng, drop_chance, shortest=10, longest=40):
        self.rng = rng
        self.drop_chance = drop_chance
        self.shortest, self.longest = shortest, longest
        self.down_for = 0

    def tick(self):
        # moves one frame on and says whether the link is up
        if self.down_for > 0:
            self.down_for -= 1
            return False
        if self.rng.random() < self.drop_chance:
            self.down_for = self.rng.randint(self.shortest, self.longest)
            return False
        return True


def run_unit(images, index, units, base, send, send_status, radio, frames, fps=1.0, sync_every=3,
             sleep=time.sleep, say=print):
    # flies one unit for frames frames and returns how many detections it made
    # the database and unit name come from config, which the parent sets for each process
    from src import sync
    from src.config import config
    from src.pipeline import dual_spectral_pipeline

    pipe = dual_spectral_pipeline()
    found = 0
    was_up = True
    for step in range(frames):
        frame = cv2.imread(str(images[step % len(images)]), cv2.IMREAD_GRAYSCALE)
        found += len(pipe.process_frames(thermal_frame=frame, position=flight_path(index, units, base, step)))

        up = radio.tick()
        if up != was_up:
            say(f"{config.unit_id} link {'back' if up else 'lost'}, {sync.pending_count()} waiting")
            was_up = up
        if up and step % sync_every == 0:
            result = sync.sync_once(send)
            try:
                send_status(sync.status_note())
            except sync.send_error:
                pass
            if result["synced"]:
                say(f"{config.unit_id} sent {result['synced']}")
        sleep(1 / fps)

    # land with everything sent so the demo ends with a complete picture
    sync.sync_once(send)
    try:
        send_status(sync.status_note())
    except sync.send_error:
        pass
    say(f"{config.unit_id} done, {found} people found, {sync.pending_count()} still waiting")
    return found


def worker(args):
    # one unit in its own process
    from src import sync
    images = sorted(Path(args.frames_dir).glob("*.jpg"))[args.index::args.units]
    if not images:
        sys.exit(f"no jpg images in {args.frames_dir}")
    radio = link(random.Random(args.seed + args.index), args.drop_chance)
    run_unit(images, args.index, args.units, (args.lat, args.lon),
             sync.http_sender(args.center, args.token), sync.http_status_sender(args.center, args.token),
             radio, args.frames, fps=args.fps)


def main():
    parser = argparse.ArgumentParser(description="run several demo field units against a command center")
    parser.add_argument("--frames-dir", required=True, help="folder of thermal images, like the hit-uav test images")
    parser.add_argument("--center", default="http://127.0.0.1:8000", help="command center address")
    parser.add_argument("--token", default=os.environ.get("NEUSE_SYNC_TOKEN"), help="command center token")
    parser.add_argument("--units", type=int, default=3, help="how many units to fly")
    parser.add_argument("--frames", type=int, default=120, help="frames each unit processes")
    parser.add_argument("--fps", type=float, default=1.0, help="frames per second per unit")
    parser.add_argument("--drop-chance", type=float, default=0.03, help="chance per frame that a link drops")
    parser.add_argument("--lat", type=float, default=-1.286389, help="centre of the search area")
    parser.add_argument("--lon", type=float, default=36.817223, help="centre of the search area")
    parser.add_argument("--data-dir", default=str(Path(tempfile.gettempdir()) / "neuse-demo"),
                        help="where each unit keeps its database")
    parser.add_argument("--seed", type=int, default=0, help="random seed for the link drops")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--index", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker(args)
        return

    # a fresh database per unit, so every run starts with empty queues
    data = Path(args.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    processes = []
    for index in range(args.units):
        db = data / f"unit-{index + 1}.db"
        for leftover in (db, Path(f"{db}-wal"), Path(f"{db}-shm")):
            leftover.unlink(missing_ok=True)
        env = {**os.environ, "NEUSE_DB": str(db), "NEUSE_UNIT_ID": f"unit-{index + 1}"}
        command = [sys.executable, __file__, "--worker", "--index", str(index)] + [
            a for a in sys.argv[1:] if a != "--worker"]
        processes.append(subprocess.Popen(command, env=env))
    print(f"flying {args.units} units, open {args.center} to watch")
    sys.exit(max(p.wait() for p in processes))


if __name__ == "__main__":
    main()
