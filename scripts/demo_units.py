"""
runs several field units at once against a command center, for demos
each unit is its own process with its own database, like a real drone
each one flies a search pattern, sends real thermal frames through the person check and syncs when its link is up
links drop at random, so the map shows units going offline, queueing detections and catching up

the frames are real hit-uav images but the flight paths and link drops are made up
each frame's height and camera tilt come from its hit-uav file name, so sightings are placed on the ground
the camera's view angles are not published with the dataset, so those positions are estimates

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
    # at the end of the last line it turns round and flies the same lines back, so it never jumps
    lines = int(area // spacing) + 1
    length = area * lines
    travelled = (step * 10.0) % (2 * length)
    if travelled > length:
        travelled = 2 * length - travelled
    line = min(int(travelled // area), lines - 1)
    along = travelled - line * area
    x = along if line % 2 == 0 else area - along
    y = line * spacing

    east, north = centre_x - area / 2 + x, centre_y - area / 2 + y
    lat = base[0] + north / metres_per_degree
    lon = base[1] + east / (metres_per_degree * math.cos(math.radians(base[0])))
    return lat, lon


def flight_heading(index, units, base, step):
    # the way the unit is facing, in degrees clockwise from north, taken from where it goes next
    here, there = flight_path(index, units, base, step), flight_path(index, units, base, step + 1)
    north = there[0] - here[0]
    east = (there[1] - here[1]) * math.cos(math.radians(base[0]))
    return math.degrees(math.atan2(east, north)) % 360 if (north or east) else 0.0


def frame_pose(image, heading):
    # height and camera tilt from a hit-uav file name like 1_130_90_0_04077.jpg, None for other names
    # in the dataset 90 degrees means the camera points straight down
    try:
        _, altitude, tilt = Path(image).stem.split("_")[:3]
        return {"altitude": float(altitude), "heading": heading, "tilt": float(tilt)}
    except ValueError:
        return None


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
             sleep=time.sleep, say=print, send_coverage=None):
    # flies one unit for frames frames and returns how many sightings it made
    # the database and unit name come from config, which the parent sets for each process
    from src import sync
    from src.config import config
    from src.pipeline import thermal_pipeline

    pipe = thermal_pipeline()
    found = 0
    was_up = True
    for step in range(frames):
        image = images[step % len(images)]
        frame = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        pose = frame_pose(image, flight_heading(index, units, base, step))
        found += len(pipe.process_frames(thermal_frame=frame, position=flight_path(index, units, base, step),
                                         pose=pose))

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
            if send_coverage is not None and not result["error"]:
                sync.sync_coverage(send_coverage)
            if result["synced"]:
                say(f"{config.unit_id} sent {result['synced']}")
        sleep(1 / fps)

    # land with everything sent so the demo ends with a complete picture
    pipe.flush()
    sync.sync_once(send)
    try:
        send_status(sync.status_note())
    except sync.send_error:
        pass
    if send_coverage is not None:
        sync.sync_coverage(send_coverage)
    say(f"{config.unit_id} done, {found} sightings, {sync.pending_count()} still waiting")
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
             radio, args.frames, fps=args.fps, send_coverage=sync.http_coverage_sender(args.center, args.token))


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
