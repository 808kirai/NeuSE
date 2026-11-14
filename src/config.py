import os
import re
import socket
from pathlib import Path

# project root so paths work from any folder
project_root = Path(__file__).resolve().parent.parent

# names the command center accepts for a unit
unit_pattern = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def parse_range(text):
    # reads "low,high" into a pair of floats, or None when unset
    if not text:
        return None
    low, high = (float(v) for v in text.split(","))
    if high <= low:
        raise ValueError("NEUSE_SENSOR_RANGE needs low,high with high above low")
    return low, high


def parse_fov(text):
    # reads "across,down" camera view angles in degrees
    across, down = (float(v) for v in text.split(","))
    if not (0 < across < 180 and 0 < down < 180):
        raise ValueError("NEUSE_CAMERA_FOV needs two angles between 0 and 180")
    return across, down


def default_unit_id():
    # the computer's name with anything the command center would refuse swapped for a dash
    name = re.sub(r"[^A-Za-z0-9._-]", "-", socket.gethostname().lower())[:59]
    return f"unit-{name or 'field'}"


def check_unit_id(unit_id):
    # stops a unit starting with a name the command center would refuse
    # without this every detection would be turned away when it is sent
    if not isinstance(unit_id, str) or not unit_pattern.match(unit_id):
        raise ValueError(f"unit id {unit_id!r} must be 1 to 64 letters, digits, dots, dashes or underscores, "
                         "set NEUSE_UNIT_ID")
    return unit_id


# shared settings for the pipeline, logger and dashboard
# most values are read when used, so tests can swap them safely
# the detector choice and contrast settings are read when a pipeline is built
class config:
    # which detector the pipeline runs
    # person-check finds small warm spots and scores each one with a small trained check, the default
    # warm-spots is the first baseline, any bright area of the right size, kept for comparison and the demo feed
    # person-cnn scores warm spots with a small convolutional network trained with mindspore on a laptop
    # mindyolo runs a model trained on modelarts, see training/README.md
    detector = os.environ.get("NEUSE_DETECTOR", "person-check")

    # weights for the person check, made by scripts/train_person_check.py
    person_check_path = project_root / "models" / "person_check.npz"

    # the person check was trained on 640 by 512 frames from 60 to 130 m, where a person is about 15 pixels tall
    # frames are resized by this factor before the person check so people land near that size
    # e.g. 0.5 for a camera where people look twice as big, boxes are scaled back afterwards
    frame_scale = float(os.environ.get("NEUSE_FRAME_SCALE", 1.0))

    # weights for the person cnn, made with mindspore by scripts/train_person_cnn.py
    person_cnn_path = project_root / "models" / "person_cnn.npz"

    # the mindyolo model converted for mindspore lite, used when the detector is mindyolo
    mindyolo_path = Path(os.environ.get("NEUSE_MINDYOLO_MODEL", project_root / "models" / "person_mindyolo.ms"))
    # the score a mindyolo box needs to be reported, pick it on the validation images once the model is trained
    mindyolo_threshold = float(os.environ.get("NEUSE_MINDYOLO_THRESHOLD", 0.25))

    # most warm spots the person check scores in one frame, the warmest are kept
    # real drone frames have a few hundred, the cap keeps a noisy frame from stalling the unit
    max_candidates = 1000

    # contrast boost applied before the threshold in the warm-spots detector
    # a higher clip limit boosts harder, smaller tiles boost more locally
    # people in drone footage are only about 14 by 15 pixels, so strong settings can blur them into the background
    clahe_clip_limit = 2.0
    clahe_tile_grid = 8

    # warm-spots, pixels brighter than this after contrast boost count as warm
    thermal_threshold = 200

    # warm-spots, warm areas outside this size in pixels are treated as noise
    min_blob_area = 100
    max_blob_area = 50000

    # fixed scale for 16 bit or float frames as low,high in the camera's own units
    # unset means each frame is scaled to its own range, so the warmest patch of a cold empty scene can look hot
    # e.g. NEUSE_SENSOR_RANGE=27315,31315 for a camera reporting hundredths of a kelvin covers 0 to 40 c
    sensor_range = parse_range(os.environ.get("NEUSE_SENSOR_RANGE"))

    # unit position used when the caller gives none, until a gps feed is added
    # set NEUSE_LAT and NEUSE_LON to place a unit somewhere else
    default_lat = float(os.environ.get("NEUSE_LAT", -1.286389))
    default_lon = float(os.environ.get("NEUSE_LON", 36.817223))

    # how wide the camera sees across and down the frame in degrees
    # only used to place a sighting on the ground when the caller gives the unit's height, heading and camera tilt
    # the default is a guess for a common thermal lens, set it to the real camera's
    camera_fov = parse_fov(os.environ.get("NEUSE_CAMERA_FOV", "45,37"))
    # sightings worked out to be further than this from the unit are placed at the unit instead
    max_ground_range = 1500.0

    # seconds between the unit's status notes, which tell the command center it is alive and where it is
    status_every = 5.0

    # database shared by the pipeline and dashboard
    # set NEUSE_DB to point it at a different file
    db_path = Path(os.environ.get("NEUSE_DB", project_root / "neuse_local.db"))

    # days to keep detections on the unit after the command center has confirmed them, 0 keeps them forever
    # a detection takes about 200 bytes, so a million of them is about 200 mb
    keep_days = float(os.environ.get("NEUSE_KEEP_DAYS", 0))

    # name this unit reports to the command center, defaults to the computer's name
    unit_id = check_unit_id(os.environ.get("NEUSE_UNIT_ID") or default_unit_id())

    # command center address for store and forward sync, unset keeps everything local
    # e.g. NEUSE_SYNC_URL=https://command.example.org
    sync_url = os.environ.get("NEUSE_SYNC_URL")
    # secret the command center expects, sent as a bearer token
    sync_token = os.environ.get("NEUSE_SYNC_TOKEN")
    # a token is only sent over plain http to this same computer, set NEUSE_SYNC_INSECURE=1 to allow it anywhere
    # anyone on the network path can read a token and people's positions sent over plain http
    sync_insecure = os.environ.get("NEUSE_SYNC_INSECURE") == "1"
    # detections sent per request, small enough for a weak satellite or radio link
    sync_batch = 100
    # seconds between checks while the link is up, and the longest wait after failures
    sync_interval = 5.0
    sync_max_wait = 300.0
