import os
from pathlib import Path

# project root so paths work from any folder
project_root = Path(__file__).resolve().parent.parent


def parse_range(text):
    # reads "low,high" into a pair of floats, or None when unset
    if not text:
        return None
    low, high = (float(v) for v in text.split(","))
    if high <= low:
        raise ValueError("NEUSE_SENSOR_RANGE needs low,high with high above low")
    return low, high


# shared settings for the pipeline, logger and dashboard
# values are read when used, so tests can swap them safely
class config:
    # which detector the pipeline runs
    # person-check finds small warm spots and scores each one with a small trained check, the default
    # warm-spots is the first baseline, any bright area of the right size, kept for comparison and the demo feed
    detector = os.environ.get("NEUSE_DETECTOR", "person-check")

    # weights for the person check, made by scripts/train_person_check.py
    person_check_path = project_root / "models" / "person_check.npz"

    # lowest score a trained model needs to report a detection
    # not used yet, the person check keeps its own threshold tuned on the validation images
    conf_threshold = 0.85

    # contrast boost applied before the threshold
    # a higher clip limit boosts harder, smaller tiles boost more locally
    # people in drone footage are only about 14 by 15 pixels, so strong settings can blur them into the background
    clahe_clip_limit = 2.0
    clahe_tile_grid = 8

    # pixels brighter than this after contrast boost count as warm
    thermal_threshold = 200

    # warm areas outside this size in pixels are treated as noise
    min_blob_area = 100
    max_blob_area = 50000

    # fixed scale for 16 bit or float frames as low,high in the camera's own units
    # unset means each frame is scaled to its own range, so the warmest patch of a cold empty scene can look hot
    # e.g. NEUSE_SENSOR_RANGE=27315,31315 for a camera reporting hundredths of a kelvin covers 0 to 40 c
    sensor_range = parse_range(os.environ.get("NEUSE_SENSOR_RANGE"))

    # fixed unit position until a gps feed is added
    default_lat = -1.286389
    default_lon = 36.817223

    # database shared by the pipeline and dashboard
    # set NEUSE_DB to point it at a different file
    db_path = Path(os.environ.get("NEUSE_DB", project_root / "neuse_local.db"))
