import os
from pathlib import Path

# project root so paths work from any folder
project_root = Path(__file__).resolve().parent.parent


# shared settings for the pipeline, logger and dashboard
# values are read when used, so tests can swap them safely
class config:
    # lowest score a trained model needs to report a detection
    # the classical baseline has no scores so it skips this check
    conf_threshold = 0.85

    # pixels brighter than this after contrast boost count as warm
    thermal_threshold = 200

    # warm areas outside this pixel range are treated as noise
    min_blob_area = 100
    max_blob_area = 50000

    # fixed unit position until a gps feed is added
    default_lat = -1.286389
    default_lon = 36.817223

    # database shared by the pipeline and dashboard
    # set NEUSE_DB to point it at a different file
    db_path = Path(os.environ.get("NEUSE_DB", project_root / "neuse_local.db"))
