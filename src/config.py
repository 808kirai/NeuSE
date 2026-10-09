# set runtime defaults for local and edge use
# share settings across the pipeline and logger
class config:
    conf_threshold = 0.85
    target_chip = "ascend310"
    default_lat = -1.286389
    default_lon = 36.817223
    db_file = "neuse_local.db"