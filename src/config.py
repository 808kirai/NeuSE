# basic runtime config for local development and edge deployment
# local config file for NeuSE project, used by src/pipeline.py and src/logger.py
class config:
    conf_threshold = 0.85
    target_chip = "ascend310"
    default_lat = -1.286389
    default_lon = 36.817223
    db_file = "neuse_local.db"