NeuSE {Neural Sentinel Edge}

Neural Sentinel Edge helps rescue teams find survivors in thermal camera footage, even where there is no network. Each field unit runs detection on the spot and is designed to send its results to a cloud command center whenever a connection is available.

Built for the Huawei ICT Competition 2026–2027 Innovation Track.

Why it matters

After earthquakes and floods, or in remote areas, the network is often the first thing to go. AI tools that depend on the cloud stop working just when rescuers need them most. NeuSE keeps detection on the device, so it works offline, and is designed to share results with the wider team once a link is back.

Where it stands

NeuSE is an early prototype.

Working now

- Thermal detection using a classical computer vision baseline
- Safe handling of missing, empty, 16 bit, float and odd shaped frames
- A local detection log, with the measured processing time of every frame
- A local dashboard showing logged detections
- Scoring against the real person labels of the HIT-UAV thermal drone dataset
- A script that prepares HIT-UAV for MindYOLO training, with person labels only
- Automated tests

Planned

- A person detector trained with MindYOLO on Huawei Cloud ModelArts, using the HIT-UAV thermal dataset
- On-device inference with MindSpore Lite
- Using the RGB camera to confirm thermal hits and cut false alarms
- Syncing detections to a command center on Huawei Cloud
- Live GPS, in place of the fixed position each unit uses today

The baseline flags any warm area of the right size, so it cannot tell a person from a car engine. It gives no confidence score because there is no model behind it. It stays in the project as the benchmark the trained detector has to beat.

Baseline results on HIT-UAV

Scored on the 579 test images, which hold 2,611 labelled people. A hit counts when its box overlaps a person's box by at least half, the usual standard for detectors. Settings were tuned on the separate validation images before the test images were used.

- Default settings - 3.2% of hits are people and 16% of people are found
- Tuned settings, threshold 200 and minimum area 60 - 4.9% of hits are people and 36% of people are found, with about 31 false hits per image
- At night the tuned baseline finds 42% of people, by day only 13%
- 1,663 of its false hits land on cars, bikes and other vehicles
- mAP@0.5, the main number YOLO tools report, is 0.070 tuned and 0.020 with default settings. The baseline has no confidence score, so its hits are ranked by their brightest pixel for this
- mAP@0.5 is 0.129 at night and 0.004 by day

Most false hits come from the contrast boost, which makes any sunlit or textured patch of ground look hot. These numbers are the bar the trained detector has to clear.

Layout

- `src/pipeline.py` - Thermal detection
- `src/logger.py` - Local detection log in SQLite
- `src/config.py` - Thresholds, unit position and database location
- `src/metrics.py` - Precision and recall against labelled boxes
- `src/hit_uav.py` - Reads the HIT-UAV labels
- `dashboard/` - Local web dashboard
- `scripts/simulate_feed.py` - Sends generated thermal frames through the real pipeline for demos
- `scripts/evaluate_hit_uav.py` - Scores the baseline on HIT-UAV
- `scripts/prepare_hit_uav.py` - Makes the person only copy of HIT-UAV for MindYOLO
- `models/` - Where trained model files will go
- `tests/` - Automated tests
- `docs/ARCHITECTURE.md` - How the parts fit together

Run it

Tested on Python 3.12.

```bash
git clone https://github.com/808kirai/NeuSE.git
cd NeuSE
pip install -r requirements.txt

# run the tests
pytest -v

# terminal 1: generate detections
python scripts/simulate_feed.py --frames 120

# terminal 2: open the dashboard at http://127.0.0.1:5000
python dashboard/app.py
```

To run the pipeline once on a test frame, use `python -m src.pipeline`.

To score the baseline yourself, clone HIT-UAV outside the project. It is about 1.3 GB.

```bash
git clone --depth 1 https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset.git ../datasets/hit-uav

# the tuned result above
python scripts/evaluate_hit_uav.py --dataset ../datasets/hit-uav --split test --threshold 200 --min-area 60

# how the settings were picked, on the validation images
python scripts/evaluate_hit_uav.py --dataset ../datasets/hit-uav --split val --sweep

# save the 20 worst test frames with labelled people in green and baseline hits in red
python scripts/evaluate_hit_uav.py --dataset ../datasets/hit-uav --split test --threshold 200 --min-area 60 --save-vis ../neuse-vis

# the person only copy for MindYOLO training
python scripts/prepare_hit_uav.py --source ../datasets/hit-uav --out ../datasets/hit-uav-person
```

Data

HIT-UAV by Suo et al., Scientific Data 10, 227 (2023), shared under CC BY 4.0 at https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset. It is not stored in this repository.

Settings

- `NEUSE_DB` - Detection database, defaults to `neuse_local.db` in the project folder
- `NEUSE_HOST` - Defaults to `127.0.0.1`, set to `0.0.0.0` to open the dashboard to other devices
- `NEUSE_PORT` - Dashboard port, defaults to `5000`
- `NEUSE_DEBUG` - Off by default, set to `1` for Flask debug mode on a trusted dev machine only
- `NEUSE_SENSOR_RANGE` - Fixed `low,high` scale for 16 bit or float frames, in the camera's own units. Unset, each frame is scaled to its own range, which can make the warmest patch of a cold empty scene look hot. For a camera reporting hundredths of a kelvin, `27315,31315` covers 0 to 40 C

Performance

Every frame is timed, and the time is saved with its detections. The dashboard shows the average. Run `pytest -s` to see the numbers on your own machine.

License

All rights reserved. The code may not be used, copied, modified or distributed without written permission. See `LICENSE`.
