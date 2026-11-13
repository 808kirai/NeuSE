NeuSE {Neural Sentinel Edge}

Neural Sentinel Edge helps rescue teams find survivors in thermal camera footage, even where there is no network. Each field unit runs detection on the spot and is designed to send its results to a cloud command center whenever a connection is available.

Built for the Huawei ICT Competition 2026–2027 Innovation Track.

Why it matters

After earthquakes and floods, or in remote areas, the network is often the first thing to go. AI tools that depend on the cloud stop working just when rescuers need them most. NeuSE keeps detection on the device, so it works offline, and is designed to share results with the wider team once a link is back.

Where it stands

NeuSE is an early prototype.

Working now

- Person detection on thermal drone footage with a classical detector and a small trained person check, running on an ordinary CPU
- The first warm spot baseline, kept for comparison
- Safe handling of missing, empty, 16 bit, float and odd shaped frames
- A local detection log, with the measured processing time of every frame
- Store and forward sync. Detections queue on the unit and go to the command center whenever a link is up, newest first, with nothing lost or doubled when the link drops
- A command center that receives detections from many units, with a live map page for coordinators showing every unit, whether it is online, its backlog and its sightings of people
- Situation reports for coordinators in plain words, written from a template with no network needed, or by a language model when one is set up
- A demo that flies several units at once on search patterns, with real thermal frames and random link drops
- A local dashboard showing logged detections and how many still wait to be sent
- Scoring against the real person labels of the HIT-UAV thermal drone dataset
- A script that prepares HIT-UAV for MindYOLO training, with person labels only
- Automated tests

Planned

- A person detector trained with MindYOLO on Huawei Cloud ModelArts, using the HIT-UAV thermal dataset
- On-device inference with MindSpore Lite
- Using the RGB camera to confirm thermal hits and cut false alarms
- Running the command center on Huawei Cloud ECS with an RDS database
- Live GPS, in place of the configured or simulated position units use today
- Map tiles from Huawei Map Kit or a self hosted server for real deployments. The page uses the public OpenStreetMap tiles today, which need internet and are meant for light use

How detection works

The person check is the default detector. It works in three steps.

- Find small warm spots by comparing each pixel with the area around it, so sunlit ground stays dark
- Score each spot as person or not with a tiny neural network that runs in plain NumPy. It looks at the direction of edges around the spot, its shape and how warm it is
- Resize each box to fit a person and drop boxes that overlap

The network and box sizes are learned from the HIT-UAV training images in about 3 minutes on a laptop CPU. The weights are 160 KB and live in `models/person_check.npz`. Each hit gets a confidence from 0 to 1.

The first baseline, warm spots, flags any bright area of the right size. It cannot tell a person from a car engine and gives no confidence. Set `NEUSE_DETECTOR=warm-spots` to use it.

Results on HIT-UAV

Scored once on the 579 test images, which hold 2,611 labelled people. Everything was trained on the training images and tuned on the separate validation images first. A hit counts when its box overlaps a person's box by at least half, the usual standard for detectors.

- Person check - mAP@0.5 0.670, 79% of hits are people and 65% of people are found, with under 1 false hit per image
- Warm spots, tuned - mAP@0.5 0.070, 4.9% of hits are people and 36% of people are found, with about 31 false hits per image
- Warm spots, default settings - mAP@0.5 0.020, 3.2% of hits are people and 16% of people are found

The person check scores mAP@0.5 0.709 at night and 0.501 by day. Daytime is harder because the sun warms the ground. It is strongest at 60 to 80 m, with 0.735, and drops to 0.504 at 120 to 130 m where people are only a few pixels across. Only 16 of its false hits land on vehicles, against 1,663 for warm spots.

The time per image is about 24 ms for the person check and 4 ms for warm spots, on a laptop CPU.

How it compares

The HIT-UAV paper reports person AP@0.5 on the same test images for well known detectors. Their tools score slightly differently from ours, so the comparison is close rather than exact.

- YOLOv4 - 89.9%
- SSD-512 - 85.6%
- Faster R-CNN - 75.5%
- Person check - 67.0%
- YOLOv4-tiny - 16.9%
- Warm spots - 7.0%

The person check beats YOLOv4-tiny, the small model meant for limited hardware, but not the full sized GPU models. Those are the bar for the MindYOLO detector.

How sync works

Every detection gets a unique id and the name of its unit when it is logged. It stays marked as waiting in the unit's own database until the command center confirms it.

- `python -m src.sync` checks every 5 seconds and sends waiting detections in batches of 100, newest first, so coordinators see the current picture as soon as a link returns
- Batches are compressed, since field links can be slow and paid by the megabyte
- When the command center cannot be reached, the unit keeps detecting and waits longer between tries, doubling up to 5 minutes with a little randomness so many units do not retry at once
- If a reply gets lost, sending the batch again is safe. The command center keeps each id once
- The command center checks every detection. One it refuses is kept on the unit with the reason and not sent again, so a bad row cannot block the queue
- With a token set, the command center refuses any request without it, since detections show where people are
- Every few seconds the unit also sends a short status note with its position, detector and how many detections still wait. The command center marks a unit offline after 60 seconds without hearing from it

The coordinator page

Open the command center address in a browser. It asks for the token, then shows a map with every unit and every sighting of a person, coloured by how recent it is, and a list of units with their state and backlog. It refreshes every 5 seconds and catches up on backlogs in the order they arrive, so detections sent late after an outage still appear.

A sighting is one detection in one frame. The same person seen in ten frames is ten sightings, so the page and the reports count sightings, never people.

Each detection is placed where the unit was when it took the frame. Working out the exact ground position from the camera angle and altitude is not done yet.

Situation reports

The Write report button on the coordinator page, or `/api/v1/report?minutes=30`, gives a short summary for the chosen window. It covers which units are online, which have gone silent with sightings still waiting to send, and the areas where people were seen. Each area lists its position, sightings, the units involved, how recent it is, the highest confidence and the nearest unit.

- Sightings within 100 m of each other count as one area
- The facts are worked out from the database and always come back with the text, so a coordinator can check them
- With no language model set up, a template writes the report, so it works with no internet
- With `NEUSE_LLM_URL`, `NEUSE_LLM_KEY` and `NEUSE_LLM_MODEL` set, an OpenAI style chat API writes it instead, such as a model on Huawei Cloud. It is told to use only the facts given and never to turn sightings into a head count. If it fails or answers badly, the template is used and the report says so
- Reports are only written when asked for, so a language model is not called on every refresh

Layout

- `src/pipeline.py` - Thermal detection, choosing between the person check and warm spots
- `src/person_check.py` - The person check
- `src/logger.py` - Local detection log in SQLite, which doubles as the queue of detections to send
- `src/sync.py` - Store and forward sync to the command center
- `src/config.py` - Detector choice, thresholds, unit position and database location
- `src/metrics.py` - Precision, recall and mAP against labelled boxes
- `src/hit_uav.py` - Reads the HIT-UAV labels
- `dashboard/` - Local web dashboard
- `command_center/` - Command center API that units sync to, the coordinator map page and situation reports
- `scripts/simulate_feed.py` - Sends made up or real thermal frames through the pipeline for demos
- `scripts/demo_units.py` - Flies several demo units at once against a command center
- `scripts/evaluate_hit_uav.py` - Scores a detector on HIT-UAV
- `scripts/train_person_check.py` - Trains the person check
- `scripts/prepare_hit_uav.py` - Makes the person only copy of HIT-UAV for MindYOLO
- `models/` - Trained model files
- `tests/` - Automated tests
- `docs/ARCHITECTURE.md` - How the parts fit together
- `docs/PROGRESS.md` - Dated record of what was built, measured and decided

Run it

Tested on Python 3.12.

```bash
git clone https://github.com/808kirai/NeuSE.git
cd NeuSE
pip install -r requirements.txt

# run the tests
pytest -v

# terminal 1: generate detections from made up frames
python scripts/simulate_feed.py --frames 120

# terminal 2: open the dashboard at http://127.0.0.1:5000
python dashboard/app.py
```

To run the pipeline once on a test frame, use `python -m src.pipeline`.

To try store and forward on one computer, use three terminals. Stop and start the command center to watch detections wait and then arrive.

```bash
# terminal 1: the command center on port 8000
NEUSE_CC_TOKEN=pick-a-secret python -m command_center.app

# terminal 2: a unit sending what it logs
NEUSE_SYNC_TOKEN=pick-a-secret python -m src.sync --url http://127.0.0.1:8000

# terminal 3: detections from real thermal frames, see the HIT-UAV steps below
python scripts/simulate_feed.py --replay ../datasets/hit-uav/normal_json/test --frames 60

# what the command center has, per unit
curl -H "Authorization: Bearer pick-a-secret" http://127.0.0.1:8000/api/v1/units
```

To watch several units at once, start the command center and open http://127.0.0.1:8000 in a browser, then fly the demo units. Each one is its own process with its own database.

```bash
NEUSE_SYNC_TOKEN=pick-a-secret python scripts/demo_units.py --frames-dir ../datasets/hit-uav/normal_json/test --units 3
```

To check the results yourself, clone HIT-UAV outside the project. It is about 1.3 GB.

```bash
git clone --depth 1 https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset.git ../datasets/hit-uav

# the person check result above
python scripts/evaluate_hit_uav.py --dataset ../datasets/hit-uav --split test

# the warm spots results, tuned and default
python scripts/evaluate_hit_uav.py --dataset ../datasets/hit-uav --split test --detector warm-spots --threshold 200 --min-area 60
python scripts/evaluate_hit_uav.py --dataset ../datasets/hit-uav --split test --detector warm-spots

# how the warm spots settings were picked, on the validation images
python scripts/evaluate_hit_uav.py --dataset ../datasets/hit-uav --split val --sweep

# save the 20 worst test frames with labelled people in green and hits in red
python scripts/evaluate_hit_uav.py --dataset ../datasets/hit-uav --split test --save-vis ../neuse-vis

# train the person check again, which rewrites models/person_check.npz
python scripts/train_person_check.py --dataset ../datasets/hit-uav

# send real test frames through the person check, with the dashboard open
python scripts/simulate_feed.py --replay ../datasets/hit-uav/normal_json/test --frames 60

# the person only copy for MindYOLO training
python scripts/prepare_hit_uav.py --source ../datasets/hit-uav --out ../datasets/hit-uav-person
```

Data

HIT-UAV by Suo et al., Scientific Data 10, 227 (2023), shared under CC BY 4.0 at https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset. It is not stored in this repository.

Settings

- `NEUSE_DETECTOR` - `person-check` by default, or `warm-spots` for the first baseline
- `NEUSE_DB` - Detection database, defaults to `neuse_local.db` in the project folder
- `NEUSE_LAT` and `NEUSE_LON` - Unit position when no GPS position is given, defaults to central Nairobi
- `NEUSE_UNIT_ID` - Name this unit reports to the command center, defaults to `unit-` and the computer's name
- `NEUSE_SYNC_URL` - Command center address for `python -m src.sync`
- `NEUSE_SYNC_TOKEN` - Secret the unit sends to the command center
- `NEUSE_CC_TOKEN` - Secret the command center requires. Unset, it accepts anyone, which is only fine on a closed test network
- `NEUSE_CC_DB` - Command center database, defaults to `command_center.db` in the project folder
- `NEUSE_LLM_URL`, `NEUSE_LLM_KEY` and `NEUSE_LLM_MODEL` - Optional OpenAI style chat API for situation reports, the base address ending before `/chat/completions`. Unset, reports come from the template
- `NEUSE_CC_HOST` and `NEUSE_CC_PORT` - Command center address, defaults to `127.0.0.1` and `8000`
- `NEUSE_HOST` - Defaults to `127.0.0.1`, set to `0.0.0.0` to open the dashboard to other devices
- `NEUSE_PORT` - Dashboard port, defaults to `5000`
- `NEUSE_DEBUG` - Off by default, set to `1` for Flask debug mode on a trusted dev machine only
- `NEUSE_SENSOR_RANGE` - Fixed `low,high` scale for 16 bit or float frames, in the camera's own units. Unset, each frame is scaled to its own range, which can make the warmest patch of a cold empty scene look hot. For a camera reporting hundredths of a kelvin, `27315,31315` covers 0 to 40 C

Performance

Every frame is timed, and the time is saved with its detections. The dashboard shows the average. Run `pytest -s` to see the numbers on your own machine, or the scoring script for real footage.

License

All rights reserved. The code may not be used, copied, modified or distributed without written permission. See `LICENSE`.
