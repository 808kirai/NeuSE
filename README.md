NeuSE {Neural Sentinel Edge}

Neural Sentinel Edge helps rescue teams find survivors in thermal camera footage, even where there is no network. Each field unit runs detection on the spot and is designed to send its results to a cloud command center whenever a connection is available.

Built for the Huawei ICT Competition 2026–2027 Innovation Track.

Why it matters

After earthquakes and floods, or in remote areas, the network is often the first thing to go. AI tools that depend on the cloud stop working just when rescuers need them most. NeuSE keeps detection on the device, so it works offline, and is designed to share results with the wider team once a link is back.

Where it stands

NeuSE is an early prototype.

Working now

- Person detection on thermal drone footage with a classical detector and a small trained person check, running on an ordinary CPU, with a confidence that is calibrated to mean what it says
- A second person detector, a small CNN trained with MindSpore and exported to MindIR, that runs on the unit with plain NumPy. It does not beat the person check yet, so the person check stays the default
- The first warm spot baseline, kept for comparison
- Safe handling of missing, empty, 16 bit, float and odd shaped frames, and of database errors, which no longer stop detection
- Sightings placed on the ground when the unit gives its height, heading and camera tilt, and marked as placed at the unit when it does not
- A local detection log, with the measured processing time of every frame
- Store and forward sync. Detections queue on the unit and go to the command center whenever a link is up, newest first, with nothing lost or doubled when the link drops. One the command center refuses is held, counted and shown
- A command center that receives detections from many units, with a live map page for coordinators showing every unit, whether it is online, whether it is still searching, its backlog and its sightings of people
- A map of the ground each unit's camera has searched, so coordinators can see where nobody has looked yet
- Alerts for coordinators when an area gets several strong sightings, with text messages through Huawei Cloud SMN when it is set up
- Correction of unit clocks, so a field computer with a wrong clock does not put sightings in the wrong hour
- Separate tokens for coordinators and for each unit, so a lost unit cannot read where people are
- Situation reports for coordinators in plain words, written from a template with no network needed, or by a language model when one is set up
- A demo that flies several units at once on search patterns, with real thermal frames and random link drops
- A local dashboard showing logged detections, how many still wait to be sent and how many were refused
- Scoring against the real person labels of the HIT-UAV thermal drone dataset, both on its own test images and on flights held out of training
- A script that prepares HIT-UAV for MindYOLO training, with person labels only
- Automated tests

Planned

- A person detector trained with MindYOLO on Huawei Cloud ModelArts, using the HIT-UAV thermal dataset. Three training configs, the guide in `training/README.md` and the slot that runs it on the unit with MindSpore Lite, on the CPU or an Ascend NPU, are ready
- Running the unit on an Orange Pi AIpro with its Ascend 310B NPU
- Using the RGB camera to confirm thermal hits and cut false alarms
- Running the command center on Huawei Cloud ECS with an RDS for MySQL database, behind HTTPS. The Docker setup, MySQL support and the guide in `docs/DEPLOY.md` are ready, and checked automatically against MySQL 8.0 and Docker on every push
- Live GPS, height, heading and camera tilt from the drone, in place of the values a caller passes in today
- A map background from Huawei Map Kit or a self hosted server for real deployments. The page uses the public OpenStreetMap tiles unless `NEUSE_TILE_URL` is set, which need internet and are meant for light use

How detection works

The person check is the default detector. It works in three steps.

- Find small warm spots by comparing each pixel with the area around it, so sunlit ground stays dark
- Score each spot as person or not with a tiny neural network that runs in plain NumPy. It looks at the direction of edges around the spot, its shape and how warm it is
- Resize each box to fit a person and drop boxes that overlap

The network and box sizes are learned from the HIT-UAV training images in about 3 minutes on a laptop CPU. The weights are 160 KB and live in `models/person_check.npz`.

Each hit gets a confidence from 0 to 1. It is calibrated on the validation images, so it is meant to match how often such hits are people. On the test images, hits shown as 0.9 or more were people 92% of the time and hits shown as 0.5 to 0.6 were people 69% of the time. Across the range it is within about 15 points and errs on the low side. A hit is reported from 0.32 up, the point that gave the best balance of finding people and avoiding false hits on the validation images.

The person check is tied to the footage it learned from, 640 by 512 frames taken from 60 to 130 m, where a person is about 15 pixels tall. For a camera where people look bigger or smaller, set `NEUSE_FRAME_SCALE` to resize frames first. It warns once when frames are a different size. It scores at most 1,000 warm spots per frame, the warmest ones, so a frame full of noise cannot stall the unit.

The person CNN, `NEUSE_DETECTOR=person-cnn`, is a second learned detector built with MindSpore. It finds warm spots at two sizes, which finds 96.7% of labelled people against 93.4% for one size, then looks at a 32 by 32 crop around each spot with three small convolution layers. It sees two channels, the frame and the warm spot map, plus eleven numbers about the spot's shape and warmth, and learns to score the spot and fit its box. It is trained in about 20 minutes on a laptop CPU with `scripts/train_person_cnn.py`, which also exports it to MindIR. On the unit it runs in plain NumPy, checked against MindSpore to agree to about 4 in a million. If its weights are missing it falls back to the person check.

The first baseline, warm spots, flags any bright area of the right size. It cannot tell a person from a car engine and gives no confidence. Set `NEUSE_DETECTOR=warm-spots` to use it.

Results on HIT-UAV

Scored once on the 579 test images, which hold 2,611 labelled people. Everything was trained on the training images and tuned on the separate validation images first. A hit counts when its box overlaps a person's box by at least half, the usual standard for detectors.

- Person check - mAP@0.5 0.670, 79% of hits are people and 65% of people are found, with under 1 false hit per image
- Person CNN - mAP@0.5 0.659, 71% of hits are people and 67% of people are found, with about 1.2 false hits per image, at about 100 ms per image
- Warm spots, tuned - mAP@0.5 0.070, 4.9% of hits are people and 36% of people are found, with about 31 false hits per image
- Warm spots, default settings - mAP@0.5 0.020, 3.2% of hits are people and 16% of people are found

The person check scores mAP@0.5 0.710 at night and 0.504 by day. Daytime is harder because the sun warms the ground. It is strongest at 60 to 80 m, with 0.734, and drops to 0.506 at 120 to 130 m where people are only a few pixels across. Only 16 of its false hits land on vehicles, against 1,663 for warm spots.

The person CNN was also behind on the validation images, 0.625 against 0.643, so the person check stays the default. The CNN finds slightly more people but makes more false hits, and is weakest at 120 to 130 m, where only 53% of its hits are people. Its value now is that it shows the MindSpore path working end to end, training, export to MindIR and the same answers on the unit.

The time per image is about 30 ms for the person check and 4 ms for warm spots, on a laptop CPU. A frame crowded with warm spots can take over 100 ms.

Results on flights it has not seen

HIT-UAV's training, validation and test images come from the same flights. Every test image is within 30 frame numbers of a training image from the same flight, and 84% are within 5. So the score above shows how the detector does on those scenes, not on a new place.

To measure that, `scripts/evaluate_unseen_flights.py` groups all 2,898 images by flight setting, the time of day, height and camera angle in the file name, and deals the 105 settings into 5 parts. Each part is scored by a detector trained without it and tuned on another part.

- Mean mAP@0.5 0.520 across the 5 parts, from 0.487 to 0.569
- 63% of hits are people and 54% of people are found, with 1.4 false hits per image

Each of these detectors learned from about three fifths of the images, fewer than the 2,029 the saved one learned from. Flight settings are the best grouping the file names allow. Two settings can still be the same place seen from a different height.

What these numbers do not show

- People on bikes. HIT-UAV labels a bike and its rider as one bicycle box and gives the rider no person box. Of 3,630 bicycle boxes in the training images only 3 have a person box on them. So the person check has learned to treat riders as not people, and the scores count that as correct
- Other cameras, heights and places. Every image is from one camera type over a few sites
- A fully blind test. The worst test frames of the warm spot baseline were looked at before the person check was designed, which is where the idea of comparing each pixel with its surroundings came from. The numbers were still scored once, after all tuning

How it compares

The HIT-UAV paper reports person AP@0.5 on the same test images for well known detectors. Their tools score slightly differently from ours, so the comparison is close rather than exact. They were trained and tested on the same flights too.

- YOLOv4 - 89.9%
- SSD-512 - 85.6%
- Faster R-CNN - 75.5%
- Person check - 67.0%
- Person CNN - 65.9%
- YOLOv4-tiny - 16.9%
- Warm spots - 7.0%

Later papers report YOLOv8 at about 92% person AP@0.5 on HIT-UAV, YOLOv8s at 640 pixels reaching 92.3%. That is the bar for the MindYOLO detector, and the reason it is the next step. The person check is well short of the full sized detectors. It scores above the paper's YOLOv4-tiny run, but that run is a poor guide to what small detectors can do. Its person score is far below its own 83.6% on cars, which points to that one run going wrong on small people rather than to a limit of small detectors. What the person check does show is how far a method with no GPU and no deep learning framework can get.

How sync works

Every detection gets a unique id and the name of its unit when it is logged. It stays marked as waiting in the unit's own database until the command center confirms it.

- `python -m src.sync` checks every 5 seconds and sends waiting detections in batches of 100, newest first, so coordinators see the current picture as soon as a link returns
- Batches are compressed, since field links can be slow and paid by the megabyte
- When the command center cannot be reached, the unit keeps detecting and waits longer between tries, doubling up to 5 minutes with a little randomness so many units do not retry at once
- If a reply gets lost, sending the batch again is safe. The command center keeps each id once
- The command center checks every detection. One it refuses is held on the unit with the reason and left out of the queue, so a bad row cannot block the rest. It is not marked as sent. The unit logs a warning, counts it on its dashboard and reports the count to the command center. Once the cause is fixed, `python -m src.sync --retry-refused` puts held detections back in the queue
- A unit will not start with a name the command center would refuse, since every one of its detections would then be turned away
- Each batch carries the unit's own clock reading. The command center compares it with its own clock and corrects the detection times by the difference when they are more than 5 seconds apart. Field computers often have no battery clock, and without this a unit two hours fast or years behind would have its sightings left out of every report. The time the unit gave is kept next to the correction. If a unit's clock is put right part way through a flight, sightings from before that are still corrected by the wrong amount
- Every few seconds the unit also sends a short status note with its position, detector, how many detections wait or were refused, and how long ago its detector last handled a frame. The command center marks a unit offline after 60 seconds without hearing from it, and as not searching when it is in contact but its detector has been quiet for 30 seconds
- To stop the unit's database growing for ever, set `NEUSE_KEEP_DAYS`. Detections the command center confirmed longer ago than that are removed. By default everything is kept, at about 200 bytes a detection

Who can do what

Detections show where people are, so the command center gives nothing out without a token.

- `NEUSE_CC_TOKEN` is the coordinator token. It can read everything and also send
- `NEUSE_CC_UNIT_TOKENS` gives each unit its own token, like `unit-1=abc,unit-2=def`. A unit token can only send, and only under its own unit's name. A lost or stolen unit then cannot read anyone's position or pretend to be another unit
- With no token set, the command center only answers the computer it runs on
- Tokens and positions sent over plain HTTP can be read by anyone on the way. The unit refuses to send a token over plain HTTP to another computer unless `NEUSE_SYNC_INSECURE=1` is set. Put the command center behind HTTPS, with a reverse proxy such as nginx or a cloud load balancer, before using it over a network

The coordinator page

Open the command center address in a browser. It asks for the coordinator token, then shows a map with every unit and every sighting of a person, coloured by how recent it is, and a list of units with their state and backlog. It refreshes every 5 seconds and catches up on backlogs in the order they arrive, so detections sent late after an outage still appear.

Ground the units' cameras have covered in the last 24 hours is shaded in blue squares about 22 m across, darker where it was searched recently. Each unit works out the ground its camera sees from its height, heading and tilt, and sends the squares with its sightings. A square with no shading has not been searched, which matters as much as where people were seen. Reports say how many square km were searched in their window. The squares need the unit's height, heading and tilt, and the camera's view angles from `NEUSE_CAMERA_FOV`.

A unit shows as online while the command center keeps hearing from it. If it is in contact but its detector has stopped handling frames, for example because the camera failed, it shows as not searching.

The map library is served by the command center itself. If the map background cannot be reached, the page says so and still shows units and sightings in the right places.

A sighting is one detection in one frame. The same person seen in ten frames is ten sightings, so the page and the reports count sightings, never people.

Where a sighting is placed

When the caller gives the unit's height above the ground, its heading and how far the camera tilts down, each sighting is placed on the ground under its box. This assumes flat ground and needs the camera's view angles, set with `NEUSE_CAMERA_FOV`. The default of 45 by 37 degrees is a guess for a common thermal lens, so set it for the real camera.

Without those, a sighting is placed at the unit itself and marked that way on the map and in reports. From 130 m with the camera 30 degrees below level, the person is then about 225 m from the point shown.

The demo units take the height and camera tilt from each HIT-UAV file name. The dataset does not give the camera's view angles, so the demo's ground positions are estimates.

Situation reports

The Write report button on the coordinator page, or `/api/v1/report?minutes=30`, gives a short summary for the chosen window. It covers which units are online, which have gone silent with sightings still waiting to send, which are in contact but no longer searching, which hold refused sightings, and the areas where people were seen. Each area lists its position, sightings, the units involved, how recent it is, the highest confidence and the nearest unit.

- Sightings within 100 m of each other count as one area
- The facts are worked out from the database and always come back with the text, so a coordinator can check them
- With no language model set up, a template writes the report, so it works with no internet
- With `NEUSE_LLM_URL`, `NEUSE_LLM_KEY` and `NEUSE_LLM_MODEL` set, an OpenAI style chat API writes it instead, such as a model on Huawei Cloud. It is told to use only the facts given and never to turn sightings into a head count. If it fails or answers badly, the template is used and the report says so
- Reports are only written when asked for, so a language model is not called on every refresh

Alerts

The command center raises an alert when at least 3 sightings in one area, each with a confidence of 0.8 or more, reach it within 10 minutes. It counts by when sightings arrive, so a unit that was out of range still raises an alert when its backlog comes in. Sightings more than 2 hours old are left out, and the text says when the people were seen. Alerts show in a list and as rings on the coordinator page. With Huawei Cloud Simple Message Notification set up, each alert also goes out as a text message, which reaches a phone on any network. An area that was just alerted stays quiet for 30 minutes. If a text fails, the alert is still kept and the page shows why. See `docs/DEPLOY.md` step 9.

Layout

- `src/pipeline.py` - Thermal detection, choosing between the detectors, and the ground covered
- `src/person_check.py` - The person check
- `src/person_cnn.py` - Runs the MindSpore trained person CNN with NumPy
- `src/geo.py` - Places a sighting on the ground from the unit's height, heading and camera tilt, and works out the ground each frame covers
- `src/logger.py` - Local detection log in SQLite, which doubles as the queue of detections to send
- `src/sync.py` - Store and forward sync to the command center
- `src/config.py` - Detector choice, thresholds, unit position and database location
- `src/metrics.py` - Precision, recall and mAP against labelled boxes
- `src/hit_uav.py` - Reads the HIT-UAV labels
- `dashboard/` - Local web dashboard
- `command_center/` - Command center API that units sync to, the coordinator map page, situation reports and alerts, on SQLite or RDS for MySQL
- `deploy/` - Docker and HTTPS setup for running the command center on a Huawei Cloud server
- `training/` - MindYOLO config and guide for training on ModelArts
- `src/mindyolo_detector.py` - Runs the trained MindYOLO model on the unit with MindSpore Lite
- `scripts/simulate_feed.py` - Sends made up or real thermal frames through the pipeline for demos
- `scripts/demo_units.py` - Flies several demo units at once against a command center
- `scripts/evaluate_hit_uav.py` - Scores a detector on HIT-UAV
- `scripts/evaluate_unseen_flights.py` - Scores the person check on flights held out of training
- `scripts/train_person_check.py` - Trains the person check
- `scripts/train_person_cnn.py` - Trains the person CNN with MindSpore and exports it to MindIR
- `scripts/prepare_hit_uav.py` - Makes the person only copy of HIT-UAV for MindYOLO
- `models/` - Trained model files
- `tests/` - Automated tests
- `docs/ARCHITECTURE.md` - How the parts fit together
- `docs/DEPLOY.md` - Step by step deployment on Huawei Cloud
- `docs/PROGRESS.md` - Dated record of what was built, measured and decided

Run it

Tested on Python 3.12.

```bash
git clone https://github.com/808kirai/NeuSE.git
cd NeuSE
pip install -r requirements.txt

# run the tests
python -m pytest -q tests

# terminal 1: generate detections from made up frames
python scripts/simulate_feed.py --frames 120

# terminal 2: open the dashboard at http://127.0.0.1:5000
python dashboard/app.py
```

To run the pipeline once on a test frame, use `python -m src.pipeline`.

Two speed checks depend on the computer, so they only run when asked for, with `NEUSE_TIMING_TESTS=1 python -m pytest -q tests`.

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

# the result on flights held out of training, trains 5 detectors and takes about 20 minutes
python scripts/evaluate_unseen_flights.py --dataset ../datasets/hit-uav

# train the person check again, which rewrites models/person_check.npz
python scripts/train_person_check.py --dataset ../datasets/hit-uav

# the person cnn result above
python scripts/evaluate_hit_uav.py --dataset ../datasets/hit-uav --split test --detector person-cnn

# train the person cnn again with mindspore, about 20 minutes, which rewrites models/person_cnn.npz
# run it with the python of a separate mindspore environment, set up as the top of the script shows
python scripts/train_person_cnn.py --dataset ../datasets/hit-uav

# send real test frames through the person check, with the dashboard open
python scripts/simulate_feed.py --replay ../datasets/hit-uav/normal_json/test --frames 60

# the person only copy for MindYOLO training, add --with-bicycles to keep bikes and their riders as a second class
python scripts/prepare_hit_uav.py --source ../datasets/hit-uav --out ../datasets/hit-uav-person
```

Training the person check again on the same computer gives exactly the same weights. Another computer can differ in the last decimal places, which can move a score by a point in the third decimal.

Data

HIT-UAV by Suo et al., Scientific Data 10, 227 (2023), shared under CC BY 4.0 at https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset. It is not stored in this repository.

Settings

- `NEUSE_DETECTOR` - `person-check` by default, `person-cnn` for the MindSpore CNN, `warm-spots` for the first baseline, or `mindyolo` for a model trained on ModelArts
- `NEUSE_FRAME_SCALE` - Resizes frames before the person check, so people are about 15 pixels tall. Defaults to `1`
- `NEUSE_CAMERA_FOV` - How wide the camera sees across and down the frame in degrees, like `45,37`. Used to place sightings on the ground
- `NEUSE_DB` - Detection database, defaults to `neuse_local.db` in the project folder. Keep it out of folders that sync to the cloud, such as OneDrive
- `NEUSE_KEEP_DAYS` - Days to keep detections on the unit after the command center confirmed them. Defaults to `0`, which keeps them for ever
- `NEUSE_LAT` and `NEUSE_LON` - Unit position when no GPS position is given, defaults to central Nairobi
- `NEUSE_UNIT_ID` - Name this unit reports to the command center, defaults to `unit-` and the computer's name. Letters, digits, dots, dashes and underscores only
- `NEUSE_SYNC_URL` - Command center address for `python -m src.sync`
- `NEUSE_SYNC_TOKEN` - Token the unit sends to the command center
- `NEUSE_SYNC_INSECURE` - Set to `1` to allow sending the token over plain HTTP to another computer, on a network you trust
- `NEUSE_CC_TOKEN` - Coordinator token for the command center, which can read and send
- `NEUSE_CC_UNIT_TOKENS` - A token per unit, like `unit-1=abc,unit-2=def`, which can only send for that unit
- `NEUSE_CC_DB` - Command center database, defaults to `command_center.db` in the project folder
- `NEUSE_CC_DATABASE_URL` - RDS for MySQL in place of the SQLite file, like `mysql://user:password@host:3306/neuse`. Write `@` in the password as `%40`
- `NEUSE_MINDYOLO_MODEL` and `NEUSE_MINDYOLO_THRESHOLD` - The trained MindYOLO model and the score its boxes need, see `training/README.md`
- `NEUSE_MINDYOLO_DEVICE` - Where MindSpore Lite runs the model, `cpu` by default or `ascend` on an Ascend NPU such as the Orange Pi AIpro's
- `NEUSE_CC_HOST` and `NEUSE_CC_PORT` - Command center address, defaults to `127.0.0.1` and `8000`. It will not open to a network with no token set
- `NEUSE_TILE_URL` and `NEUSE_TILE_CREDIT` - Map background for the coordinator page and the credit line shown for it. Defaults to the public OpenStreetMap tiles
- `NEUSE_LLM_URL`, `NEUSE_LLM_KEY` and `NEUSE_LLM_MODEL` - Optional OpenAI style chat API for situation reports, the base address ending before `/chat/completions`. Unset, reports come from the template
- `NEUSE_SMN_TOPIC_URN`, `NEUSE_SMN_REGION`, `NEUSE_SMN_PROJECT_ID`, `CLOUD_SDK_AK` and `CLOUD_SDK_SK` - Optional text alerts through Huawei Cloud SMN. Unset, alerts only show on the map
- `NEUSE_ALERT_MIN_SIGHTINGS`, `NEUSE_ALERT_MIN_CONFIDENCE`, `NEUSE_ALERT_WINDOW_MINUTES`, `NEUSE_ALERT_QUIET_MINUTES` and `NEUSE_ALERT_MAX_AGE_MINUTES` - When alerts are raised, defaults `3`, `0.8`, `10`, `30` and `120`
- `NEUSE_PUBLIC_URL` - The command center's public address, added to alert texts
- `NEUSE_HOST` - Defaults to `127.0.0.1`, set to `0.0.0.0` to open the unit dashboard to other devices. It has no token, so only do this on a network you trust
- `NEUSE_PORT` - Dashboard port, defaults to `5000`
- `NEUSE_DEBUG` - Off by default, set to `1` for Flask debug mode on a trusted dev machine only
- `NEUSE_SENSOR_RANGE` - Fixed `low,high` scale for 16 bit or float frames, in the camera's own units. Unset, each frame is scaled to its own range, which can make the warmest patch of a cold empty scene look hot. For a camera reporting hundredths of a kelvin, `27315,31315` covers 0 to 40 C
- `NEUSE_TIMING_TESTS` - Set to `1` to also run the two speed checks in the tests

Performance

Every frame is timed, and the time is saved with its detections. The dashboard shows the average over the pipeline's last 100 frames, whether or not they had detections. The scoring script reports the time per image on real footage.

License

All rights reserved. The code may not be used, copied, modified or distributed without written permission. The Huawei ICT Competition 2026–2027 committee and judges may run, reproduce and present it to evaluate this entry. See `LICENSE`.

The coordinator page includes the Leaflet map library, © Volodymyr Agafonkin, under the BSD 2-Clause licence in `command_center/static/leaflet/LICENSE`.
