Progress log

A dated record of what was built, what was measured and why each choice was made. Newest entries are at the bottom. Every number here can be reproduced with the commands in the README.

Competition

Huawei ICT Competition 2026–2027, Innovation Track, Southern Africa region. Preliminary submission due 24 December 2026.

2026-10-05 to 2026-10-13 - First prototype

- Thermal detection pipeline, SQLite detection log and configuration
- Local Flask dashboard for logged detections
- First tests, and a fix for slow processing on CPU

2026-10-16 - Honest numbers

- Removed placeholder metrics and replaced them with measured ones
- Removed a dashboard database wipe on startup, external fonts and Flask debug mode
- Added the licence, a demo feed script, code comments and 22 tests
- Timing tests check the median of 20 runs, because single runs were flaky

Decision - Keep the search and rescue idea and position NeuSE as offline field units plus a command center on Huawei Cloud. The innovation rests on working without a network, many units feeding one picture and reports for coordinators, not on thermal detection alone.

Decision - Train the future detector with MindYOLO on ModelArts, using the HIT-UAV dataset. MindYOLO is Apache 2.0 and Huawei technology. HIT-UAV is CC BY 4.0. Ultralytics was avoided because of its AGPL licence and because it is not Huawei technology.

2026-11-07 to 2026-11-11 - Hardening, PRs 2 to 16

- Tests run on every push, with the timing check skipped on shared CI machines
- One stuck hot pixel no longer hides a real target in 16 bit or float frames
- Optional fixed sensor range, so a cold empty scene is not stretched into false hits
- Detection times are saved with their UTC offset
- Requirements pinned to the versions the tests pass with
- Demo feed can send 16 bit frames with stuck hot pixels
- Warm areas found with connected components, noisy frames went from about 200 ms to 12 ms

2026-11-13 - Real scores on HIT-UAV, PR 17

- Scored the warm spot baseline against the real person labels in HIT-UAV. Test set of 579 images with 2,611 people, hits counted at IoU 0.5
- Default settings - mAP@0.5 0.020, precision 3.2%, recall 16%
- Tuned on the validation images - mAP@0.5 0.070, precision 4.9%, recall 36%, about 31 false hits per image
- Daytime is the weak spot, mAP@0.5 0.004 by day against 0.129 at night
- Built the person only copy of HIT-UAV in MindYOLO format, ready for ModelArts

Reference - The HIT-UAV paper, Suo et al. 2023 table 4, reports person AP@0.5 on the same test images of 89.9% for YOLOv4, 85.6% for SSD-512, 75.5% for Faster R-CNN and 16.9% for YOLOv4-tiny. People are only about 14 by 15 pixels here. The YOLOv4-tiny figure is a poor guide to small detectors, see the entry of 2026-11-14.

Decision - Settings are tuned on the validation images and the test images are scored once, so the test numbers stay fair.

2026-11-13 - Classical person check, PR 18

- New default detector. Finds small warm spots by local contrast, scores each with a tiny neural network in NumPy, then fits and de-duplicates the boxes
- Trains in about 3 minutes on a laptop CPU. The weights are 160 KB and kept in the repository
- Test set - mAP@0.5 0.670, precision 79%, recall 65%, under 1 false hit per image, about 24 ms per frame on a laptop CPU
- By condition - mAP@0.5 0.709 at night, 0.501 by day, 0.735 at 60 to 80 m and 0.504 at 120 to 130 m
- False hits on vehicles fell from 1,663 to 16
- A top-hat filter was tried first and stopped at about mAP@0.5 0.09 on validation, so it was not kept

Decision - Everything is plain NumPy and OpenCV. OpenCV 5 no longer includes HOG or its classic classifiers, and no extra libraries are needed on the unit.

Decision - The person check is the bar for MindYOLO, which has to beat 67% person AP to replace it. It also stays as a fallback that runs on any CPU.

2026-11-13 - Store and forward sync, PR 19

- Every detection gets a unique id and the unit's name, and waits in the unit's database until the command center confirms it
- Sync sends batches of 100, compressed, newest first, and retries with waits that double up to 5 minutes
- The command center keeps each id once, so resending after a lost reply is safe
- Refused detections stay on the unit with the reason and are not resent, so one bad row cannot block the queue
- Live check - 21 real detections logged with the command center off all stayed queued, then all 21 arrived once it was on, and a second sync added none

Decision - Newest first, so coordinators see the current picture as soon as a link returns, then the backlog fills in.

2026-11-13 - Coordinator map and several units, PR 20

- Units send a status note every few seconds with position, detector and backlog. A unit counts as offline after 60 seconds without contact
- Detections are saved at the unit's position for that frame
- Coordinator page with a live map of units and people found, coloured by how recent they are
- Demo of several drones as separate processes, flying search patterns with real HIT-UAV frames and random link drops
- Live check - 3 units, 161 detections, unit-3 queued 16 during an outage and sent them all when the link returned

Decision - The map follows detections in the order they arrive, not by detection time, because backlogs come in newest first and would otherwise be missed.

Known limits - Map tiles come from public OpenStreetMap and need internet. Detections are placed at the unit's position, not projected onto the ground.

2026-11-13 - Situation reports, PR 21

- Short plain language reports for a chosen window. They cover which units are online, which have gone silent with sightings still waiting, and the areas where people were seen, each with its position, sightings, units, recency, highest confidence and nearest unit
- Sightings within 100 m count as one area
- A template writes the report by default, so it works with no internet. An OpenAI style chat API can write it instead when set up, such as a model on Huawei Cloud, with the template used if it fails
- The facts always come back with the text so coordinators can check them
- Live check - 3 units with link drops, 360 sightings grouped into 7 areas, each with its nearest unit and distance

Decision - The page and reports now say sightings, not people found. The same person seen in many frames is many sightings, so counting people would overstate the result.

Decision - Reports are written only when a coordinator asks, so a paid language model is not called on every refresh.

2026-11-14 - Code review and fixes, PR 22

The whole codebase was read for bugs and weak logic, each suspected problem was tested, and every one found was fixed. Nothing was wrong with the headline scores. The same model scores the same on the test images after the fixes, mAP@0.5 0.670 with 79% precision and 65% recall.

Bugs that could lose or hide data

- A detection the command center refused was marked as sent on the unit and never tried again, with no warning. A unit name with a space in it emptied a unit's queue while nothing reached coordinators. Refused detections are now held with the reason, counted on the dashboard and at the command center, and put back with `python -m src.sync --retry-refused`. A unit will not start with a name the command center would refuse
- The retry wait overflowed after 1,024 failures in a row, a few days offline, and stopped sync. It is capped now
- Unit clocks were trusted. A detection stamped two hours fast or in 2020 was accepted and then left out of every report. Each batch now carries the unit's clock reading and the command center corrects the times by the difference
- A unit showed as online whenever its sync program was alive, even if its detector had stopped hours ago. Units now report how long their detector has been quiet and the map shows them as not searching
- A database write error stopped detection. Hits are now held in memory and saved on a later frame

Outputs that misled

- Sightings were placed at the drone, up to about 225 m from the person at 130 m with a slanted camera. They are now placed on the ground when the unit gives its height, heading and camera tilt, and marked as placed at the unit otherwise
- Confidence overstated. Hits shown as 0.90 to 0.95 were people 56% of the time, and 27% of reported hits showed 1.00. Confidence is now calibrated on the validation images. On the test images, hits shown as 0.9 or more were people 92% of the time, and the gap is within about 15 points across the range
- The dashboard's average processing time came from detection rows, so busy frames counted many times and empty frames never. It now averages the last 100 frames

Security

- With no token the command center was open to anyone. It now only answers its own computer until a token is set
- One shared token let any unit read everything and post as any other. Units can now have their own tokens, which can only send for that unit
- The unit no longer sends a token over plain HTTP to another computer unless told to
- Six kinds of bad input made the server answer 500. They now get a plain refusal

Evaluation

- Every HIT-UAV test image is within 30 frame numbers of a training image from the same flight, and 84% are within 5, so the test score is for scenes the detector has seen
- New score on unseen flights - the 105 flight settings were dealt into 5 parts and each part scored by a detector trained without it. Mean mAP@0.5 0.520, from 0.487 to 0.569, with 63% precision, 54% recall and 1.4 false hits per image
- Of 3,630 bicycle boxes in the training images, 3 have a person box on them. The dataset does not label riders as people, so the person check and the prepared MindYOLO data treat them as background. `prepare_hit_uav.py --with-bicycles` keeps bicycles as a second class
- The comparison with YOLOv4-tiny was overstated. Its 16.9% is one weak run, far below its own 83.6% on cars, and is not what small detectors are limited to. The README now says so
- The worst test frames of the warm spot baseline were looked at before the person check was designed. The numbers were scored once after tuning, but the test images were not fully unseen. The README says so

Demo and deployment

- The coordinator page loaded its map library from another site and stopped working without it. The library is now served by the command center, and the page keeps its lists when the map background cannot load
- The command center runs on the waitress web server, not Flask's development one
- The licence now lets the competition committee and judges run and reproduce the code
- The two speed checks in the tests only run when asked for, so a slow computer does not fail the suite
- Smaller fixes - an explicit arrival number for map paging, a setting to prune old confirmed detections, a warning when the database sits in a OneDrive folder, one name per detector everywhere, and `thermal_pipeline` as the honest name of the pipeline class, with the old name kept working

Decision - Refused detections are held, not dropped and not resent on their own. Resending them blindly would block the queue, and dropping them would hide a fault.

Decision - Clock correction uses the gap between the unit's clock and the command center's at the moment of sending. It cannot fix sightings logged before a unit's clock was put right mid flight.

Decision - The default camera view angles of 45 by 37 degrees are a guess, since HIT-UAV does not publish them. Ground positions in the demo are estimates until a real camera's values are set.

A second check of the fixes before merging found four more, each tested and fixed

- Detections the first version had refused were still hidden after upgrading, since it had marked them as sent. They now show as refused and can be retried
- Moving the command center's first version table was not done in one step. A stop part way left the old rows where the map could not see them. The move is now all or nothing, and rows stranded by an earlier stop are picked up
- With unit tokens set but no coordinator token, nobody could open the map or reports and nothing said why. The command center now warns at start and says why when asked
- The scoring scripts made an empty database in the project folder. They now use a throwaway one

Known limits - No HTTPS is built in, the command center needs a reverse proxy in front of it. The person check is tied to 640 by 512 frames from 60 to 130 m. Time per image on test footage is about 30 ms, and over 100 ms on a frame crowded with warm spots.

2026-11-14 - Ready for Huawei Cloud, deployment and training kits

Everything that does not need cloud access is built, so access turns straight into a live demo and a training run.

- The command center works on RDS for MySQL as well as SQLite, chosen with one setting. The same storage checks run against both
- A Docker image for the command center, with Caddy in front for HTTPS, and `docs/DEPLOY.md` with every step on ECS and RDS
- Every push now also runs the storage checks against a real MySQL 8.0 and builds and tries the Docker image. All three passed on the first push
- A MindYOLO config for HIT-UAV in `training/`, starting from YOLOv8n trained on COCO, with a step by step guide for ModelArts, the export and the conversion for MindSpore Lite
- A `mindyolo` detector slot on the unit that runs the converted model with MindSpore Lite and decodes its output as MindYOLO's YOLOv8 head gives it. It accepts boxes listed either way round. If the model or MindSpore Lite is missing, the unit falls back to the person check
- The scoring scripts score the MindYOLO model exactly as they score the person check
- A bug found while writing the deployment file was fixed. A setting left empty, such as `NEUSE_TILE_URL=`, used to leave the map with no background

Decision - The MindYOLO slot is tested only with a stand in for the model, built from MindYOLO's own YOLOv8 code, since MindSpore Lite is not available on the team's laptop. The first real export must be checked against it.

Decision - MindYOLO moved from GitHub to Gitee, at https://gitee.com/mindspore-lab/mindyolo. The guide points there.

Next

- Command center on Huawei Cloud ECS with RDS, behind HTTPS
- MindYOLO training on ModelArts once access is ready, then MindSpore Lite on the unit

2026-11-14 - A stronger second baseline, searched ground, alerts and a plan for MindYOLO

Built while the team waits for Huawei Cloud access and the Orange Pi AIpro.

- A person CNN trained with MindSpore on the laptop CPU, in about 20 minutes, and exported to MindIR. It finds warm spots at two sizes, which reach 96.7% of labelled people against 93.4% for one size, and looks at a 32 by 32 crop of the frame and its warm spot map around each one. On the unit it runs in NumPy and agrees with MindSpore to 3.8 in a million
- The searched ground. Each unit works out the ground each frame covers and marks squares of about 22 m as searched. They sync to the command center, are shaded on the map for the last 24 hours and counted in reports in square km
- Alerts when an area gets 3 sightings of confidence 0.8 or more within 10 minutes, kept and shown on the map, and sent as text messages through Huawei Cloud SMN when it is set up. They count sightings by when they arrive, so a unit that was out of range still raises an alert when its backlog comes in. Checked against the real SMN SDK, which takes the request as built
- Three MindYOLO runs planned from published results, yolov8n at 640, yolov8n at 960 for small people and yolov8s, with the reasons in `training/README.md`. The MindYOLO slot now reads the input size from the model and can run on an Ascend NPU
- Detections are indexed by arrival time, which the alerts look them up by

Results on the HIT-UAV test images, scored once after picking on the validation images

- Person check - mAP@0.5 0.670, 79% of hits are people, 65% of people found, about 30 ms per image
- Person CNN - mAP@0.5 0.659, 71% of hits are people, 67% of people found, about 100 ms per image
- On the validation images, 0.643 for the person check and 0.625 for the CNN

Decision - The person check stays the default. The CNN finds slightly more people but makes more false hits and is three times slower. It stays as the working MindSpore path, from training to export to the same answers on the unit, and as a second opinion to compare MindYOLO against.

Decision - Published YOLOv8 runs reach about 0.92 person AP@0.5 on HIT-UAV. So the MindYOLO model, not more work on hand built detectors, is the way to a big step up, and it is ready to train once ModelArts access arrives.

Decision - Alerts count sightings by when they reach the command center, not by when they were seen. Counting by when they were seen would never alert on a late backlog, which is the case NeuSE is built for.

Decision - Running the MindYOLO model on the Orange Pi AIpro's NPU through MindSpore Lite has to be checked on the board. MindSpore Lite's Ascend documents list the 310P chip but not the 310B. If it does not work, the board's own CANN tools are the way.

Next

- Command center on Huawei Cloud ECS with RDS, behind HTTPS, with SMN texts
- MindYOLO training on ModelArts, then MindSpore Lite on the unit and on the Orange Pi AIpro
