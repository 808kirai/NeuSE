Architecture

NeuSE has two halves. Field units work with no network, and a command center brings their results together. The field unit and the command center with its coordinator map are built and run locally. Situation reports are built too. Moving the command center to Huawei Cloud is planned.

Field unit, built

```
thermal frames -> detection pipeline -> local SQLite log -> local dashboard
                                              |
                                              v
                                   store and forward sync
```

- Detection, `src/pipeline.py` and `src/person_check.py` - Finds small warm spots by local contrast, scores each one with a small neural network run in NumPy, then fits and de-duplicates the boxes. On the HIT-UAV test images it scores mAP@0.5 0.670 at about 30 ms per frame on a laptop CPU. Its confidence is calibrated on the validation images. The first warm spot baseline, at mAP@0.5 0.070, can still be chosen with `NEUSE_DETECTOR=warm-spots`. A database error never stops detection, hits are held in memory and saved on a later frame.
- Person CNN, `src/person_cnn.py` - A second learned detector, a small CNN trained with MindSpore by `scripts/train_person_cnn.py` and exported to MindIR. It looks at a 32 by 32 crop around warm spots found at two sizes and scores each one. On the unit it runs in NumPy and gives the same answers as MindSpore to about 4 in a million. It scores mAP@0.5 0.659 on the test images at about 100 ms per frame, behind the person check, which stays the default.
- Ground position, `src/geo.py` - When the caller gives the unit's height, heading and camera tilt, each sighting is placed on the ground under its box, assuming flat ground. Without them it is placed at the unit and marked that way.
- Searched ground, `src/geo.py` and `src/pipeline.py` - From the same height, heading and tilt, the unit works out the four corners of the ground each frame sees and marks the squares of about 22 m inside it as searched. Squares are kept in memory and saved with each status note, so the disk is not written every frame.
- Log, `src/logger.py` - Stores each detection with a unique id, the unit's name, its time, its position and how that was worked out, box, confidence, peak intensity, detector name and processing time. The log is also the queue of detections waiting to be sent. A single status row keeps the unit's latest position, frame count and average time per frame.
- Sync, `src/sync.py` - Sends waiting detections to the command center in compressed batches, newest first, and marks each one sent only once the command center confirms it. A detection the command center refuses is held with the reason, counted and shown, and can be retried. When the link is down it keeps waiting longer between tries, up to 5 minutes. Each batch carries the unit's own clock reading. Each round it also sends a short status note, so the command center knows the unit is alive, where it is, how much it still has to send and how long ago its detector last handled a frame. Searched squares are sent the same way, and sent again when the same ground is seen later.
- Dashboard, `dashboard/app.py` - Shows recent detections, how many wait to be sent and how many were refused, on the unit itself with no internet needed.

Command center, built and run locally

```
many field units -> /api/v1/detections, /api/v1/status, /api/v1/coverage -> command center database -> coordinator map
                                                                                         -> alerts -> SMN text messages
```

- API, `command_center/app.py` - Receives batches and status notes from units, checks every detection and reports which it took and which it refused. A coordinator token can read and send. Each unit can have its own token, which can only send, and only under that unit's name. With no token set it only answers the computer it runs on. It runs on the waitress web server.
- Clock correction, `command_center/store.py` - Field computers often have no battery clock. The command center compares the clock reading sent with each batch against its own clock and corrects the detection times by the difference, keeping the time the unit gave next to the correction.
- Coordinator map, `command_center/templates/index.html` - Shows every unit, whether it is online and whether it is still searching, its backlog and its sightings on a live map. It follows detections in the order they arrive, so a backlog sent after an outage still shows up. The map library is served by the command center itself, and the lists keep working when the map background cannot load.
- Searched ground on the map - Squares from all units are merged, keeping the latest time each was seen, and shaded on the map for the last 24 hours. Reports say how many square km were searched.
- Alerts, `command_center/alerts.py` - After each batch arrives, a check in the background looks for areas with at least 3 strong sightings that arrived in the last 10 minutes, so a late backlog counts too. Each alert is kept and shown on the map, and goes out as a text through Huawei Cloud Simple Message Notification when that is set up. An area just alerted stays quiet for 30 minutes.
- Situation reports, `command_center/reports.py` - Work out the facts for a time window, such as silent units with sightings still waiting, units in contact that have stopped searching and areas where people were seen, then write them up in plain words. A template writes them with no network needed, or an OpenAI style language model does when one is set up, falling back to the template if it fails.
- Storage, `command_center/store.py` - Keeps each detection id once, so units can safely resend, and numbers detections in the order they arrive. Works on an SQLite file or on RDS for MySQL, chosen with `NEUSE_CC_DATABASE_URL`, and is checked against MySQL 8.0 on every push.
- Deployment, `deploy/` and `docs/DEPLOY.md` - A Docker image for the command center and Caddy in front of it for HTTPS, for one Huawei Cloud ECS server with RDS for MySQL. The image is built and tried on every push.

Planned

- Training - A MindYOLO person detector trained on Huawei Cloud ModelArts with the HIT-UAV thermal drone dataset, then exported to MindIR. Three runs are planned, yolov8n at 640 and at 960 for small people, and yolov8s, picked on the validation images. The configs and guide are in `training/`.
- On-device inference - MindSpore Lite runs the trained model on the field unit in place of the person check, chosen with `NEUSE_DETECTOR=mindyolo`. The slot, `src/mindyolo_detector.py`, is built and tested with a stand in for the model. It reads the input size from the model and runs on an ordinary CPU or, with `NEUSE_MINDYOLO_DEVICE=ascend`, on an Ascend NPU such as the Orange Pi AIpro's.
- Edge board - The unit on an Orange Pi AIpro with its Ascend 310B NPU, the board the team is getting.
- RGB and thermal together - The RGB camera confirms thermal hits to cut false alarms from hot objects that are not people.
- Command center on Huawei Cloud - The API, map, reports and alerts on ECS with RDS for MySQL, behind HTTPS, with reports written by a model in ModelArts Studio and alert texts sent by SMN. Everything for it is ready and waiting for cloud access.
- Live pose - Height, heading and camera tilt from the drone's own flight data, in place of the values a caller passes in today.

Full flow once built

```
camera frames -> on-device model -> local log -> sync when online -> command center on Huawei Cloud -> coordinators
```
