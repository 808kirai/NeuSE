Architecture

NeuSE has two halves. Field units work with no network, and a command center brings their results together. The field unit and a first command center API are built. Moving the command center to Huawei Cloud, with a map and reports for coordinators, is planned.

Field unit, built

```
thermal frames -> detection pipeline -> local SQLite log -> local dashboard
                                              |
                                              v
                                   store and forward sync
```

- Detection, `src/pipeline.py` and `src/person_check.py` - Finds small warm spots by local contrast, scores each one with a small neural network run in NumPy, then fits and de-duplicates the boxes. On the HIT-UAV test images it scores mAP@0.5 0.670 at about 24 ms per frame on a laptop CPU. The first warm spot baseline, at mAP@0.5 0.070, can still be chosen with `NEUSE_DETECTOR=warm-spots`.
- Log, `src/logger.py` - Stores each detection with a unique id, the unit's name, its time, position, box, confidence, peak intensity, detector name and processing time. The log is also the queue of detections waiting to be sent.
- Sync, `src/sync.py` - Sends waiting detections to the command center in compressed batches, newest first, and marks each one sent only once the command center confirms it. When the link is down it keeps waiting longer between tries, up to 5 minutes.
- Dashboard, `dashboard/app.py` - Shows recent detections and how many still wait to be sent, on the unit itself with no internet needed.

Command center, first version built

```
many field units -> /api/v1/detections -> command center database -> /api/v1/units
```

- API, `command_center/app.py` - Receives batches from units, checks every detection and reports which it took and which it refused. With `NEUSE_CC_TOKEN` set, every request needs the token.
- Storage, `command_center/store.py` - Keeps each detection id once, so units can safely resend. SQLite for now, kept in one module so it can move to Huawei Cloud RDS.

Planned

- Training - A MindYOLO person detector trained on Huawei Cloud ModelArts with the HIT-UAV thermal drone dataset, then exported to MindIR.
- On-device inference - MindSpore Lite runs the trained model on the field unit in place of the person check. It targets an ordinary CPU first, and Ascend hardware if it becomes available.
- RGB and thermal together - The RGB camera confirms thermal hits to cut false alarms from hot objects that are not people.
- Command center on Huawei Cloud - The API on ECS with an RDS database, plus a map of every unit and its detections and plain language situation reports for rescue coordinators.

Full flow once built

```
camera frames -> on-device model -> local log -> sync when online -> command center on Huawei Cloud -> coordinators
```
