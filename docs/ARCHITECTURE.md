# Architecture

NeuSE has two halves: field units that work with no network, and a command center on Huawei Cloud that brings their results together. Only the field unit exists today. The command center is planned.

## Field unit (built)
```
thermal frames -> detection pipeline -> local SQLite log -> local dashboard
```
- **Detection** (`src/pipeline.py`, `src/person_check.py`) finds small warm spots by local contrast, scores each one with a small neural network run in NumPy, then fits and de-duplicates the boxes. On the HIT-UAV test images it scores mAP@0.5 0.670 at about 24 ms per frame on a laptop CPU. The first warm spot baseline, at mAP@0.5 0.070, can still be chosen with `NEUSE_DETECTOR=warm-spots`.
- **Log** (`src/logger.py`) stores each detection with its time, position, box, confidence, peak intensity, detector name and processing time. Confidence is empty for warm spots, which has none.
- **Dashboard** (`dashboard/app.py`) shows recent detections on the unit itself, with no internet needed.

## Planned
- **Training.** A MindYOLO person detector trained on Huawei Cloud ModelArts with the HIT-UAV thermal drone dataset, then exported to MindIR.
- **On-device inference.** MindSpore Lite runs the trained model on the field unit in place of the baseline. It targets an ordinary CPU first, and Ascend hardware if it becomes available.
- **RGB and thermal together.** The RGB camera confirms thermal hits to cut false alarms from hot objects that are not people.
- **Store and forward.** Detections queue on the unit and upload whenever a connection comes back.
- **Command center.** A web app on Huawei Cloud ECS with an RDS database. It collects detections from all units, maps them, and writes plain-language situation reports for rescue coordinators.

## Full flow once built
```
camera frames -> on-device model -> local log -> sync when online -> cloud command center -> coordinators
```
