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

Reference - The HIT-UAV paper, Suo et al. 2023 table 4, reports person AP@0.5 on the same test images of 89.9% for YOLOv4, 85.6% for SSD-512, 75.5% for Faster R-CNN and 16.9% for YOLOv4-tiny. The small model meant for limited hardware does poorly on people, who are only about 14 by 15 pixels here.

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

Next

- Situation reports for coordinators, working offline from a template and able to use an LLM on Huawei Cloud
- Command center on Huawei Cloud ECS with RDS
- MindYOLO training on ModelArts once access is ready, then MindSpore Lite on the unit
