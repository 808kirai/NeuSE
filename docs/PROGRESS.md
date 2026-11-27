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

Reference - The HIT-UAV paper, Suo et al. 2023 table 4, reports person AP@0.5 on the same test images of 89.9% for YOLOv4, 85.6% for SSD-512, 75.5% for Faster R-CNN and 16.9% for YOLOv4-tiny. People are small here, the median person box is 12 by 19 pixels. The YOLOv4-tiny figure is a poor guide to small detectors, see the entry of 2026-11-14.

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
- Reports say how many alerts were raised in their window and how many went out by text. The map lists the five newest alerts and marks every alert with a dashed ring that changes colour with age
- In a 30 minute demo window three units searched about 2.08 square km and the command center raised 13 alerts

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

2026-11-14 - AI safety and security

Researched against the OWASP Top 10 for LLM applications 2025, the OWASP machine learning top 10, the NIST AI Risk Management Framework and Kenya's Data Protection Act 2019, then built and tested. Everything is in `docs/SAFETY.md`.

- A blind camera never marks ground as searched. Before this, a lens cap or failed sensor still shaded ground as searched, the worst kind of failure for a search tool. Frames that spread less than 2 grey levels are blind, against 11 for the flattest real HIT-UAV frame, and the unit shows as not searching while they last
- Model files are checked against their SHA-256 in `models/checksums.txt` before loading, with a fallback to a detector the unit can trust. This also fixed damaged person check weights crashing the unit
- Units can only send plain detector names, closing a way to put instructions into the language model's input
- A language model's report is checked against the facts, every number must come from them and a report counting people is refused, and the facts only go to a model over HTTPS
- The command center refuses weak tokens on a network, turns away a caller after 20 wrong tokens in 5 minutes, sends protective headers and a content security policy, logs every API request without its token, and removes old records with `NEUSE_CC_KEEP_DAYS`
- The unit dashboard refuses to run Flask's debugger on a network address, since the debugger runs any code it is sent
- CI now audits every installed package with pip-audit. Its first run found known vulnerabilities in pymongo 4.18.2 and pyasn1 0.6.3, both pinned by Huawei Cloud's SMN SDK even in its newest release. The SDK is now installed without its pins, with the fixed pymongo 4.18.3 and pyasn1 0.6.5, and publishing through it to a test server was checked
- CodeQL code scanning, Dependabot, a read only CI token, HSTS from Caddy, and `SECURITY.md` for private reports

CodeQL's first scan found 7 issues. Three were real log injection, the access log wrote the request path as given, so a path with an encoded line break could forge a log line. Everything taken from a request is now escaped and cut to 200 characters. Four were in tests that checked HTML with a pattern and addresses by matching text, which now use an HTML parser and read the policy by its parts.

Decision - When an address is turned away for guessing tokens, even a right token is refused until the window passes, so a lucky guess tells the guesser nothing. Units see this as a dropped link and keep their sightings.

Decision - Records are kept by default on both the unit and the command center, since how long a search needs them is for the team to decide. The settings to remove them are documented for the end of a search.

Dependabot's first proposals were merged once their tests passed - numpy 2.5.4, with both detectors scoring exactly as before on the test images, pytest 9.2.0, pymongo 4.19.0 after publishing through the SMN SDK was checked again, and newer GitHub Actions. The move of the Docker image from Python 3.12 to 3.15 is left open, since the project is tested on 3.12.

Decision - Physical attacks that hide people from thermal cameras are not defended against. NeuSE is for finding people who want to be found, and the remaining risk is written down instead of hidden.

2026-11-22 - What the organisers require, from their Q&A

The organising committee's Task Analysis and Technical Enablement session and its Q&A settled what each stage needs.

- At the preliminary stage no deployment, demo link or runnable application is required. A prototype, a technical proposal, notes on how Huawei Cloud is used, images and a video are enough
- Huawei Cloud vouchers are given only to teams that reach the regional stage, so deployment on ECS and RDS, SMS alerts and MindYOLO training on ModelArts belong to that stage
- MindSpore and ModelArts are not required, and calling existing AI services is allowed. At least one Huawei Cloud service must be used
- Huawei hardware is optional and earns no extra points, so the Orange Pi AIpro is a bonus
- The entry should show what is built and what is planned with a clear line between them, which Template 1 and the paper already do
- At the regional stage the committee reruns the work from the code, data and README, and a mismatch forfeits advancement
- Deadlines are set per region, and a late submission counts as withdrawal

Template 1, the technical paper and the handoff notes were updated to match.

The organisers also said judges should be able to deploy entries themselves. `docs/DEPLOY.md` now starts with a ten minute deployment on any computer with Docker, flying demo units against it and checking the scores, with the Huawei Cloud steps as part 2. The Docker image it uses is built and run on every push.

The README, Template 1 and the technical paper now open the problem with four recent disasters where people had to be found as networks failed, each from published reports. The Kahramanmaraş earthquakes in Türkiye and Syria, February 2023, the Mai Mahiu flash flood in Kenya, April 2024, the Valencia floods in Spain, October 2024, and the Himalayan flash flood on the Nepal and Tibet border, August 2026.

2026-11-26 - Polish on the redesigned pages

Joshua's redesign of the command center and the unit dashboard was kept as it is, a dark console with the overview along the top, the map and a side panel, and refined.

- The map disappeared on phones, squeezed out beside the full width side panel. It now sits above the panel
- Structure uses quiet lines, and the strong border is kept for controls and focus, so the cards no longer all shout at once
- The waiting, refused and alerts numbers turn amber or red only when they are above zero, so colour means someone should look
- The day theme he built for the dashboard now also switches the command center, and both remember the choice
- One vocabulary throughout. Sightings everywhere instead of a mix with detections, sentence case labels, and status lines that say what to do, such as Not searching for 2 min. Check the camera
- Smaller fixes. A lighter blue for searched ground in daylight, singular wording for one earlier alert, a shorter time column, and the retry flag kept on one line

Checked in Chrome at 1440 by 900 and 390 by 844 in both themes, with no content security policy errors and no sideways scrolling on a phone. There were no emojis to remove in the pages, the code or the documents.

Later the same day, after the team looked at the pages

- OpenStreetMap began refusing the map tiles with Access blocked. The security header that told browsers to send no referrer was the cause, since OpenStreetMap's tile policy requires one. Pages now send other sites only the server's own address, never a page's path, and the tiles load again
- One light theme on both pages, and the theme switch is gone. Detection is not affected, the theme is only the pages' colours
- The status button blinks green while live, and black and white while connecting. Write report blinks black and white while it works, and a unit, alert or sighting card blinks a few times when clicked, so it is clear which one the map jumped to. Blinks are slow, and anyone set to reduce motion sees the same colours held still
- Bolder text on both pages. Windows 11's variable Segoe UI comes first, since plain Segoe UI has no medium weight and jumps from regular to semibold
- The map scrolls like a touchpad. Two fingers move it in any direction and a pinch zooms around the pointer, where before two fingers zoomed. A mouse wheel moves it up and down, and ctrl with the wheel zooms. Arrow buttons under the zoom buttons move it too, with a middle button that shows every unit and sighting. The icons are built as elements, since the page never sets HTML from strings

2026-11-26 - Two fixes found while recording the demo video

- A unit whose link dropped was shown as not searching after 30 seconds, while it still counted as online for 60. The command center added the time since the unit's last note to how long its detector had been quiet, so silence looked like a stopped detector, and the situation report repeated it. Now only the unit's own note can say its detector is quiet. A unit that has only gone silent keeps its last heard time and goes offline after 60 seconds. A unit whose last note already said its detector was quiet stays not searching
- Before the coordinator token was entered, the map page said Connection lost behind the token dialog, though nothing had failed. It now says Token needed and asks for the token, and goes live once the right one is entered

Three new tests cover the first fix and fail on the old code. Both fixes were checked in Chrome. 372 tests pass.

2026-11-27 - Research behind the problem

Searched for published research that supports each part of the problem statement and cited it in Template 1 and a new four page technical paper. Every paper was found and its journal details checked, mostly from abstracts and publisher or repository pages, so claims only repeat what those pages say.

- Time decides survival. Adams, Schmidt, Newgard et al. 2007 found finding survivors became unlikely after about 51 hours in Oregon search missions, and Macintyre, Barbera and Smith 2006 found no single cutoff for earthquake entrapment, with the longest survival averaging 6.8 days over 18 events. Both warn against reading their numbers as a fixed deadline, so the text does not use one
- Networks fail. Yamamura, Kaneda and Mizobata 2014 report that medical teams rated mobile phones poor to moderate for the first four days after the 2011 Great East Japan earthquake. Matracia et al. 2022 survey post-disaster communications and Sobin et al. 2016 survey delay tolerant networking, the idea behind the store and forward sync
- Drones need onboard detection. Burke et al. 2019, Lygouras et al. 2019, Kyrkou and Theocharides 2020, Mishra et al. 2020, Karaca et al. 2018 and Rudol and Doherty 2008, including that people become blobs in low resolution thermal at range, which matches HIT-UAV, where the median person box is 12 by 19 pixels

Template 1 fits its 500 word background limit at 494 words, with the reference list at the end of the form. The short paper is submitted with Template 1, kept outside this repository with the other competition files, and gives the method, results, sync, command center, Huawei Cloud status, safety measures and limits in one place. Template 1 also now says 372 automated tests instead of 346.

2026-11-27 - Old sightings no longer wait behind new ones

The unit's outbox was sent strictly newest first, so coordinators see the current picture as soon as a link returns. On a busy unit with a weak link that could leave old sightings waiting for ever, never lost but never sent while new ones kept arriving. Now ten places in every batch of 100 go to the oldest waiting sightings, and the rest stay newest first. When everything fits in one batch nothing changes.

A new test runs a unit that logs 100 sightings a round on a link that drops after one batch. Before the change the 300 sightings it started with never left the unit, now the 50 oldest are through after five rounds and nothing is sent twice. The test fails on the old code and passes on the new. 374 tests pass.

The docs, Template 1 and the technical paper now call the store an outbox, sent newest first with room for the oldest, since a strict newest first order behaves more like a stack than a queue.
