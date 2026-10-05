- Neural Sentinel Edge [NeuSE]

NeuSE is an offline dual spectral edge AI system built for search and rescue operations in zero connectivity environments. It processes concurrent optical and long wave infrared thermal streams on local edge hardware to detect survivor signatures through foliage, smoke, and debris.

* System Architecture
The project implements a hybrid edge to cloud processing loop:
1. Edge Inference: MindSpore Lite runs lightweight target detection on Huawei Ascend 310 or Atlas 200 hardware with under 15ms frame latency.
2. Local Data Engine: GaussDB Lite stores detection coordinates, confidence metrics, and thermal vector data locally without network access.
3. Cloud Feedback: When network connectivity is reestablished, unannotated edge detection cases sync to Huawei ModelArts for automated dataset expansion and model retraining.

* Repository Layout:
NeuSE
─ docs
 ─ ARCHITECTURE.md
 ─ PROPOSAL.md
─ src
 ─ config.py
 ─ logger.py
 ─ pipeline.py
─ dashboard
 ─ app.py
 ─ templates
 ─ index.html
─ models
─ tests
─ .gitignore
─ requirements.txt
─ README.md

To Run NeuSE:
* Prerequisites
Python 3.9 or higher with OpenCV runtime dependencies.

* Installation
git clone https://github.com/808kirai/NeuSE.git
cd NeuSE
pip install -r requirements.txt

* Execution
Run the edge processing simulation:
python src/pipeline.py

Launch the local telemetry dashboard:
python dashboard/app.py