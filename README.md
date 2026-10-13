NeuSE (Neural Sentinel Edge)
Offline edge-native dual-spectral AI system for search and rescue operations in zero-connectivity environments. Built for the Huawei ICT Competition Innovation Track.

Problem Statement
In disaster zones such as earthquakes, floods, or remote wilderness, communication infrastructure is routinely destroyed or unavailable. Standard cloud-dependent AI tools fail in these situations because they require continuous high-bandwidth internet connectivity to process optical and thermal camera feeds. NeuSE solves this by running quantized computer vision models directly on edge hardware, providing real-time survivor and hazard detection without any network dependency.

System Architecture
The project bridges cloud-based model training with completely air-gapped local edge execution:
- Huawei Cloud ModelArts: Handles dataset management, data labeling, model training, and hyperparameter tuning on high-performance cloud clusters.
- MindSpore Lite: Quantizes and optimizes trained dual-spectral detection models for low-power edge accelerators.
- Intelligent EdgeFabric (IEF): Controls containerized deployments, manages edge node configurations, and syncs incident telemetry when network links are temporarily established.
- Local Edge Gateway: Processes incoming RGB and thermal frames locally and hosts an offline web telemetry interface for incident commanders.

Technical Specifications
- Core Runtime: Python 3.12, Flask
- Vision Pipeline: OpenCV, NumPy (dual-spectral frame fusion and anomaly detection)
- Cloud and Edge Tools: Huawei Cloud ModelArts, Intelligent EdgeFabric (IEF), MindSpore Lite
- Testing Suite: Pytest (automated resilience, adversarial input, and latency verification)

Repository Layout
- [src/]: Core dual-spectral processing pipeline (`pipeline.py`), system configuration (`config.py`) and local event logging (`logger.py`)
- [dashboard/]: Offline Flask web interface (`app.py`) for local command telemetry and real-time alerts
- [models/]: Storage directory for quantized edge weights
- [tests/]: Automated resilience and latency test suite (`test_resilience.py`)
- [requirements.txt]: Project environment dependencies

Getting Started
1. Clone the repository:
git clone https://github.com/808kirai/NeuSE.git
cd NeuSE
2. Install dependencies:
pip install -r requirements.txt
3. Execute verification tests:
pytest tests/test_resilience.py -v
4. Start the offline monitoring dashboard:
python dashboard/app.py

Performance Verification
The system includes an automated test suite verifying system stability across edge-case conditions, including missing frames (None), completely dark frames, fully white/saturated frames, and unexpected frame resolutions (1x1). Execution latency is benchmarked locally to ensure frame processing remains within operational execution budgets.

License
Distributed under the MIT License. See LICENSE for details.