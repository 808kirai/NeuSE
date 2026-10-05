Welcome to NeuSE technical architecture

Below is the Overview:
NeuSE decouples execution from central infrastructure. The primary system runs fully autonomous on edge compute nodes attached to uncrewed aerial vehicles or portable ground equipment.

Below are our component specifications:
1. Vision engine
Processing framework: MindSpore Lite
Execution target: Huawei Ascend 310 or Atlas 200 DK
Processing mode: Dual stream spatial alignment combining RGB optical and LWIR thermal feeds
Primary outputs: Bounding boxes, peak thermal values, confidence scores

2. Local persistence layer
Database engine: GaussDB Lite / SQLite spatial engine
Stored parameters: Timestamp, Latitude, Longitude, Confidence, Thermal Peak, Bounding Box Data

3. Cloud integration layer
Platform: Huawei ModelArts
Functionality: Centralized dataset training, hyperparameter optimization, model export in MINDIR format

The simplified data pipeline flow is shown below
Input Video Streams -> Frame Alignment -> Feature Detection -> GaussDB Local Log -> Cloud Sync on Connectivity