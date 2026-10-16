Models

Trained model files are kept out of git. Files ending in `.mindir` and `.ms` are ignored.

The plan is to train a MindYOLO person detector on Huawei Cloud ModelArts with the HIT-UAV thermal dataset, convert it to MindSpore Lite format and place it here. Until then, the pipeline uses the classical baseline in `src/pipeline.py`.
