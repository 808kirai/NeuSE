Models

Trained model files are kept out of git. Files ending in `.mindir` and `.ms` are ignored.

The plan is to train a MindYOLO person detector on Huawei Cloud ModelArts with the HIT-UAV thermal dataset, convert it to MindSpore Lite format and place it here. Until then, the pipeline uses the classical baseline in `src/pipeline.py`.

`scripts/prepare_hit_uav.py` prepares the training data. It keeps only the person labels and writes the dataset section of a MindYOLO config. Use `--data-root` to set where the folder will live on ModelArts.
