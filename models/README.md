Models

`person_check.npz` holds the weights of the person check in `src/person_check.py`. It is 160 KB, made by `scripts/train_person_check.py` from the HIT-UAV training images, and kept in git so the results can be checked without training again.

Other trained model files are kept out of git. Files ending in `.mindir` and `.ms` are ignored.

The plan is to train a MindYOLO person detector on Huawei Cloud ModelArts with the HIT-UAV thermal dataset, convert it to MindSpore Lite format and place it here. It has to beat the person check to replace it.

`scripts/prepare_hit_uav.py` prepares the training data. It keeps only the person labels and writes the dataset section of a MindYOLO config. Use `--data-root` to set where the folder will live on ModelArts.
