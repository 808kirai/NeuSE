Models

`person_check.npz` holds the weights of the person check in `src/person_check.py`. It is 160 KB, made by `scripts/train_person_check.py` from the HIT-UAV training images, and kept in git so the results can be checked without training again. Besides the network it holds the calibration that turns its output into a confidence, and the confidence a hit needs to be reported, both fitted on the validation images.

Training it again on the same computer gives exactly the same network weights. Other computers can differ in the last decimal places, since they add numbers up in a different order.

Other trained model files are kept out of git. Files ending in `.mindir` and `.ms` are ignored.

The plan is to train a MindYOLO person detector on Huawei Cloud ModelArts with the HIT-UAV thermal dataset, convert it to MindSpore Lite format and place it here as `person_mindyolo.ms`. It has to beat the person check to replace it. The steps are in `training/README.md`.

`scripts/prepare_hit_uav.py` prepares the training data. It keeps only the person labels and writes the dataset section of a MindYOLO config. Use `--data-root` to set where the folder will live on ModelArts.

HIT-UAV labels a bike and its rider as one bicycle box and gives the rider no person box. Of 3,630 bicycle boxes in the training images only 3 have a person box on them. A person only model therefore learns to ignore people on bikes. Add `--with-bicycles` to keep bicycles as a second class, so riders are found as bicycles instead of being treated as background.

The prepared file names load in MindYOLO's dataset reader, which turns each name into a number, and all of them come out different. A full MindYOLO training run has not been tried yet.
