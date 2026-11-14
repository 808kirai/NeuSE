Training the MindYOLO person detector on ModelArts

This trains a small YOLOv8 person detector with MindYOLO and MindSpore on Huawei Cloud ModelArts, converts it for MindSpore Lite and scores it with the same scripts as the person check, so the two can be compared fairly. The steps have not been run yet, since the team does not have Huawei Cloud access. The detector slot on the unit is tested with a stand in for the model.

To replace the person check, the trained model has to beat mAP@0.5 0.670 on the HIT-UAV test images. The person check's score on unseen flights, 0.520, is the second bar once the MindYOLO model is also trained per flight group.

Files

- `training/hit_uav_yolov8n.yaml` - MindYOLO config that starts from yolov8n and changes only what HIT-UAV needs
- `scripts/prepare_hit_uav.py` - Makes the person only dataset in MindYOLO's format
- `src/mindyolo_detector.py` - Runs the converted model on the unit and turns its output into sightings

1. Prepare the data on your own computer

```bash
python scripts/prepare_hit_uav.py --source ../datasets/hit-uav --out ../datasets/hit-uav-person --data-root /home/ma-user/work/hit-uav-person
```

Add `--with-bicycles` to keep bicycles and their riders as a second class. HIT-UAV labels a rider as a bicycle, not as a person, so a person only model learns to ignore people on bikes.

Zip the `hit-uav-person` folder, about 200 MB, and upload it to an OBS bucket in the same region you will train in.

2. Open a ModelArts notebook

Create a notebook with a MindSpore image on Ascend, or on GPU if that is what your region and credit offer. Copy the zip from OBS into `/home/ma-user/work` and unzip it there, so the paths in the config match.

3. Set up MindYOLO

MindYOLO now lives at https://gitee.com/mindspore-lab/mindyolo. Pick the release that matches the notebook's MindSpore version, as listed in its README.

```bash
cd /home/ma-user/work
git clone https://github.com/808kirai/NeuSE.git
git clone https://gitee.com/mindspore-lab/mindyolo.git
cd mindyolo
pip install -r requirements.txt
cp /home/ma-user/work/NeuSE/training/hit_uav_yolov8n.yaml configs/yolov8/
```

Starting from MindYOLO's YOLOv8n weights trained on COCO is much faster than starting from nothing. The link is in `configs/yolov8/README.md`, for example `yolov8-n_500e_mAP372-cc07f5bd.ckpt`.

4. Train

```bash
python train.py --config configs/yolov8/hit_uav_yolov8n.yaml --device_target Ascend --weight yolov8-n_500e_mAP372-cc07f5bd.ckpt --strict_load False --run_eval True --save_dir ./runs/hit_uav
```

`--strict_load False` lets the COCO weights load even though the last layer now has one class instead of 80. Use `--device_target GPU` on a GPU notebook. The config trains for 200 rounds through the images. Lower `epochs` and `stage_epochs` together to try a shorter run first.

5. Export and convert

```bash
python deploy/export.py --config configs/yolov8/hit_uav_yolov8n.yaml --weight ./runs/hit_uav/weights/BEST.ckpt --file_format MINDIR --device_target Ascend
converter_lite --fmk=MINDIR --modelFile=hit_uav_yolov8n.mindir --outputFile=person_mindyolo
```

Replace `BEST.ckpt` with the checkpoint that scored best in the training log. The second command needs MindSpore Lite's converter tool, from the MindSpore Lite download for the unit's system. It writes `person_mindyolo.ms`. Put that file in the repository's `models/` folder on the unit. It is kept out of git.

6. Run it on the unit

Install MindSpore Lite's Python package for the unit's system, then pick the detector.

```bash
NEUSE_DETECTOR=mindyolo python scripts/simulate_feed.py --replay ../datasets/hit-uav/normal_json/test --frames 30
```

If the model file or MindSpore Lite is missing, the unit warns and keeps detecting with the person check. If the model's output is not the shape the slot expects, it says so. The slot accepts boxes listed either way round.

7. Score it the same way as the person check

Pick the threshold on the validation images first, then score the test images once.

```bash
NEUSE_DETECTOR=mindyolo python scripts/evaluate_hit_uav.py --dataset ../datasets/hit-uav --split val
NEUSE_MINDYOLO_THRESHOLD=0.3 python scripts/evaluate_hit_uav.py --dataset ../datasets/hit-uav --split test --detector mindyolo
```

mAP@0.5 ranks every box the model gives, so it does not depend on the threshold. Precision, recall and false hits per image do.

Settings

- `NEUSE_DETECTOR` - Set to `mindyolo` to use the trained model
- `NEUSE_MINDYOLO_MODEL` - The converted model, defaults to `models/person_mindyolo.ms`
- `NEUSE_MINDYOLO_THRESHOLD` - The score a box needs to be reported, defaults to `0.25`. Set it from the validation images

Not done yet

- Training and scoring per flight group on ModelArts, for a fair comparison with the person check's 0.520 on unseen flights
- Measuring the model's speed on the unit's CPU
