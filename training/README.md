Training the MindYOLO person detector on ModelArts

This trains a small YOLOv8 person detector with MindYOLO and MindSpore on Huawei Cloud ModelArts, converts it for MindSpore Lite and scores it with the same scripts as the person check, so the two can be compared fairly. The steps have not been run yet, since the team does not have Huawei Cloud access. The detector slot on the unit is tested with a stand in for the model.

To replace the person check, the trained model has to beat mAP@0.5 0.670 on the HIT-UAV test images. The person check's score on unseen flights, 0.520, is the second bar once the MindYOLO model is also trained per flight group.

Files

- `training/hit_uav_yolov8n.yaml` - MindYOLO config that starts from yolov8n and changes only what HIT-UAV needs
- `training/hit_uav_yolov8n_960.yaml` - The same with frames scaled up to 960, for small people
- `training/hit_uav_yolov8s.yaml` - The larger yolov8s, if the yolov8n runs fall short
- `scripts/prepare_hit_uav.py` - Makes the person only dataset in MindYOLO's format
- `src/mindyolo_detector.py` - Runs the converted model on the unit and turns its output into sightings

Which runs to try, and why

People in HIT-UAV are small. The median person box is 12 by 19 pixels, and one in ten is 8 by 14 or smaller. YOLOv8's finest grid has a cell every 8 pixels, so many people cover only one or two cells. The plan follows what published work found for this.

- Run 1, `hit_uav_yolov8n.yaml` at 640 with MindYOLO's own settings. Published YOLOv8 runs on HIT-UAV reach about 0.92 person AP50 at 640 with mostly default settings, so this is the run to try first
- Run 2, `hit_uav_yolov8n_960.yaml`. Scaling the input up is the simplest way to help small objects. On VisDrone, another drone dataset of small objects, raising the input from 640 to 1280 helped YOLOv8 far more than adding an extra detection layer for small objects, and needed no change to the network
- Run 3, `hit_uav_yolov8s.yaml`, only if both yolov8n runs fall short. It is about 3.5 times larger and slower

Pick between the runs on the validation images with `scripts/evaluate_hit_uav.py`, then score only the chosen one on the test images.

Left out on purpose

- An extra detection layer for small objects, known as a P2 head. MindYOLO has no ready config or COCO weights for it, so it would start from nothing. Try it only if run 2 still misses small people
- Cutting frames into tiles, as SAHI does. HIT-UAV frames are only 640 by 512, so tiles would do much the same as run 2 at a higher cost
- Flipping frames upside down. MindYOLO only flips left to right. The cameras look forward and down at an angle, so upside down frames would not look like real ones anyway

Checked in MindYOLO's own files

- Each training image is padded to 160 labels, and labels past that are dropped. The busiest HIT-UAV frame has 33 people, so even a mosaic of four frames stays near 100
- `random_perspective` scales images between half and one and a half times their size. Half size makes a small person about 6 by 10 pixels. That is MindYOLO's and Ultralytics' default, and the published runs used it, so it is kept. If small people are missed, lower `scale` to 0.25 in the config's first `random_perspective`

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

For run 2, change the config to `hit_uav_yolov8n_960.yaml` and keep the same weights. For run 3, use `hit_uav_yolov8s.yaml` with the yolov8s COCO weights from the same README. At 960, if the notebook runs out of memory, halve `per_batch_size`.

5. Export and convert

```bash
python deploy/export.py --config configs/yolov8/hit_uav_yolov8n.yaml --weight ./runs/hit_uav/weights/BEST.ckpt --file_format MINDIR --device_target Ascend
converter_lite --fmk=MINDIR --modelFile=hit_uav_yolov8n.mindir --outputFile=person_mindyolo
```

Use the same config the model was trained with, so it is exported at the same input size. Replace `BEST.ckpt` with the checkpoint that scored best in the training log. The second command needs MindSpore Lite's converter tool, from the MindSpore Lite download for the unit's system. It writes `person_mindyolo.ms`. Put that file in the repository's `models/` folder on the unit. It is kept out of git.

6. Run it on the unit

Install MindSpore Lite's Python package for the unit's system, then pick the detector.

```bash
NEUSE_DETECTOR=mindyolo python scripts/simulate_feed.py --replay ../datasets/hit-uav/normal_json/test --frames 30
```

The unit reads the input size from the model, so a model trained at 960 gets 960 by 960 frames without any setting.

On the Orange Pi AIpro, set `NEUSE_MINDYOLO_DEVICE=ascend` to run the model on its Ascend NPU instead of the CPU. MindSpore Lite needs the model converted for Ascend, by adding `--optimize=ascend_oriented` to the conversion in step 5, on a computer with the same CANN version as the board. MindSpore Lite's Ascend documents list the 310P chip but not the 310B in the Orange Pi AIpro, so check this on the board first. If it does not work there, convert the model with the board's own CANN tools.

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
- `NEUSE_MINDYOLO_DEVICE` - `cpu` by default, or `ascend` to run on an Ascend NPU

Not done yet

- Training and scoring per flight group on ModelArts, for a fair comparison with the person check's 0.520 on unseen flights
- Measuring the model's speed on the unit's CPU and on the Orange Pi AIpro's NPU

Sources

- HIT-UAV, the dataset and its first detector scores, https://arxiv.org/abs/2204.03245
- YOLOv8s on HIT-UAV at 640, person AP50 0.923, https://arxiv.org/abs/2507.06797
- BDK-YOLOv8, built on YOLOv8n and scored on HIT-UAV, https://doaj.org/article/f8e5102478a340ecad5f8068ecd8798a
- Input size against an extra small object layer on VisDrone, a preprint not yet reviewed, https://www.preprints.org/manuscript/202603.0990
- MindSpore Lite conversion for Ascend, https://www.mindspore.cn/lite/docs/en/master/use/cloud_infer/converter_tool_ascend.html
