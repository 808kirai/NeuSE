"""
reads the hit-uav thermal drone dataset
https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset, cc by 4.0
Suo et al., Scientific Data 10, 227 (2023)

uses the normal_json labels, which keep the dont care regions the yolo labels drop
images are 640 by 512, 8 bit grayscale
"""
import json
from pathlib import Path

# class ids in the dataset's json files
person_id = 0
bicycle_id = 2
dont_care_id = 4
class_names = {0: "person", 1: "car", 2: "bicycle", 3: "other vehicle", 4: "dont care"}

splits = ("train", "val", "test")


def parse_name(filename):
    # file names look like 1_130_90_0_04077.jpg
    # time 0 is day and 1 is night, then altitude in metres, camera angle in degrees, weather and frame number
    time, altitude, angle = Path(filename).stem.split("_")[:3]
    return {"time": "night" if time == "1" else "day", "altitude": int(altitude), "angle": int(angle)}


def load_split(root, split):
    # returns one record per image with its person boxes, dont care regions and other objects
    root = Path(root)
    with open(root / "normal_json" / "annotations" / f"{split}.json", encoding="utf-8") as f:
        data = json.load(f)

    records = {}
    for image in data["images"]:
        records[image["id"]] = {
            "file": image["filename"],
            "path": root / "normal_json" / split / image["filename"],
            "width": image["width"],
            "height": image["height"],
            "people": [],
            "ignore": [],
            "others": [],
            "bicycles": [],
            **parse_name(image["filename"]),
        }

    # the dataset calls this list annotation, not annotations like coco
    for label in data["annotation"]:
        record = records[label["image_id"]]
        box = list(label["bbox"])
        if label["category_id"] == person_id:
            record["people"].append(box)
        elif label["category_id"] == dont_care_id:
            record["ignore"].append(box)
        else:
            # cars, bicycles and other vehicles, bicycles are also listed on their own
            # the dataset marks a bike and its rider as one bicycle box, with no person box for the rider
            record["others"].append(box)
            if label["category_id"] == bicycle_id:
                record["bicycles"].append(box)

    # sorted by file name so every run goes through images in the same order
    return sorted(records.values(), key=lambda r: r["file"])
