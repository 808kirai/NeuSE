"""
turns hit-uav into a person only dataset for mindyolo training
cars, bikes and other vehicles are left out so the model learns to treat them as background
images with no people are kept as empty examples, so the model also learns what nothing looks like

the dataset labels a bike and its rider as one bicycle box and gives the rider no person box
so a person only model learns to ignore people on bikes, --with-bicycles keeps bicycles as a second class instead

writes
    images/<split>/*.jpg
    labels/<split>/*.txt, one "0 x_center y_center width height" line per person, scaled 0 to 1
    <split>.txt, a list of image paths per split
    hit_uav_person.yaml, the dataset section for a mindyolo config

usage
    python scripts/prepare_hit_uav.py --source C:/Users/you/datasets/hit-uav --out C:/Users/you/datasets/hit-uav-person
    add --data-root /home/ma-user/work/hit-uav-person when the folder will be uploaded to modelarts
    add --with-bicycles to keep bicycles and their riders as class 1
"""
import argparse
import shutil
import sys
from pathlib import Path

# put the project root on the path so src can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hit_uav import load_split, splits


def yolo_line(box, width, height, label=0):
    # one label line with the class, then the box centre and size scaled to the image
    x, y, w, h = box
    return f"{label} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} {w / width:.6f} {h / height:.6f}"


def prepare(source, out, data_root=None, with_bicycles=False):
    # converts every split and returns how many images and people each has
    out = Path(out)
    root = (data_root or out.as_posix()).rstrip("/")
    summary = {}

    for split in splits:
        records = load_split(source, split)
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

        listing = []
        for record in records:
            shutil.copy2(record["path"], out / "images" / split / record["file"])
            lines = [yolo_line(box, record["width"], record["height"]) for box in record["people"]]
            if with_bicycles:
                lines += [yolo_line(box, record["width"], record["height"], 1) for box in record["bicycles"]]
            label_file = out / "labels" / split / (Path(record["file"]).stem + ".txt")
            label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            # mindyolo reads paths starting with ./ as relative to this list file
            listing.append(f"./images/{split}/{record['file']}")

        (out / f"{split}.txt").write_text("\n".join(listing) + "\n", encoding="utf-8")
        summary[split] = {
            "images": len(records),
            "with_people": sum(1 for r in records if r["people"]),
            "people": sum(len(r["people"]) for r in records),
        }

    # mindyolo finds each label by swapping images for labels in the image path
    classes = "  nc: 2\n  names: [ 'person', 'bicycle' ]\n" if with_bicycles else "  nc: 1\n  names: [ 'person' ]\n"
    (out / "hit_uav_person.yaml").write_text(
        "# person only hit-uav, made by scripts/prepare_hit_uav.py\n"
        "# source https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset, cc by 4.0\n"
        "data:\n"
        "  dataset_name: hit_uav_person\n"
        f"  train_set: {root}/train.txt\n"
        f"  val_set: {root}/val.txt\n"
        f"  test_set: {root}/test.txt\n"
        + classes,
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="make a person only copy of hit-uav for mindyolo")
    parser.add_argument("--source", required=True, help="folder of the cloned hit-uav repository")
    parser.add_argument("--out", required=True, help="folder to write the person only dataset to")
    parser.add_argument("--data-root", help="where the folder will live when training, defaults to --out")
    parser.add_argument("--with-bicycles", action="store_true",
                        help="keep bicycles and their riders as a second class instead of background")
    args = parser.parse_args()

    summary = prepare(args.source, args.out, args.data_root, args.with_bicycles)
    for split, counts in summary.items():
        print(f"{split}: {counts['images']} images, {counts['with_people']} with people, {counts['people']} people")


if __name__ == "__main__":
    main()
