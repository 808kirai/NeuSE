"""
scores detector boxes against labelled boxes
all boxes are x, y, width and height in pixels
used for the classical baseline now and for the trained model later, so both are judged the same way
"""


def iou(a, b):
    # overlap of two boxes divided by the area they cover together
    # 1 means the same box, 0 means no overlap
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def center_inside(det, box):
    # true when the centre of det lands inside box
    cx = det[0] + det[2] / 2
    cy = det[1] + det[3] / 2
    return box[0] <= cx <= box[0] + box[2] and box[1] <= cy <= box[1] + box[3]


def match_image(detections, people, ignore=(), others=(), rule="iou", iou_threshold=0.5):
    # counts hits and misses for one image
    # detections are the detector's boxes and people are the labelled person boxes
    # ignore holds dont care regions, hits there are neither right nor wrong
    # others holds labelled boxes that are not people, like cars, so false hits on them can be counted
    # rule "iou" needs the boxes to overlap by iou_threshold, the usual detection standard
    # rule "center" only needs the hit's centre inside a person box, a looser check for blob detectors

    # every detection and person pair that could count as a match
    pairs = []
    for d, det in enumerate(detections):
        for p, person in enumerate(people):
            overlap = iou(det, person)
            if rule == "iou" and overlap >= iou_threshold:
                pairs.append((overlap, d, p))
            elif rule == "center" and center_inside(det, person):
                pairs.append((overlap, d, p))

    # best overlaps claim their match first, so the result does not depend on detection order
    # each detection and each person can only be used once
    pairs.sort(reverse=True)
    used_dets, used_people = set(), set()
    for _, d, p in pairs:
        if d not in used_dets and p not in used_people:
            used_dets.add(d)
            used_people.add(p)

    counts = {"tp": len(used_dets), "fp": 0, "fn": len(people) - len(used_people), "fp_on_other": 0, "ignored": 0}
    for d, det in enumerate(detections):
        if d in used_dets:
            continue
        if any(center_inside(det, box) for box in ignore):
            counts["ignored"] += 1
            continue
        counts["fp"] += 1
        # a false hit sitting on a car or bike shows the detector cannot tell objects apart
        if any(center_inside(det, box) for box in others):
            counts["fp_on_other"] += 1
    return counts


def add_counts(total, counts):
    # adds one image's counts into a running total
    for key, value in counts.items():
        total[key] = total.get(key, 0) + value
    return total


def average_precision(images, iou_threshold=0.5):
    # area under the precision and recall curve, the number yolo tools report as map@0.5
    # images is a list of (detections, people, ignore), each detection a (box, score) pair
    # uses the coco method of 101 recall points so it lines up with mindyolo's own results
    ranked = []
    total_people = 0
    for index, (detections, people, ignore) in enumerate(images):
        total_people += len(people)
        for box, score in detections:
            ranked.append((score, index, box))
    if total_people == 0:
        return 0.0

    # highest scores claim people first, ties keep their original order
    ranked.sort(key=lambda item: -item[0])
    claimed = [set() for _ in images]
    hits = []
    for _, index, box in ranked:
        people, ignore = images[index][1], images[index][2]
        best, best_overlap = None, iou_threshold
        for p, person in enumerate(people):
            overlap = iou(box, person)
            if p not in claimed[index] and overlap >= best_overlap:
                best, best_overlap = p, overlap
        if best is not None:
            claimed[index].add(best)
            hits.append(True)
        elif any(center_inside(box, region) for region in ignore):
            # hits on dont care regions count neither way
            continue
        else:
            hits.append(False)

    # precision and recall after each detection in score order
    precisions, recalls = [], []
    found = 0
    for n, hit in enumerate(hits, start=1):
        found += hit
        precisions.append(found / n)
        recalls.append(found / total_people)

    # each precision becomes the best precision at that recall or higher
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # read precision at recall 0, 0.01, up to 1 and average
    total, i = 0.0, 0
    for step in range(101):
        target = step / 100
        while i < len(recalls) and recalls[i] < target:
            i += 1
        if i < len(recalls):
            total += precisions[i]
    return total / 101


def summarize(total):
    # precision is the share of hits that were people
    # recall is the share of people that were found
    tp, fp, fn = total.get("tp", 0), total.get("fp", 0), total.get("fn", 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, **total}
