import _init_paths

from utils import *
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *

import argparse
import json

ps = argparse.ArgumentParser()
ps.add_argument("label")
ps.add_argument("gt")
ps.add_argument("-t", "--threshold", type=float, default=0.5)
# arg = ps.parse_args(['../make-sense/face-coco.json','../make-sense/face-coco_new.json'])
arg = ps.parse_args()

with open(arg.label) as f:
    label = json.load(f)

with open(arg.gt) as f:
    gt = json.load(f)

# all boxes
boxes = BoundingBoxes()
for x in gt['annotations']:
    boxes.addBoundingBox(BoundingBox(
        x['image_id'],
        x['category_id'],
        *x['bbox'],
        bbType=BBType.GroundTruth
    ))

for x in label['annotations']:
    boxes.addBoundingBox(BoundingBox(
        x['image_id'],
        x['category_id'],
        *x['bbox'],
        classConfidence=x.get('score') or 1.,
        bbType=BBType.Detected
    ))

allClasses = [i['name'] for i in gt['categories']]

# start evaluation
res = Evaluator().GetPascalVOCMetrics(boxes, IOUThreshold=arg.threshold)
valid = [i['AP'] for i in res if i['total positives'] > 0]
print('mAP', sum(valid)/len(valid))
