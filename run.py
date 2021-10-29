import _init_paths

from utils import *
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes

import argparse
import json

ps = argparse.ArgumentParser()
ps.add_argument("pred")
ps.add_argument("gt")
ps.add_argument("-t", "--threshold", type=float, default=0.5)
ps.add_argument("-c", "--simulate_coco", action="store_true")
ps.add_argument("-m", "--remove_empty", action="store_true")
args = ps.parse_args()

with open(args.pred) as f:
    pred = json.load(f)

with open(args.gt) as f:
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

for x in pred['annotations']:
    boxes.addBoundingBox(BoundingBox(
        x['image_id'],
        x['category_id'],
        *x['bbox'],
        classConfidence=x.get('score') or 1.,
        bbType=BBType.Detected
    ))

allClasses = [i['name'] for i in gt['categories']]
print('pred len', len(pred['annotations']))

# start evaluation
if args.simulate_coco:
    from Evaluator2 import *
    res = Evaluator().GetPascalVOCMetrics(boxes, IOUThreshold=args.threshold, method='101pts')
    # print('debug', [(i['class'], i['total TP']) for i in res if i['total positives'] > 0])
else:
    from Evaluator import *
    res = Evaluator().GetPascalVOCMetrics(boxes, IOUThreshold=args.threshold)

print('mAP (P>0):', np.mean([i['AP'] for i in res if i['total positives'] > 0]))

import pandas as pd
import numpy as np
id_to_cls = {c['id']:c['name'] for c in gt['categories']}
a = pd.DataFrame([(
    r['class'],
    id_to_cls[r['class']],
    r['total positives'],
    r['AP'],
    np.quantile(r['recall'], [0.1, 0.25, 0.5, 0.75, 0.9]) if len(r['recall']) > 0 else [],
    np.quantile(r['precision'], [0.1, 0.25, 0.5, 0.75, 0.9]) if len(r['recall']) > 0 else []) for r in res])
a.columns = ['id', 'name', 'P', 'AP', 'recall(0.1-0.9)', 'precision(0.1-0.9)']
a = a.dropna()[a['AP']>1e-6]
a.to_csv('evaluation_result.csv')
