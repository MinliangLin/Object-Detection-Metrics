from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import argparse

ps = argparse.ArgumentParser()
ps.add_argument("pred")
ps.add_argument("gt")
ps.add_argument("-t", "--threshold", type=float, default=0.5)
arg = ps.parse_args()

gt = COCO(arg.gt)
pred = COCO(arg.pred)
for x in gt.dataset['annotations']:
    x.setdefault('iscrowd', 0)
    x.setdefault('area', x['bbox'][2]*x['bbox'][3])
for x in pred.dataset['annotations']:
    x.setdefault('score', 1.0)
    x.setdefault('iscrowd', 0)
    x.setdefault('area', x['bbox'][2]*x['bbox'][3])

cocoEval = COCOeval(gt, pred, 'bbox')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
