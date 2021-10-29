import json
import sys
import argparse
import pandas as pd
import joblib
ps = argparse.ArgumentParser()
ps.add_argument('pkl_file')
ps.add_argument('out_file')
ps.add_argument('threshold', type=float, default=0.4, nargs='?')
args = ps.parse_args()

data = joblib.load(args.pkl_file)

# TODO: change this file more robust to other format path
images = [{"id": i, "file_name": k.split('/')[-1]} for i, k in enumerate(data)]

# open_images class id is start from 1, we change it to 0
categories = [{"id": i[3]-1, "name": i[2]} for i in pd.read_csv('open_images_500_cls.csv',header=None).itertuples()]
annos = []
for i, (k,v) in enumerate(data.items()):
    for j, lst in enumerate(v):
        for box in lst:
            if box[-1] >= args.threshold:
                box[2] -= box[0]
                box[3] -= box[1]
                annos.append({
                    "image_id": i,
                    "category_id": j,
                    "bbox": [int(round(x)) for x in box[:4]],
                    "score": float(box[-1]),
                })

js = {
    "images": images,
    "annotations": annos,
    "categories": categories,
}

with open(args.out_file, 'w') as f:
    json.dump(js, f)
