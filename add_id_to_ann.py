import json
import sys

jf, njf = sys.argv[1:3]

with open(jf) as f:
    js = json.load(f)

for i, x in enumerate(js['annotations']):
    x['id'] = i+1

with open(njf, 'w') as f:
    json.dump(js, f)
