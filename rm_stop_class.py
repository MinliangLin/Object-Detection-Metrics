import json
import sys

jf, njf = sys.argv[1:3]
stop_cls = '''Human face
Man
Human head
Human hair
Human nose
Woman
Human eye
Human arm
Human mouth
Person
Human ear
Tree
Human hand
Human beard
Dress
Girl
Jellyfish'''.split('\n')

with open(jf) as f:
    js = json.load(f)

categ_dc = {i['name']:i['id'] for i in js['categories']}
stop_cls = {categ_dc[i] for i in stop_cls}
js['annotations'] = [i for i in js['annotations'] if i['category_id'] not in stop_cls] 

with open(njf, 'w') as f:
    json.dump(js, f)
