import json

with open('./scannet/scannet_anno_list.json' , 'r') as f:
    anno_list = json.load(f)

entity_list = []
view_list = []
scene_list = []

for anno in anno_list:
    if anno['type'] == 'entity':
        entity_list.append(anno)
    elif anno['type'] == 'view':
        view_list.append(anno)
    elif anno['type'] == 'scene':
        scene_list.append(anno)

with open('./scannet/scannet_entity_list.json', 'w') as f:
    json.dump(entity_list, f)

with open('./scannet/scannet_view_list.json', 'w') as f:
    json.dump(view_list, f)

with open('./scannet/scannet_scene_list.json', 'w') as f:
    json.dump(scene_list, f)