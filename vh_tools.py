import matplotlib.pyplot as plt
import re
import random
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import json


def get_random_goal(graph, episode_id):
    with open('data_input/env_to_pred_random_name.json', 'r') as f:
        env_to_pred = json.load(f)
    
    predicted_goal_class = env_to_pred[str(episode_id)]
    # graph = arena.env.graph
    idnodes = {}
    valid_ids = []
    for class_name in ['fridge', 'dishwasher', 'kitchentable', 'coffeetable', 'sofa']:
        idnodes[class_name] = [node['id'] for node in graph['nodes'] if node['class_name'] == class_name][0]
        valid_ids += [node['id'] for node in graph['nodes'] if node['class_name'] == class_name]

    predicted_goal = {}
    for kpred, itv in predicted_goal_class.items():
        spl = kpred.split('_')
        if spl[-1].isdigit():
            if int(spl[-1]) not in valid_ids:
                print("ERRRO IN GOAL")
            target_name = '{}_{}_{}'.format(spl[0], spl[1], spl[-1])

        else:
            if not spl[-1] in idnodes:
                if spl[0] == 'sit':
                   target_name = '_'.join([spl[0], '1', spl[1]])
                else:
                    target_name = '_'.join([spl[0], spl[1], '1'])
            else:
                id_target = idnodes[spl[-1]]
                if spl[0] == 'sit':
                    spl[1] = '1'
                target_name = '{}_{}_{}'.format(spl[0], spl[1], id_target)
        predicted_goal[target_name] = itv
    return predicted_goal

def get_predicted_goal(graph, episode_id):
    with open('data_input/env_to_pred.json', 'r') as f:
        env_to_pred = json.load(f)
    
    predicted_goal_class = env_to_pred[str(episode_id)]
    # graph = arena.env.graph
    idnodes = {}
    for class_name in ['fridge', 'dishwasher', 'kitchentable', 'coffeetable', 'sofa']:
        idnodes[class_name] = [node['id'] for node in graph['nodes'] if node['class_name'] == class_name][0]

    predicted_goal = {}
    for kpred, itv in predicted_goal_class.items():
        spl = kpred.split('_')
        if not spl[-1] in idnodes:
            if spl[0] == 'sit':
               target_name = '_'.join([spl[0], '1', spl[1]])
            else:
                target_name = '_'.join([spl[0], spl[1], '1'])
        else:
            id_target = idnodes[spl[-1]]
            if spl[0] == 'sit':
                spl[1] = '1'
            target_name = '{}_{}_{}'.format(spl[0], spl[1], id_target)
        predicted_goal[target_name] = itv
    return predicted_goal

def random_goal(graph):
    preds = [
    "on_plate_kitchentable",
    "on_waterglass_kitchentable",
    "on_wineglass_kitchentable",
    "on_cutleryfork_kitchentable",
    "inside_plate_dishwasher",
    "inside_waterglass_dishwasher",
    "inside_wineglass_dishwasher",
    "inside_cutleryfork_dishwasher",
    "inside_cupcake_fridge",
    "inside_juice_fridge",
    "inside_pancake_fridge",
    "inside_poundcake_fridge",
    "inside_wine_fridge",
    "inside_pudding_fridge",
    "inside_apple_fridge",
    "on_cupcake_kitchentable",
    "on_juice_kitchentable",
    "on_pancake_kitchentable",
    "on_poundcake_kitchentable",
    "on_wine_kitchentable",
    "on_pudding_kitchentable",
    "on_apple_kitchentable",
    "on_coffeepot_kitchentable",
    "on_cupcake_coffeetable",
    "on_juice_coffeetable",
    "on_wine_coffeetable",
    "on_pudding_coffeetable",
    "on_apple_coffeetable"]

    idnodes = {}
    for class_name in ['fridge', 'dishwasher', 'kitchentable', 'coffeetable']:
        idnodes[class_name] = [node['id'] for node in graph['nodes'] if node['class_name'] == class_name][0]

    preds_selected = random.choices(preds, k=6)
    dict_preds = {}
    for pred in preds_selected:

        spl = pred.split('_')
        id_target = idnodes[spl[-1]]
        target_name = '{}_{}_{}'.format(spl[0], spl[1], id_target)
        dict_preds[target_name] = random.choice([0,1,2])
    return dict_preds



dict_objects_actions = {
    "objects_inside": [
        "bathroomcabinet", 
        "kitchencabinets", 
        "cabinet", 
        "fridge", 
        "oven", 
        "stove", 
        "microwave",
        "dishwasher"
    ],
    "objects_surface": [
        "bench",
         "cabinet",
         "chair",
         "coffeetable",
         "desk",
         "kitchencounter",
         "kitchentable",
         "nightstand",
         "sofa"],

    "objects_grab": [
         "apple",
         "book",
         "coffeepot",
         "cupcake",
         "cutleryfork",
         "juice",
         "pancake",
         "plate",
         "poundcake",
         "pudding",
         "remotecontrol",
         "waterglass",
         "whippedcream",
         "wine",
         "wineglass"
     ]
}
ignore_class = ['window', 'door', 'doorjamb', 'floor', 'wall', 
            'ceiling', 'ceilinglamp', 'hanger', 'closetdrawer', 'wallpictureframe', 
            'rug', 'folder', 'box', 'pillow']



def get_visible_graph(graph, agent_id=1, full_obs=False):
    if full_obs:
        ids_visible = [node['id'] for node in graph['nodes'] if node['class_name'] not in ignore_class]
    else:
        ids_visible = get_objects_visible(None, graph, agent_id)
    curr_g = {
        'nodes': [node for node in graph['nodes'] if node['id'] in ids_visible],
        'edges': [edge for edge in graph['edges'] if edge['from_id'] in ids_visible and edge['to_id'] in ids_visible]
    }
    curr_g = inside_not_trans(curr_g) 
    return curr_g


def get_objects_visible(ids, graph, agent_id=1, ignore_bad_class=False, full_obs=False):
    if full_obs:
        id2node = {node['id']: node for node in graph['nodes']}
        visible_ids = [node['id'] for node in graph['nodes']]
        if ignore_bad_class:
            visible_ids = [id_node for id_node in visible_ids if id2node[id_node]['class_name'].lower() not in get_classes_ignore()]
        return visible_ids
    # Return only the objects that are inside the room and not somewhere closed
    if ids is None:
        curr_room = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == agent_id and edge['relation_type'] == 'INSIDE'][0]
        # Objects inside room
        ids = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == curr_room and edge['relation_type'] == 'INSIDE']
        
        # Objects inside objects inside room
        ids += [edge['from_id'] for edge in graph['edges'] if edge['to_id'] in ids and edge['relation_type'] == "INSIDE"]
        ids += [curr_room]
        ids = list(set(ids))
    
    id2node = {node['id']: node for node in graph['nodes']}
    inside_no_room = {}
    for edge in graph['edges']:
        if edge['relation_type'] == 'INSIDE' and id2node[edge['to_id']]['category'] != 'Rooms':
            if edge['from_id'] in inside_no_room:
                pdb.set_trace()
            inside_no_room[edge['from_id']] = edge['to_id']
    
    visible_ids = []
    for curr_id in ids:
        if curr_id not in inside_no_room:
            # If the object is just inside room, visualize it
            visible_ids.append(curr_id)
        else:
            node_inside = id2node[inside_no_room[curr_id]]
            if 'OPEN' in node_inside['states']:
                visible_ids.append(curr_id)
    if ignore_bad_class:
        visible_ids = [id_node for id_node in visible_ids if id2node[id_node]['class_name'].lower() not in get_classes_ignore()]
    return visible_ids

def get_classes_ignore():
    return ignore_class

def parse(action_str):
    patt_action = r'^\[(\w+)\]'
    patt_params = r'\<(.+?)\>\s*\((.+?)\)'

    action_match = re.search(patt_action, action_str.strip())
    action_string = action_match.group(1)
    param_match = re.search(patt_params, action_match.string[action_match.end(1):])
    params = []
    object_id = -1
    object_name = ''
    while param_match:
        params.append((param_match.group(1), int(param_match.group(2))))
        param_match = re.search(patt_params, param_match.string[param_match.end(2):])
    if len(params) > 0:
        object_id = params[0][1]
        object_name = params[0][0]
    return action_string, object_name, object_id

def can_perform_action(action, object_name, object_id, current_graph, agent_id=1):
    graph_helper = None
    graph = current_graph
    obj2_str = ''
    obj1_str = ''
    o1_id = object_id
    id2node = {node['id']: node for node in graph['nodes']}
    grabbed_objects = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == agent_id and edge['relation_type'] in ['HOLDS_RH', 'HOLD_LH']]
    close_edge = len([edge['to_id'] for edge in graph['edges'] if edge['from_id'] == agent_id and edge['to_id'] == o1_id and edge['relation_type'] == 'CLOSE']) > 0
    if action == 'grab':
        if len(grabbed_objects) > 0:
            return None, {'msg': 'You cannot grab an already grabbed object'}
    if action.startswith('walk'):
        if o1_id in grabbed_objects:
            return None, {'msg': 'You cannot walk towards an object you grabbed'}

    if o1_id == agent_id:
        return None, {'msg': 'You cannot walk towards yourself'}
    
    if (action in ['grab', 'open', 'close']) and not close_edge:
        return None, {'msg': 'You are not close enought to {}'.format(object_name)}

    if action == 'close':
        if 'CLOSED' in id2node[o1_id]['states'] or 'OPEN' not in id2node[o1_id]['states']:
            message = 'You cannot close {}'.format(object_name)
            return None, {'msg': message}

    if 'put' in action:
        if len(grabbed_objects) == 0:
            return None, {'msg': 'You did not grab any object'}
        else:
            o2_id = grabbed_objects[0]
            if o2_id == o1_id:
                return None, {'msg': 'cannot put an object into itself'}
            o2 = id2node[o2_id]['class_name']
            obj2_str = f'<{o2}> ({o2_id})'

    if object_name is not None:
        obj1_str = f'<{object_name}> ({object_id})'
        if o1_id in id2node.keys():
            if id2node[o1_id]['class_name'] == 'character':
                return None, {'msg': 'You cannot interact with another agent'}

    if action.startswith('put'):
        if graph_helper is not None:
            if id2node[o1_id]['class_name'] in graph_helper.object_dict_types['objects_inside']:
                action = 'putin'
            if id2node[o1_id]['class_name'] in graph_helper.object_dict_types['objects_surface']:
                action = 'putback'
        else:
            if 'CONTAINERS' in id2node[o1_id]['properties']:
                action = 'putin'
            elif 'SURFACES' in id2node[o1_id]['properties']:
                action = 'putback'


    action_str = f'[{action}] {obj2_str} {obj1_str}'.strip()
    return action_str, {'msg': "Success"}


def graph_info(graph, id2classid, ids_select=None, restrictive=True):
    """ Return objects and actions that can be performed now """
    id2node = {node['id']: node for node in graph['nodes']}
    current_room = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == 1 and edge['relation_type'] == 'INSIDE'][0]

    objects_room = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == current_room and edge['relation_type'] == 'INSIDE']
    objects_close = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == 1 and edge['relation_type'] == 'CLOSE']
    objects_grabbed = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == 1 and 'HOLDS' in edge['relation_type']]
    objects_grabbed_second = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == 2 and 'HOLDS' in edge['relation_type']]



    if ids_select is None:
        ids = get_objects_visible(None, graph, agent_id=1, full_obs=True)
        objects_interact = get_objects_visible(None, graph, agent_id=1)
        # ids = set(objects_room)
    else:
        ids = ids_select

    second_char_vis = 2 in ids
    objects_and_actions = []
    

    classes_open = dict_objects_actions['objects_inside']
    classes_surface = dict_objects_actions['objects_surface']
    classes_grab = dict_objects_actions['objects_grab']

    classes_switch = ['faucet', 'toaster', 'microwave', 'stove', 'lightswitch', 'television', 'tv']
    
    ids = sorted(ids, key=lambda idi: (0 if idi in objects_interact else 0, id2node[idi]['class_name'], idi))

    # Containers
    containers = {}
    rooms_obj = {}
    for edge in graph['edges']:
        if edge['relation_type'] == 'INSIDE' and id2node[edge['to_id']]['class_name'] in classes_open:
            containers[edge['from_id']] = edge['to_id']
        if edge['relation_type'] == 'ON' and id2node[edge['to_id']]['class_name'] in classes_surface:
            if edge['from_id'] not in containers:
                containers[edge['from_id']] = edge['to_id']
        if edge['relation_type'] == 'INSIDE' and id2node[edge['to_id']]['category'] == 'Rooms':
            rooms_obj[edge['from_id']] = edge['to_id']

    for id_obj in ids:
        if id_obj in objects_grabbed or id_obj in objects_grabbed_second:
            continue
            # actions.append('walktowards')

        if id_obj == 1:
            continue
        actions = []
        node = id2node[id_obj]
        if node['class_name'] in ignore_class or 'clothes' in node['class_name'] or 'character' in node['class_name']:
            continue

        if restrictive:
            if node['class_name'] not in (classes_open + classes_surface + classes_grab):
                continue
        if id_obj not in objects_grabbed:
            actions.append('walktowards')

        is_open = 'OPEN' in node['states']
        if id_obj in objects_close:
            if 'GRABBABLE' in node['properties'] and id_obj not in objects_grabbed and id2node[id_obj]['class_name'] in classes_grab:
                actions.append('grab')
            if 'CAN_OPEN' in node['properties'] and node['class_name'] in classes_open:
                if 'OPEN' in node['states']:
                    actions.append('close')
                else:
                    actions.append('open')
            if node['class_name'] in classes_switch:
                if 'ON' in node['states']:
                    actions.append('switchoff')
                else:
                    actions.append('switchon')

            if len(objects_grabbed) > 0:
                if node['class_name'] in classes_surface or node['class_name'] in classes_open and 'OPEN' in node['states']:
                    actions.append('put object')


        if node['class_name'] in classes_grab:
            obj_type = 'grab'
        elif node['class_name'] in classes_open:

            obj_type = 'container'
        else:
            obj_type = 'surface';

        is_close = '1' if id_obj in objects_close else '0'
        container = -1
        room_obj = rooms_obj[id_obj]
        # print(rooms_obj)
        room_name = id2node[room_obj]['class_name']
        # to what container it belongs
        if id_obj in containers:
            container = containers[id_obj]

        object_curr = [id_obj, node['class_name'], id2classid[id_obj], obj_type, is_close, is_open, container, room_obj, room_name]
        objects_and_actions.append((object_curr, actions))
    
    
    visible_objects = [v[0] for v in objects_and_actions]
    grabbed_second = -1
    if second_char_vis:
        if len(objects_grabbed_second) == 0:
            grabbed_second = []
        else:
            grabbed_second = [
                    id2node[objects_grabbed_second[0]]['class_name'], 
                    str(objects_grabbed_second[0]), 
                    id2classid[objects_grabbed_second[0]]
                    ]
    grabbed_first = [] if len(objects_grabbed) == 0 else [id2node[objects_grabbed[0]]['class_name'], str(objects_grabbed[0]), str(id2classid[int(objects_grabbed[0])])]
    print('====')
    print("SECOND GRABBED OBJECTS", grabbed_second)
    print(grabbed_first)
    print('___')
    objects_relations = {
            'grabbed_object': grabbed_first,
            'grabbed_object_second': grabbed_second,
            'close_objects': objects_close,
            'interactable_objects': objects_interact,
            'visible_objects': visible_objects,
            'inside_objects': [current_room],
            'current_room': id2node[current_room]['class_name']
    }
    return objects_and_actions, objects_relations



def reduced_node_info(node):
    # Display only id, state and coords
    new_node = {
        'id': node['id'],
        'states': node['states'],
        'bounding_box': node['bounding_box'],
        'obj_transform': node['obj_transform']
    }
    return new_node




def check_progress(state, goal_spec):
    """TODO: add more predicate checkers; currently only ON"""
    print(goal_spec)
    unsatisfied = {}
    satisfied = {}
    reward = 0.
    id2node = {node['id']: node for node in state['nodes']}

    for key, value in goal_spec.items():
        if type(value) == int:
            value = [value]
        elements = key.split('_')
        unsatisfied[key] = value[0] if elements[0] not in ['offOn', 'offInside'] else 0
        satisfied[key] = [None] * 2
        satisfied[key]
        satisfied[key] = []
        for edge in state['edges']:
            if elements[0] in 'close':
                if edge['relation_type'].lower().startswith('close') and id2node[edge['to_id']]['class_name'] == elements[1] and edge['from_id'] == int(elements[2]):
                    predicate = '{}_{}_{}'.format(elements[0], edge['to_id'], elements[2])
                    satisfied[key].append(predicate)
                    unsatisfied[key] -= 1
            if elements[0] in ['on', 'inside']:
                if edge['relation_type'].lower() == elements[0] and edge['to_id'] == int(elements[2]) and (id2node[edge['from_id']]['class_name'] == elements[1] or str(edge['from_id']) == elements[1]):
                    predicate = '{}_{}_{}'.format(elements[0], edge['from_id'], elements[2])
                    satisfied[key].append(predicate)
                    unsatisfied[key] -= 1
            elif elements[0] == 'offOn':
                if edge['relation_type'].lower() == 'on' and edge['to_id'] == int(elements[2]) and (id2node[edge['from_id']]['class_name'] == elements[1] or str(edge['from_id']) == elements[1]):
                    predicate = '{}_{}_{}'.format(elements[0], edge['from_id'], elements[2])
                    unsatisfied[key] += 1
            elif elements[0] == 'offInside':
                if edge['relation_type'].lower() == 'inside' and edge['to_id'] == int(elements[2]) and (id2node[edge['from_id']]['class_name'] == elements[1] or str(edge['from_id']) == elements[1]):
                    predicate = '{}_{}_{}'.format(elements[0], edge['from_id'], elements[2])
                    unsatisfied[key] += 1
            elif elements[0] == 'holds':
                if edge['relation_type'].lower().startswith('holds') and id2node[edge['to_id']]['class_name'] == elements[1] and edge['from_id'] == int(elements[2]):
                    predicate = '{}_{}_{}'.format(elements[0], edge['to_id'], elements[2])
                    satisfied[key].append(predicate)
                    unsatisfied[key] -= 1
            elif elements[0] == 'sit':
                if edge['relation_type'].lower().startswith('sit') and edge['to_id'] == int(elements[2]) and edge['from_id'] == int(elements[1]):
                    predicate = '{}_{}_{}'.format(elements[0], edge['to_id'], elements[2])
                    satisfied[key].append(predicate)
                    unsatisfied[key] -= 1
        if elements[0] == 'turnOn':
            if 'ON' in id2node[int(elements[1])]['states']:
                predicate = '{}_{}_{}'.format(elements[0], elements[1], 1)
                satisfied[key].append(predicate)
                unsatisfied[key] -= 1
    return satisfied, unsatisfied

def inside_not_trans(graph):
    id2node = {node['id']: node for node in graph['nodes']}
    parents = {}
    grabbed_objs = []
    for edge in graph['edges']:
        if edge['relation_type'] == 'INSIDE':

            if edge['from_id'] not in parents:
                parents[edge['from_id']] = [edge['to_id']]
            else:
                parents[edge['from_id']] += [edge['to_id']]
        elif edge['relation_type'].startswith('HOLDS'):
            grabbed_objs.append(edge['to_id'])

    edges = []
    for edge in graph['edges']:
        if edge['relation_type'] == 'INSIDE' and id2node[edge['to_id']]['category'] == 'Rooms':
            if len(parents[edge['from_id']]) == 1:
                edges.append(edge)
            
        else:
            edges.append(edge)
    graph['edges'] = edges

    # # add missed edges
    # missed_edges = []
    # for obj_id, action in self.obj2action.items():
    #     elements = action.split(' ')
    #     if elements[0] == '[putback]':
    #         surface_id = int(elements[-1][1:-1])
    #         found = False
    #         for edge in edges:
    #             if edge['relation_type'] == 'ON' and edge['from_id'] == obj_id and edge['to_id'] == surface_id:
    #                 found = True
    #                 break
    #         if not found:
    #             missed_edges.append({'from_id': obj_id, 'relation_type': 'ON', 'to_id': surface_id})
    # graph['edges'] += missed_edges

    parent_for_node = {}

    char_close = {1: [], 2: []}
    for char_id in range(1, 3):
        for edge in graph['edges']:
            if edge['relation_type'] == 'CLOSE':
                if edge['from_id'] == char_id and edge['to_id'] not in char_close[char_id]:
                    char_close[char_id].append(edge['to_id'])
                elif edge['to_id'] == char_id and edge['from_id'] not in char_close[char_id]:
                    char_close[char_id].append(edge['from_id'])
    ## Check that each node has at most one parent
    for edge in graph['edges']:
        if edge['relation_type'] == 'INSIDE':
            if edge['from_id'] in parent_for_node and not id2node[edge['from_id']]['class_name'].startswith('closet'):
                print('{} has > 1 parent'.format(edge['from_id']))
                pdb.set_trace()
                raise Exception
            parent_for_node[edge['from_id']] = edge['to_id']
            # add close edge between objects in a container and the character
            if id2node[edge['to_id']]['class_name'] in ['fridge', 'kitchencabinets', 'cabinet', 'microwave',
                                                        'dishwasher', 'stove']:
                for char_id in range(1, 3):
                    if edge['to_id'] in char_close[char_id] and edge['from_id'] not in char_close[char_id]:
                        graph['edges'].append({
                            'from_id': edge['from_id'],
                            'relation_type': 'CLOSE',
                            'to_id': char_id
                        })
                        graph['edges'].append({
                            'from_id': char_id,
                            'relation_type': 'CLOSE',
                            'to_id': edge['from_id']
                        })

    ## Check that all nodes except rooms have one parent
    nodes_not_rooms = [node['id'] for node in graph['nodes'] if node['category'] not in ['Rooms', 'Doors']]
    nodes_without_parent = list(set(nodes_not_rooms) - set(parent_for_node.keys()))
    nodes_without_parent = [node for node in nodes_without_parent if node not in grabbed_objs]
    if len(nodes_without_parent) > 0:
        for nd in nodes_without_parent:
            print(id2node[nd])
        # pdb.set_trace()
        raise Exception

    return graph
