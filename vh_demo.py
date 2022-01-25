#!flask/bin/python
#import cStringIO
import numpy as np
import torch
import time
import json
import ipdb
import sys, os
from collections import namedtuple


sys.path.append('..')
from tools.plotting_code_planner import plot_graph_2d_v2 as plot_graph_2d

# import plotly.io
sys.path.append(os.path.join('virtualhome', 'simulation'))
import argparse
import pickle as pkl
from flask import Flask, render_template, request, redirect, Response, send_file
from virtualhome.simulation.unity_simulator import comm_unity
import vh_tools
import random, json
import cv2
from datetime import datetime
import time
from PIL import Image
import io
import base64

import matplotlib.pyplot as plt

app = Flask(__name__)


image_top = None
comm = None
lc = None
instance_colors = None
current_image = None
images = None
prev_images = None
graph = None
id2node = None
aspect_ratio = 9./16.
bad_class_name_ids = []
curr_task = None
last_completed = {}

extra_agent = None
next_helper_action = None

# parameters for record graph
time_start = None
record_graph_flag = True # False
#vis_graph_flag = True
graph_save_dir = None

record_step_counter = 0

# Contains a mapping so that objects have a smaller id. One unique pr object+instance
# instead of instance. glass.234, glass.267 bcecome glass.1 glass.2
class2numobj = {}
id2classid = {}

task_index = -1 # for indexing task_id in task_group
task_index_shuffle = []

current_goal = {}

parser = argparse.ArgumentParser(description='Collection data simulator.')
parser.add_argument("--deployment", type=str, choices=["local", "remote"], default="remote")
parser.add_argument("--simversion", type=str, choices=["v1", "v2"], default="v1")
parser.add_argument("--portflask", type=int, default=5005)
parser.add_argument("--execname", type=str, help="The location of the executable name")
parser.add_argument("--portvh", type=int, default=8080)
parser.add_argument("--task_id", type=int, default=0)
parser.add_argument('--showmodal', action='store_true')
parser.add_argument("--task_group", type=int, nargs="+", default=[0], help='usage: --task group 41 42 43 44')
parser.add_argument("--exp_name", type=str, default="debugtest")
parser.add_argument("--extra_agent", type=str, nargs="+", default=['none']) # none, planner, rl_mcts


args = parser.parse_args()

def convert_image(img_array):
    #cv2.imwrite('current.png', img_array.astype('uint8'))
    img = Image.fromarray(img_array.astype('uint8'))
    #file_like = cStringIO.cStringIO(img)
    file_object = io.BytesIO()
    img.save(file_object, 'JPEG')
    img_str = base64.b64encode(file_object.getvalue())
    return img_str
    #img.save(img, 'current.png')
    #file_object.seek(0)
    #return file_object

def send_command(command):
    global graph, id2classid, record_step_counter, time_start, last_completed, next_helper_action, curr_task
    global task_index, current_goal, extra_agent
    executed_instruction = False
    info = {}

    if task_index >= len(args.task_group):
        return [], {'all_done': True}

    if command['instruction'] == 'reset':
        # executed_instruction = True
        if 'data_form' in command:
            with open(os.path.join(graph_save_dir, 'final_form.json'), 'w') as f:
                f.write(json.dumps(command['data_form']))

        images, info = reset(command['scene_id'])
        if info['all_done']:
            return [], {'all_done': True}


        # return images, info
    else:    
        instr = command['instruction']
        if instr == 'refresh':
            image_top = 'include_top' in command
            if image_top:
                info['image_top'] = get_top_image()

        else:
            other_info = None
            if 'other_info' in command:
                other_info = command['other_info']

            action = instr
            if other_info is not None:
                object_name, object_id = other_info[0]
            else:
                object_name, object_id = None, None
            script, message = vh_tools.can_perform_action(action, object_name, object_id, graph)
            if script is not None:
                executed_instruction = True
                script = ['<char0> {}'.format(script)]
                
                if extra_agent is not None:
                    command_other = next_helper_action
                    if command_other is not None:
                        action_other, object_name_other, object_id_other = vh_tools.parse(command_other)
                        same_action = False
                        if action_other != 'walktowards' or action != 'walktowards':
                            if object_id_other > -1 and object_id_other == object_id:
                                same_action = True

                        if not same_action:
                            other_script, _ = vh_tools.can_perform_action(action_other, object_name_other, object_id_other, graph, 2)

                            script[0] = script[0] + '| <char1> {}'.format(command_other)
        
                comm.render_script(script, skip_animation=True, recording=False, image_synthesis=[])
                
            else:
                print(message, "ERROR")
                info.update({'errormsg': message})
    
    s, g = comm.environment_graph()
    graph = g
    rooms = [(node['class_name'], node['id']) for node in graph['nodes'] if node['category'].lower() == 'rooms']
    info.update({'rooms': rooms})
    

    # For now always

    if extra_agent is not None:
        next_helper_action = get_helper_action(graph, goal_spec)
        print("GETTING ACTION")


    object_action, other_info = vh_tools.graph_info(g, id2classid)
    for it, (obj, actions) in enumerate(object_action):
        if object_action[it][0][4] == '1':
            object_action[it] = (object_action[it][0], [action for action in object_action[it][1] if action != 'walktowards'])
    # If the object is close, remove walktowards


    #print(object_action)
    #print(other_info.keys())

    # Information about task name
    other_info['task_name'] = curr_task['task_name']
    # Task preds
    if len(last_completed) == 0:
        for task_pred, count in curr_task['goal_class'].items():
            last_completed[task_pred] = 0
            # total_completed[task_pred] = 0

    # TODO: make this more general
    id2node = {node['id']: node for node in g['nodes']}
    first_pred = list(curr_task['task_goal'][0].keys())[0]
    obj_id_pred = int(first_pred.split('_')[-1])
    visible_ids = [v[0] for v in other_info['visible_objects']]

    num_preds_done = 0
    num_preds_needed = 0
    num_preds_done_all = 0

    # Everything is on or inside...
    total_completed = {}
    ids_curr = None
    if first_pred.split('_')[0] == 'on':
        ids_curr = [id2node[edge['from_id']]['class_name'] for edge in g['edges'] if edge['relation_type'] == 'ON' and edge['to_id'] == obj_id_pred]
    elif first_pred.split('_')[0] == 'inside':
        ids_curr = [id2node[edge['from_id']]['class_name'] for edge in g['edges'] if edge['relation_type'] == 'INSIDE' and edge['to_id'] == obj_id_pred]
    
    grabbed = [id2node[edge['to_id']]['class_name'] for edge in g['edges'] if 'HOLD' in edge['relation_type'] and edge['from_id'] == 1]
    # print([edge for edge in g['edges'] if 'HOLD' in edge['relation_type']])
    if ids_curr is not None:
        for task_pred in curr_task['goal_class'].keys():
            if 'holds' in task_pred:
                print("GRABED", grabbed)

                current_object = task_pred.split('_')[1]
                if len(grabbed) > 0 and grabbed[0] in current_object:
                    last_completed[task_pred] = 1
                    total_completed[task_pred] = 1
                else:
                    total_completed[task_pred] = 0
                    last_completed[task_pred] = 0

            else:
                current_object = task_pred.split('_')[1]
                if obj_id_pred in visible_ids:
                    last_completed[task_pred] = len([obj_name for obj_name in ids_curr if obj_name in current_object])
                total_completed[task_pred] = len([obj_name for obj_name in ids_curr if obj_name in current_object])


    # print(curr_task['task_goal'])
    other_info['task_preds'] = []

    # Generate string
    for task_pred, count in curr_task['goal_class'].items():
        if count > 0 and 'sit' not in task_pred:
            completed = last_completed[task_pred]
            completed_all = total_completed[task_pred]
            task_pred_name = task_pred
            if task_pred == 'holds_book_character':
                task_pred_name = 'hold_book'
            task_pred_name = task_pred_name.replace('_', ' ')
            num_preds_needed += count
            num_preds_done += min(completed, count)
            num_preds_done_all += min(completed_all, count)

            other_info['task_preds'].append('{}: {}/{}'.format(task_pred_name, completed, count))

    if num_preds_done_all == num_preds_needed or record_step_counter >= 250:
        other_info['task_finished'] = '1'
    else:
        other_info['task_finished'] = '0'

    other_info['step_str'] = '{}/250'.format(record_step_counter)
    
    other_info['total_completed_str'] = '{}/{}'.format(num_preds_done, num_preds_needed)

    other_info['task_id_str'] = '{}/{}'.format(task_index+1, len(args.task_group))


    other_info['goal_id'] = obj_id_pred
    info.update({'object_action':object_action, 'other_info': other_info})
    images = refresh_image(g, [obj_id_pred])


    if executed_instruction and record_graph_flag:
        if not os.path.exists(graph_save_dir):
            os.makedirs(graph_save_dir)
        with open(os.path.join(graph_save_dir, 'file_{}.json'.format(record_step_counter)), 'w') as f:
            g_clean = {
                'nodes': [vh_tools.reduced_node_info(node) for node in graph['nodes'] if node['id'] not in bad_class_name_ids], 
                'edges': [edge for edge in graph['edges'] if edge['from_id'] not in bad_class_name_ids and edge['from_id'] not in bad_class_name_ids]
            }
            file_info = {
                'graph': g_clean,
                'instruction': script,
                'time': time.time() - time_start,
                'predicates': total_completed
            }
            json.dump(file_info, f)
        record_step_counter += 1

    info['all_done'] = False
    return images, info

def reset(scene, init_graph=None, init_room=[]):
    global lc
    global instance_colors
    global image_top
    global next_helper_action
    global instance_colors_reverse
    global bad_class_name_ids, record_step_counter, task_index, graph_save_dir, curr_task, task_index_shuffle
    global last_completed, current_goal
    global goal_spec
    global extra_agent
    global graph
    global extra_agent_list
    global id2classid, class2numobj

    class2numobj = {}
    id2classid = {}
    

    record_step_counter = 0
    task_index = task_index + 1
    all_done = False
    if task_index >= len(args.task_group):
        return None, {'all_done': True}
        print('All tasks in task_group {} are finished.'.format(args.task_group))
        task_index = task_index - 1
    temp_task_id = int(args.task_group[task_index_shuffle[task_index]])
    graph_save_dir = 'record_graph/{}/task_{}/time.{}'.format(args.exp_name, temp_task_id, time_str)

    #### For debug  ####
    pkl_file = 'data_input/test_env_set_help_20_neurips.pik'
    with open(pkl_file, 'rb') as f:
        file_content = pkl.load(f)
    curr_task = file_content[temp_task_id]
    scene = curr_task['env_id']
    init_graph = curr_task['init_graph']
    init_room = curr_task['init_rooms']
    #### Debug code finished.

    last_completed = {}

    comm.reset(scene)
    if init_graph is not None:
        s, m = comm.expand_scene(init_graph)
        print("EXPNAD", m)
    

    comm.add_character('Chars/Female1', initial_room=init_room[0])
    extra_agent_name = extra_agent_list[task_index_shuffle[task_index]]
    if extra_agent_name != "none":
        import agents
        comm.add_character('Chars/Male1', initial_room=init_room[1])

    s, g = comm.environment_graph()
    graph = g
    images = refresh_image(current_graph=g)
    # images = [image]
    #image_top = images[0]
    

    if extra_agent_name == "planner":
        extra_agent = agents.MCTS_agent(agent_id=2,
                                       char_index=1,
                                       max_episode_length=5,
                                       num_simulation=100,
                                       max_rollout_steps=5,
                                       c_init=0.1,
                                       c_base=1000000,
                                       num_samples=1,
                                       num_processes=1,
                                       seed=temp_task_id)
        gt_graph = g
        #print([node for node in gt_graph['nodes'] if node['id']  in [1,2]])
        task_goal = None
        observed_graph = vh_tools.get_visible_graph(g, agent_id=2)
        extra_agent.reset(observed_graph, gt_graph, task_goal)
        goal_spec = vh_tools.get_predicted_goal(gt_graph, temp_task_id) 
        # print("GOALS", temp_task_id)
        # print(curr_task['task_goal'][0])
        # print(goal_spec)
    
    if extra_agent_name  == "random_goal":
        print("RANDOM")
        extra_agent = agents.MCTS_agent(agent_id=2,
                                       char_index=1,
                                       max_episode_length=5,
                                       num_simulation=100,
                                       max_rollout_steps=5,
                                       c_init=0.1,
                                       c_base=1000000,
                                       num_samples=1,
                                       num_processes=1,
                                       seed=temp_task_id)
        gt_graph = g
        #print([node for node in gt_graph['nodes'] if node['id']  in [1,2]])
        task_goal = None
        observed_graph = vh_tools.get_visible_graph(g, agent_id=2)
        extra_agent.reset(observed_graph, gt_graph, task_goal)
        goal_spec = vh_tools.get_random_goal(g, temp_task_id)
        print("GOAL", goal_spec, temp_task_id)

    elif extra_agent_name  == 'rl_mcts':
        from util import utils_rl_agent
        trained_model_path = 'ADD HERE the training path'
        model_path = (f'{trained_model_path}/trained_models/env.virtualhome/'
                      'task.full-numproc.5-obstype.mcts-sim.unity/taskset.full/agent.hrl_mcts_alice.False/'
                      'mode.RL-algo.a2c-base.TF-gamma.0.95-cclose.0.0-cgoal.0.0-lr0.0001-bs.32_finetuned/'
                      'stepmcts.50-lep.250-teleport.False-gtgraph-forcepred/2000.pt')
        arg_dict = {
                'name': 'model_args',
                'evaluation': True,
                'max_num_objects': 150,
                'hidden_size': 128,
                'init_epsilon': 0,
                'base_net': 'TF',
                'teleport': False,
                'model_path': model_path,
                'num_steps_mcts': 40
        }
        Args = namedtuple('agent_args', sorted(arg_dict))
        model_args = Args(**arg_dict)
        graph_helper = utils_rl_agent.GraphHelper(max_num_objects=150,
                                                  max_num_edges=10, current_task=None,
                                                  simulator_type='unity')
        extra_agent = agents.HRL_agent(agent_id=2,
                                       char_index=1,
                                       args=model_args,
                                       graph_helper=graph_helper)
        curr_model = torch.load(model_args.model_path)[0]
        extra_agent.actor_critic.load_state_dict(curr_model.state_dict())
        gt_graph = g
        #print([node for node in gt_graph['nodes'] if node['id']  in [1,2]])
        task_goal = None
        observed_graph = vh_tools.get_visible_graph(g, agent_id=2)
        extra_agent.reset(observed_graph, gt_graph, task_goal)
        goal_spec = vh_tools.get_predicted_goal(gt_graph, temp_task_id)

    
    elif extra_agent_name  == 'none':
        extra_agent = None
        goal_spec = None


    if not os.path.exists(graph_save_dir):
        os.makedirs(graph_save_dir)
    with open(os.path.join(graph_save_dir, 'init_graph.json'), 'w') as f:
        file_info = {
            'graph': graph,
            'instruction': "START",
            'extra_agent': str(extra_agent_name),
            'goal_spec': goal_spec,
        }
        json.dump(file_info, f)

    bad_class_name_ids = [node['id'] for node in graph['nodes'] if node['class_name'] in vh_tools.get_classes_ignore()]
    for node in graph['nodes']:
        if node['class_name'] not in class2numobj:
            class2numobj[node['class_name']] = 0
        class2numobj[node['class_name']] += 1
        id2classid[node['id']] = class2numobj[node['class_name']]

    return images, {'image_top': get_top_image(), 'all_done': all_done}


def get_helper_action(gt_graph, goal_spec):
    curr_obs = vh_tools.get_visible_graph(gt_graph, agent_id=2)
    print('({})'.format(extra_agent.agent_type), '--')
    if extra_agent.agent_type == 'MCTS':
        command_other = extra_agent.get_action(curr_obs, goal_spec)[0]
    
    else:
        # The ids iwth which we can do actions
        new_goal_spec = {}
        for pred, ct in goal_spec.items():
            if 'holds' in pred or ct == 0:
                continue
            required = True
            reward = 0
            count = ct
            new_goal_spec[pred] = [count, required, reward]
        action_space = [node['id'] for node in curr_obs['nodes']] 
        action, info = extra_agent.get_action(curr_obs, new_goal_spec, action_space_ids=action_space, full_graph=None)
        command_other = action
        print('------------')
        print("EXTRA AGENT")
        print(action, info['predicate'], goal_spec)
        print('------------')

    if command_other is not None:
        command_other = command_other.replace('[walk]', '[walktowards]')
    return command_other


def get_top_image():
    global image_top
    return image_top

def refresh_image(current_graph, curr_goal_id=[]):
    # global prev_images
    print("GOAL", curr_goal_id)
    # ipdb.set_trace()
    visible_ids = vh_tools.get_objects_visible(None, current_graph, ignore_bad_class=True)
    char_ids = [node['id'] for node in current_graph['nodes'] if node['id'] in visible_ids and node['class_name'] == 'character']
    curr_goal_id = [cgid for cgid in curr_goal_id if cgid in visible_ids]
    fig = plot_graph_2d(current_graph, visible_ids=visible_ids, action_ids=[], char_id=char_ids, goal_ids=curr_goal_id, display_furniture=False)
    plt.axis('off')

    fig.tight_layout(pad=0)
    fig.canvas.draw()

    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if curr_task['env_id'] == 6:
        h = int(0.2 * image_from_plot.shape[0])
        image_from_plot = image_from_plot[h:-h, :]
    # image_top = np.transpose(image_top, (1,0,2))
    # else:
        # h = int(0. * image_from_plot.shape[0]) 
        # image_from_plot = image_from_plot[h:-h, :]
    plt.close()
    return [np.transpose(image_from_plot, (1,0,2))[:, ::-1, :]]


@app.route('/')
def output():
    # serve index template
    return render_template('index_headless.html', name='Joe', show_modal=args.showmodal)


@app.route('/querymaskid', methods = ['POST'])
def get_mask_id():
    global instance_colors_reverse
    data = request.get_json(silent=False)
    obj_id = data['obj_id']

    visible_ids = vh_tools.get_objects_visible(None, graph, ignore_bad_class=True)
    char_ids = [node['id'] for node in graph['nodes'] if node['id'] in visible_ids and node['class_name'] == 'character']

    fig = plot_graph_2d(graph, visible_ids=visible_ids, action_ids=[obj_id], char_id=char_ids, goal_ids=[], display_furniture=False)
    plt.axis('off')

    fig.tight_layout(pad=0)
    fig.canvas.draw()

    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if curr_task['env_id'] == 6:
        h = int(0.2 * image_from_plot.shape[0])
        image_from_plot = image_from_plot[h:-h, :]

    current_images = [np.transpose(image_from_plot, (1,0,2))[:, ::-1, :]]
    # current_images = [image_from_plot]
    img_64s = [convert_image(current_image) for current_image in current_images]
    result = {}
    result.update({'img': str(img_64s[0].decode("utf-8"))})
    result = json.dumps(result)

    
    return result




@app.route('/record_graph_flag', methods = ['POST'])
def set_record_graph_flag():
    global record_graph_flag
    record_graph_flag = not record_graph_flag
    return {'record_graph_flag': record_graph_flag}

@app.route('/receiver', methods = ['POST'])
def worker():
    start_t = time.time() 
    # read json + reply
    data = request.get_json(silent=False)
    t1 = time.time()
    # print('t1 - start_t: ', t1 - start_t)
    current_images, out = send_command(data)
    t2 = time.time()
    # print('t2 - t1: ', t2 - t1)
    #current_images  = [current_images[0]]
    # print(current_images)
    # print("HERE")

    result = {'resp': out}
    if out['all_done']:
        result = json.dumps(result)
        return result
    # print(current_images)
    # result.update({'plot_top': str(current_images)})
    img_64s = [convert_image(current_image[:,:,:]) for current_image in current_images]
    result.update({'img': str(img_64s[0].decode("utf-8"))})


    #print(result['img'][:5])
    #print(result['resp']['image_top'][:5])

    result = json.dumps(result)
    #cv2.imwrite('static/curr_im.png', current_images[0])
    #result = json.dumps({'resp': 'current.png'})
    end_t2 = time.time()
    return result

if __name__ == '__main__':
    # run!
    # global graph_save_dir
    task_index_shuffle = list(range(len(args.task_group)))
    code2agent = {
            'none': 'none',
            'B1': 'planner',
            'B2': 'random_goal',
            'B3': 'rl_mcts',
    }
    extra_agent_list = [code2agent[name] for name in args.extra_agent]
    print(extra_agent_list)
    random.shuffle(task_index_shuffle)

    #random.Random(args.exp_name).shuffle(task_index_shuffle)

    now = datetime.now()
    date_time = now.strftime("%m.%d.%Y-%H.%M.%S")
    time_str = str(date_time)
    all_done = False
    if args.task_group is not None:
        if task_index > len(args.task_group):
            all_done = True
            print('All tasks in task_gropu {} are finished.'.format(args.task_group))
        temp_task_id = args.task_group[task_index_shuffle[task_index]]
    else:
        temp_task_id = args.task_id

    graph_save_dir = 'record_graph/{}/task_{}/time.{}'.format(args.exp_name, temp_task_id, time_str)
    comm = comm_unity.UnityCommunication(file_name=args.execname, port=str(args.portvh), no_graphics=True)



    if args.exp_name == "debug":
        images, _ = reset(0)
        s, graph = comm.environment_graph()
    else:
        pkl_file = 'data_input/test_env_set_help_20_neurips.pik'
        with open(pkl_file, 'rb') as f:
            file_content = pkl.load(f)

        curr_task = file_content[temp_task_id]
        # ipdb.set_trace()

        images, _ = reset(curr_task['env_id'], curr_task['init_graph'], init_room=curr_task['init_rooms'])
        s, graph = comm.environment_graph()

    time_start = time.time()
    if args.deployment == 'local':
        app.run(host='localhost', port=str(args.portflask))
    else:
        app.run(host='0.0.0.0', port=str(args.portflask))

