import os
import time
import json
import random
import pathlib
from collections import defaultdict
from typing import Dict, List, Tuple
# from enum import IntEnum

import numpy as np
from tqdm import trange
import torch
import matplotlib.pyplot as plt

from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajdata.data_structures.scene_metadata import Scene as trajdata_Scene
from trajdata.data_structures.state import StateArray
from trajdata.simulation import SimulationScene, sim_metrics, sim_stats, sim_vis
from trajdata.visualization.vis import plot_agent_batch

import trajectron.evaluation as evaluation
import trajectron.visualization as vis
from trajectron.argument_parser import args
from trajectron.model.online.online_trajectron import OnlineTrajectron
from trajectron.model.model_registrar import ModelRegistrar
from trajectron.environment import Environment, Scene, Node, DoubleHeaderNumpyArray, SceneGraph

if not torch.cuda.is_available() or args.device == 'cpu':
    args.device = torch.device('cpu')
else:
    if torch.cuda.device_count() == 1:
        # If you have CUDA_VISIBLE_DEVICES set, which you should,
        # then this will prevent leftover flag arguments from
        # messing with the device allocation.
        args.device = 'cuda:0'

    args.device = torch.device(args.device)

if args.device is None:
    args.device = 'cpu'

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

# def get_clipped_input_dict(obs: AgentBatch, agent_ids, agent_data, hyperparams):
#     input_dict = dict()
#     existing_nodes = get_clipped_nodes(obs, agent_ids, agent_data, hyperparams)
#     for idx, node in enumerate(existing_nodes):
#         input_dict[node] = agent_data[idx]
    
#     return input_dict

def get_clipped_nodes(obs: AgentBatch, agent_ids, agent_data, hyperparams):
    state = hyperparams["state"]

    clipped_nodes = list()
    for idx, node_type in enumerate(obs.agent_type):
        data_header = list()
        for quantity, values in state[AgentType(node_type.item()).name].items():
            for value in values:
                data_header.append((quantity, value))

        data = DoubleHeaderNumpyArray(agent_data[idx], data_header)

        clipped_nodes.append(Node(node_type=AgentType(node_type.item()), node_id=agent_ids[idx], data=data))
    
    return clipped_nodes

def create_online_env(env : UnifiedDataset, test_scene : trajdata_Scene, obs : AgentBatch, agent_ids, hyperparams, init_timestep):
    # test_scene = env.scenes[scene_idx]

    online_scene = Scene(timesteps=init_timestep + 1,
                         map=None,
                         dt=test_scene.dt)
    # online_scene.nodes = test_scene.agents
    agent_hist_data = obs.agent_hist
    online_scene.nodes = get_clipped_nodes(obs, agent_ids, agent_hist_data, hyperparams)
    # online_scene.robot = test_scene.robot
    online_scene.calculate_scene_graph(attention_radius=env.agent_interaction_distances,
                                       edge_addition_filter=hyperparams['edge_addition_filter'],
                                       edge_removal_filter=hyperparams['edge_removal_filter'])

    env_standardization = {'PEDESTRIAN': {'position': {'x': {'mean': 0, 'std': 1}, 'y': {'mean': 0, 'std': 1}}, 'velocity': {'x': {'mean': 0, 'std': 2}, 'y': {'mean': 0, 'std': 2}}, 'acceleration': {'x': {'mean': 0, 'std': 1}, 'y': {'mean': 0, 'std': 1}}}}
    node_type_list = [AgentType(node_type.item()).name for node_type in obs.agent_type]
    return Environment(node_type_list=node_type_list,
                       standardization=env_standardization,
                       scenes=[online_scene],
                       attention_radius=env.agent_interaction_distances
        )

def get_new_dict(obs):
    new_dict: Dict[str, StateArray] = dict()
    for idx, agent_name in enumerate(obs.agent_name):
        curr_yaw = obs.curr_agent_state[idx].heading.item()
        curr_pos = obs.curr_agent_state[idx].position.numpy()
        world_from_agent = np.array(
            [
                [np.cos(curr_yaw), np.sin(curr_yaw)],
                [-np.sin(curr_yaw), np.cos(curr_yaw)],
            ]
        )
        next_state = np.zeros((4,))
        if obs.agent_fut_len[idx] < 1:
            next_state[:2] = curr_pos
            yaw_ac = 0
        else:
            next_state[:2] = (
                obs.agent_fut[idx, 0].position.numpy() @ world_from_agent
                + curr_pos
            )
            yaw_ac = obs.agent_fut[idx, 0].heading.item()

        next_state[-1] = curr_yaw + yaw_ac
        new_dict[agent_name] = StateArray.from_array(next_state, "x,y,z,h")

    return new_dict

def main():
    log_dir = '/home/abbas/Projects/trajectron/adaptive-trajectron-plus-plus/experiments/pedestrians/kf_models'
    model_dir = os.path.join(log_dir, 'eth_1mode_base_tpp-10_Nov_2024_21_41_22')

    # Load hyperparameters from json
    conf = 'config.json'
    config_file = os.path.join(model_dir, conf)
    if not os.path.exists(config_file):
        raise ValueError('Config json not found!')
    with open(config_file, 'r') as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams['dynamic_edges'] = args.dynamic_edges
    hyperparams['edge_state_combine_method'] = args.edge_state_combine_method
    hyperparams['edge_influence_combine_method'] = args.edge_influence_combine_method
    hyperparams['edge_addition_filter'] = args.edge_addition_filter
    hyperparams['edge_removal_filter'] = args.edge_removal_filter
    hyperparams['batch_size'] = args.batch_size
    hyperparams['k_eval'] = args.k_eval
    hyperparams['incl_robot_node'] = args.incl_robot_node
    hyperparams['edge_encoding'] = not args.no_edge_encoding
    hyperparams['use_map_encoding'] = args.map_encoding

    output_save_dir = os.path.join(model_dir, 'pred_figs')
    pathlib.Path(output_save_dir).mkdir(parents=True, exist_ok=True)

    # dataset = UnifiedDataset(
    #     desired_data=["eupeds_eth-val"],
    #     only_types=[AgentType.PEDESTRIAN],
    #     agent_interaction_distances=defaultdict(lambda: 10.0),
    #     verbose=True,
    #     num_workers=4,
    #     data_dirs={  # Remember to change this to match your filesystem!
    #         "eupeds_eth": "~/Projects/trajectron/datasets/eth_ucy_peds",
    #     },
    # )

    # Load evaluation environments and scenes
    attention_radius = defaultdict(
        lambda: 20.0
    )  # Default range is 20m unless otherwise specified.
    attention_radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] = 10.0
    attention_radius[(AgentType.PEDESTRIAN, AgentType.VEHICLE)] = 20.0
    attention_radius[(AgentType.VEHICLE, AgentType.PEDESTRIAN)] = 20.0
    attention_radius[(AgentType.VEHICLE, AgentType.VEHICLE)] = 30.0

    dataset = UnifiedDataset(
        desired_data=["eupeds_eth-val"],
        history_sec=(hyperparams["history_sec"], hyperparams["history_sec"]),
        future_sec=(hyperparams["prediction_sec"], hyperparams["prediction_sec"]),
        agent_interaction_distances=attention_radius,
        incl_robot_future=hyperparams["incl_robot_node"],
        incl_raster_map=hyperparams["map_encoding"],
        only_predict=[AgentType.PEDESTRIAN],
        no_types=[AgentType.UNKNOWN],
        num_workers=hyperparams["preprocess_workers"],
        cache_location=hyperparams["trajdata_cache_dir"],
        data_dirs={  # Remember to change this to match your filesystem!
            "eupeds_eth": "~/Projects/trajectron/datasets/eth_ucy_peds",
        },
        verbose=True,
    )

    # Creating a dummy environment with a single scene that contains information about the world.
    # When using this code, feel free to use whichever scene index or initial timestep you wish.
    scene_idx = 0

    # You need to have at least acceleration, so you want 2 timesteps of prior data, e.g. [0, 1],
    # so that you can immediately start incremental inference from the 3rd timestep onwards.
    init_timestep = 1

    sim_env_name = "eth_sim"
    # all_sim_scenes: List[Scene] = list()
    desired_scene: Scene = dataset.get_scene(scene_idx)
    sim_scene: SimulationScene = SimulationScene(
        env_name=sim_env_name,
        scene_name="sim_scene-0",
        scene=desired_scene,
        dataset=dataset,
        init_timestep=0,
        freeze_agents=True,
    )

    # print(type(sim_scene.scene))
    # print(sim_scene.scene.agents)
    # print(sim_scene.scene.agents)
    agent_ids = [item.name for item in sim_scene.scene.agents]
    # print(agent_ids)

    obs: AgentBatch = sim_scene.reset()
    # print('--------------------------------')
    # print(obs.scene_ts)
    # print('--------------------------------')
    # print(obs.num_neigh)
    # print('--------------------------------')
    # print(obs.scene_ids)
    # print('--------------------------------')
    # print(obs.robot_fut_len)
    # print('--------------------------------')
    # # print(obs.agent_types)
    # print('--------------------------------')
    # print(obs.agent_type)
    # print('--------------------------------')
    # print(obs.agent_hist_len)
    # print('--------------------------------')
    # print(len(obs.curr_agent_state))
    # print('--------------------------------')
    # print(obs.agent_fut_len)
    # print('--------------------------------')
    # print(obs.agent_hist)

    # print(hyperparams["state"]["PEDESTRIAN"].items())

    # clipped_nodes = get_clipped_nodes(obs, agent_ids, hyperparams)
    # print(clipped_nodes)
    
    hyperparams["maximum_history_length"] = (hyperparams["history_sec"]/sim_scene.scene.dt) + 1

    model_registrar = ModelRegistrar(model_dir, args.device)

    trajectron = OnlineTrajectron(model_registrar,
                                  hyperparams,
                                  args.device)
    
    epoch = 50
    model_path = pathlib.Path(model_dir) / f'model_registrar-{epoch}.pt'
    checkpoint = torch.load(model_path, map_location=args.device)
    trajectron.load_state_dict(checkpoint["model_state_dict"], strict=False)

    online_env = create_online_env(dataset, sim_scene.scene, obs, agent_ids, hyperparams, init_timestep)
    trajectron.env = online_env
    trajectron.scene_graph = SceneGraph(edge_radius=online_env.attention_radius)
    trajectron.nodes.clear()
    trajectron.node_data.clear()
    trajectron.node_models_dict.clear()

    # print('-----------------------------------------')
    # print(type(obs.curr_agent_state))
    # print(type(obs.__dict__))
    
    for t in trange(1, sim_scene.scene.length_timesteps):
        new_xyzh_dict: Dict[str, StateArray] = dict()
        if t<=init_timestep:
            trajectron.incremental_forward(
                obs.__dict__,
                maps=None,
                run_models=False
            )
        else:
            start = time.time()
            dists, preds = trajectron.incremental_forward(
                obs.__dict__,
                prediction_horizon=6,
                num_samples=1,
                full_dist=True
            )
            end = time.time()
            print("t=%d: took %.2f s (= %.2f Hz) w/ %d nodes and %d edges" % (t, end - start,
                                                                          1. / (end - start), len(trajectron.nodes),
                                                                          trajectron.scene_graph.get_num_edges()))
            
            detailed_preds_dict = dict()
            for name in obs.agent_name:
                if name in preds:
                    detailed_preds_dict[name] = preds[name]

            fig, ax = plt.subplots()
            vis.visualize_distribution(ax,
                                    dists)
            vis.visualize_prediction(ax,
                                    {t: preds},
                                    sim_scene.scene.dt,
                                    hyperparams['maximum_history_length'],
                                    hyperparams['prediction_horizon'])
            
            fig.savefig(os.path.join(output_save_dir, f'pred_{t}.pdf'), dpi=300)
            plt.close(fig)

        new_xyzh_dict = get_new_dict(obs)
        obs = sim_scene.step(new_xyzh_dict)


if __name__ == "__main__":
    main()