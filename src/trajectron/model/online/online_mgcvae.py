# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trajdata import AgentBatch

import trajectron.model.dynamics as dynamic_module
from trajectron.environment.node_type import NodeType
from trajectron.environment.scene_graph import DirectedEdge
from trajectron.model.components import *
from trajectron.model.dataset import get_relative_robot_traj
from trajectron.model.mgcvae import MultimodalGenerativeCVAE
from trajectron.model.model_utils import *


class OnlineMultimodalGenerativeCVAE(MultimodalGenerativeCVAE):
    def __init__(self, node_type, model_registrar, hyperparams, device, edge_types):

        super(OnlineMultimodalGenerativeCVAE, self).__init__(
            node_type, model_registrar, hyperparams, device, edge_types
        )
        self.n_s_t0: torch.Tensor = torch.empty(size=(1, self.state_length))
        self.x : torch.Tensor
        
        # self.env = env
        # self.node = node
        # self.robot = env.scenes[0].robot

        # self.scene_graph = None

        # self.curr_hidden_states = dict()
        # self.edge_types = Counter()

        # self.create_initial_graphical_model()

    def create_initial_graphical_model(self):
        """
        Creates or queries all trainable components.

        :return: None
        """
        self.clear_submodules()

        ############################
        #   Everything but Edges   #
        ############################
        self.create_node_models()

        for name, module in self.node_modules.items():
            module.to(self.device)

    def update_graph(self, new_scene_graph, new_neighbors, removed_neighbors):
        self.scene_graph = new_scene_graph

        if self.node in new_neighbors:
            for edge_type, new_neighbor_nodes in new_neighbors[self.node].items():
                self.edge_types += Counter({edge_type: len(new_neighbor_nodes)})
                self.add_edge_model(edge_type)

        if self.node in removed_neighbors:
            for edge_type, removed_neighbor_nodes in removed_neighbors[self.node].items():
                self.edge_types -= Counter({edge_type: len(removed_neighbor_nodes)})
                if self.edge_types[edge_type]==0:
                    self.remove_edge_model(edge_type)
                

    def get_edge_to(self, other_node):
        return DirectedEdge(self.node, other_node)

    def add_edge_model(self, edge_type):
        if self.hyperparams["edge_encoding"]:
            if edge_type + "/edge_encoder" not in self.node_modules:
                neighbor_state_length = int(
                    np.sum(
                        [
                            len(entity_dims)
                            for entity_dims in self.state[
                                self._get_other_node_type_from_edge(edge_type)
                            ].values()
                        ]
                    )
                )
                if self.hyperparams["edge_state_combine_method"] == "pointnet":
                    self.add_submodule(
                        edge_type + "/pointnet_encoder",
                        model_if_absent=nn.Sequential(
                            nn.Linear(self.state_length, 2 * self.state_length),
                            nn.ReLU(),
                            nn.Linear(2 * self.state_length, 2 * self.state_length),
                            nn.ReLU(),
                        ),
                    )

                    edge_encoder_input_size = 2 * self.state_length + self.state_length

                elif self.hyperparams["edge_state_combine_method"] == "attention":
                    self.add_submodule(
                        self.node.type.name + "/edge_attention_combine",
                        model_if_absent=TemporallyBatchedAdditiveAttention(
                            encoder_hidden_state_dim=self.state_length,
                            decoder_hidden_state_dim=self.state_length,
                        ),
                    )
                    edge_encoder_input_size = self.state_length + neighbor_state_length

                else:
                    edge_encoder_input_size = self.state_length + neighbor_state_length

                self.add_submodule(
                    edge_type + "/edge_encoder",
                    model_if_absent=nn.LSTM(
                        input_size=edge_encoder_input_size,
                        hidden_size=self.hyperparams["enc_rnn_dim_edge"],
                        batch_first=True,
                    ),
                )

    def _get_other_node_type_from_edge(self, edge_type_str):
        n2_type_str = edge_type_str.split("->")[1]
        return NodeType(n2_type_str, self.env.node_type_list.index(n2_type_str) + 1)

    def _get_edge_type_from_str(self, edge_type_str):
        n1_type_str, n2_type_str = edge_type_str.split("->")
        return (
            NodeType(n1_type_str, self.env.node_type_list.index(n1_type_str) + 1),
            NodeType(n2_type_str, self.env.node_type_list.index(n2_type_str) + 1),
        )

    def remove_edge_model(self, edge_type):
        if self.hyperparams["edge_encoding"]:
            if (
                len(
                    self.scene_graph.get_neighbors(
                        self.node, self._get_other_node_type_from_edge(edge_type)
                    )
                )
                == 0
            ):
                del self.node_modules[edge_type + "/edge_encoder"]

    def create_encoder_rep(self, mode, TD, robot_present_st, robot_future_st):
        # Unpacking TD
        node_history_encoded = TD["node_history_encoded"]
        if self.hyperparams["edge_encoding"]:
            total_edge_influence = TD["total_edge_influence"]
        if (
            self.hyperparams["map_encoding"]
            and self.node_type in self.hyperparams["map_encoder"]
        ):
            encoded_map = TD["encoded_map"]

        if (
            self.hyperparams["incl_robot_node"]
            and self.robot is not None
            and robot_future_st is not None
            and robot_present_st is not None
        ):
            robot_future_encoder = self.encode_robot_future(
                mode, robot_present_st, robot_future_st
            )

            # Tiling for multiple samples
            # This tiling is done because:
            #   a) we must consider the prediction case where there are many candidate robot future actions,
            #   b) the edge and history encoders are all the same regardless of which candidate future robot action
            #      we're evaluating.
            node_history_encoded = TD["node_history_encoded"].repeat(
                robot_future_st.size()[0], 1
            )
            if self.hyperparams["edge_encoding"]:
                total_edge_influence = TD["total_edge_influence"].repeat(
                    robot_future_st.size()[0], 1
                )
            if (
                self.hyperparams["map_encoding"]
                and self.node_type in self.hyperparams["map_encoder"]
            ):
                encoded_map = TD["encoded_map"].repeat(robot_future_st.size()[0], 1)

        elif self.hyperparams["incl_robot_node"] and self.robot is not None:
            # Four times because we're trying to mimic a bi-directional RNN's output (which is c and h from both ends).
            robot_future_encoder = torch.zeros(
                [1, 4 * self.hyperparams["enc_rnn_dim_future"]], device=self.device
            )

        x_concat_list = list()

        # Every node has an edge-influence encoder (which could just be zero).
        if self.hyperparams["edge_encoding"]:
            x_concat_list.append(total_edge_influence)  # [bs/nbs, 4*enc_rnn_dim]

        # Every node has a history encoder.
        x_concat_list.append(node_history_encoded)  # [bs/nbs, enc_rnn_dim_history]

        if self.hyperparams["incl_robot_node"] and self.robot is not None:
            x_concat_list.append(
                robot_future_encoder
            )  # [bs/nbs, 4*enc_rnn_dim_history]

        if (
            self.hyperparams["map_encoding"]
            and self.node_type in self.hyperparams["map_encoder"]
        ):
            x_concat_list.append(encoded_map)  # [bs/nbs, CNN output size]

        return torch.cat(x_concat_list, dim=1)
    
    def obtain_encoded_tensors(
        self, mode: ModeKeys, obs: AgentBatch, idx
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Encodes input and output tensors for node and robot.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :return: tuple(x, x_nr_t, y_e, y_r, y, n_s_t0)
            WHERE
            - x: Encoded input / condition tensor to the CVAE x_e.
            - x_r_t: Robot state (if robot is in scene).
            - y_e: Encoded label / future of the node.
            - y_r: Encoded future of the robot.
            - y: Label / future of the node.
            - n_s_t0: Standardized current state of the node.
        """

        enc, x_r_t, y_e, y_r, y = None, None, None, None, None
        initial_dynamics = dict()

        batch_size = 1

        #########################################
        # Provide basic information to encoders #
        #########################################
        node_history_st_len = obs.agent_hist_len[idx].unsqueeze(0)
        node_history_st = obs.agent_hist[idx][-node_history_st_len:].unsqueeze(0)
        
        node_present_state_st = node_history_st[0, node_history_st_len - 1, :]

        initial_dynamics["pos"] = node_present_state_st[:, 0:2]
        initial_dynamics["vel"] = node_present_state_st[:, 2:4]

        self.dynamic.set_initial_condition(initial_dynamics)

        if self.hyperparams["incl_robot_node"]:
            robot = obs.robot_fut
            robot_lens = obs.robot_fut_len
            x_r_t, y_r = robot[:, 0], robot[:, 1:]

        ##################
        # Encode History #
        ##################
        node_history_encoded = self.encode_node_history(
            mode, node_history_st, node_history_st_len
        )

        ##################
        # Encode Present #
        ##################
        node_present = node_present_state_st  # [bs, state_dim]

        ##################
        # Encode Future #
        ##################
        # if mode != ModeKeys.PREDICT:
        #     y = batch.agent_fut[..., :2]
        #     y_lens = batch.agent_fut_len

        ##############################
        # Encode Node Edges per Type #
        ##############################
        if self.hyperparams["edge_encoding"]:
            if obs.num_neigh.max() == 0:
                total_edge_influence = torch.zeros_like(node_history_encoded)
            else:
                # Encode edges
                num_neigh = obs.num_neigh[idx].unsqueeze(0)
                neigh_hist_len = obs.neigh_hist_len[idx, :num_neigh].unsqueeze(0)
                neigh_types = obs.neigh_types[idx, :num_neigh].unsqueeze(0)
                max_hist_len = torch.max(neigh_hist_len)
                neigh_hist = torch.full((1, num_neigh, max_hist_len, self.state_length), float('nan'))
                for neigh_idx, hist_len in enumerate(neigh_hist_len):
                    neigh_hist[0, neigh_idx, :hist_len, :] = obs.neigh_hist[idx, neigh_idx, -hist_len:, :]
                
                encoded_edges = self.encode_edge(
                    mode,
                    node_history_st,
                    node_history_st_len,
                    neigh_hist,
                    neigh_hist_len,
                    neigh_types,
                    num_neigh,
                )
                #####################
                # Encode Node Edges #
                #####################
                total_edge_influence, attn_weights = self.encode_total_edge_influence(
                    mode,
                    encoded_edges,
                    num_neigh,
                    node_history_encoded,
                    node_history_st_len,
                    batch_size,
                )

        ################
        # Map Encoding #
        ################
        if (
            self.hyperparams["map_encoding"]
            and self.node_type in self.hyperparams["map_encoder"]
        ):
            if (
                self.hyperparams["log_maps"]
                and self.log_writer
                and (self.curr_iter + 1) % 500 == 0
            ):
                # TODO: not implemented
                raise
                # image = wandb.Image(batch.maps[0], caption=f"Batch Map 0")
                # self.log_writer.log(
                #     {f"{self.node_type}/maps": image}, step=self.curr_iter, commit=False
                # )

            # encoded_map = self.node_modules[self.node_type + "/map_encoder"](
            #     batch.maps * 2.0 - 1.0, (mode == ModeKeys.TRAIN)
            # )
            # do = self.hyperparams["map_encoder"][self.node_type]["dropout"]
            # encoded_map = F.dropout(encoded_map, do, training=(mode == ModeKeys.TRAIN))

        ######################################
        # Concatenate Encoder Outputs into x #
        ######################################
        enc_concat_list = list()

        # Every node has an edge-influence encoder (which could just be zero).
        if self.hyperparams["edge_encoding"]:
            enc_concat_list.append(total_edge_influence)  # [bs/nbs, enc_rnn_dim]

        # Every node has a history encoder.
        enc_concat_list.append(node_history_encoded)  # [bs/nbs, enc_rnn_dim_history]

        if self.hyperparams["incl_robot_node"]:
            robot_future_encoder = self.encode_robot_future(
                mode, x_r_t, y_r, robot_lens
            )
            enc_concat_list.append(robot_future_encoder)

        if (
            self.hyperparams["map_encoding"]
            and self.node_type in self.hyperparams["map_encoder"]
        ):
            # TODO: not implemented
            raise
            # if self.log_writer:
            #     self.log_writer.log(
            #         {
            #             f"{self.node_type}/encoded_map_max": torch.max(
            #                 torch.abs(encoded_map)
            #             ).item()
            #         },
            #         step=self.curr_iter,
            #         commit=False,
            #     )
            # enc_concat_list.append(
            #     encoded_map.unsqueeze(1).expand((-1, node_history_encoded.shape[1], -1))
            #     if self.hyperparams["adaptive"]
            #     else encoded_map
            # )

        enc = torch.cat(enc_concat_list, dim=-1)

        # if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
        #     y_e = self.encode_node_future(mode, node_present, y, y_lens)
        #     if self.hyperparams["adaptive"]:
        #         y_e = y_e.expand((-1, enc.shape[1], -1))

        return enc, x_r_t, y_e, y_r, y

    def encoder_forward(self, obs: AgentBatch, agent_name):
        # Always predicting with the online model.
        mode = ModeKeys.PREDICT

        # index of the agent being tracked (among all agents in scene at current timestep)
        idx = obs.agent_name.index(agent_name)

        self.x, x_nr_t, _, y_r, _ = self.obtain_encoded_tensors(mode, obs, idx)

        # This is the old n_s_t0 (just the state at the current timestep, t=0).
        self.n_s_t0: torch.Tensor = obs.agent_hist[idx, -1, :]
        nan_mask = torch.isnan(self.n_s_t0)
        self.n_s_t0[nan_mask] = 0

        self.latent.p_dist = self.p_z_x(mode, self.x)

    # robot_future_st is optional here since you can use the same one from encoder_forward,
    # but if it's given then we'll re-run that part of the model (if the node is adjacent to the robot).
    def decoder_forward(
        self,
        dt,
        prediction_horizon,
        num_samples,
        robot_present_and_future=None,
        z_mode=False,
        gmm_mode=False,
        full_dist=False,
        all_z_sep=False,
    ):
        # Always predicting with the online model.
        mode = ModeKeys.PREDICT

        x_nr_t, y_r = None, None
        if (
            self.hyperparams["incl_robot_node"]
            and self.robot is not None
            and robot_present_and_future is not None
        ):
            # TODO: not implemented
            raise
            # our_inputs = torch.tensor(
            #     self.node.get(
            #         np.array([self.node.last_timestep]),
            #         self.state[self.node.type.name],
            #         padding=0.0,
            #     ),
            #     dtype=torch.float,
            #     device=self.device,
            # )

            # node_state = torch.zeros(
            #     (robot_present_and_future.shape[-1],),
            #     dtype=torch.float,
            #     device=self.device,
            # )
            # node_state[: our_inputs.shape[1]] = our_inputs[0]

            # robot_present_and_future_st = get_relative_robot_traj(
            #     self.env,
            #     self.state,
            #     node_state,
            #     robot_present_and_future,
            #     self.node.type,
            #     self.robot.type,
            # )
            # x_nr_t = robot_present_and_future_st[..., 0, :]
            # y_r = robot_present_and_future_st[..., 1:, :]
            # self.x = self.create_encoder_rep(mode, self.TD, x_nr_t, y_r)
            # self.latent.p_dist = self.p_z_x(mode, self.x)

            # # Making sure n_s_t0 has the same batch size as x_nr_t
            # self.n_s_t0 = self.n_s_t0[[0]].repeat(x_nr_t.size()[0], 1)

        z, num_samples, num_components = self.latent.sample_p(
            num_samples,
            mode,
            most_likely_z=z_mode,
            full_dist=full_dist,
            all_z_sep=all_z_sep,
        )

        y_dist, our_sampled_future = self.p_y_xz(
            mode,
            x = self.x,
            x_nr_t=x_nr_t,
            y_r=y_r,
            n_s_t0 = self.n_s_t0,
            pos_hist_len=None,
            z_stacked=z,
            dt=dt,
            prediction_horizon=prediction_horizon,
            num_samples=num_samples,
            num_components=num_components,
            gmm_mode=gmm_mode,
        )

        return y_dist, our_sampled_future
