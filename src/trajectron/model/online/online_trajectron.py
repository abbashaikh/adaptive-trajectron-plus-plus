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

from collections import Counter

import numpy as np
import torch

from trajdata import AgentBatch, AgentType

from trajectron.environment import (
    RingBuffer,
    SceneGraph,
    TemporalSceneGraph,
    derivative_of,
)
from trajectron.model.model_utils import ModeKeys
from trajectron.model.online.online_mgcvae import OnlineMultimodalGenerativeCVAE
# from trajectron.model.mgcvae import MultimodalGenerativeCVAE
from trajectron.model.trajectron import Trajectron


class OnlineTrajectron(Trajectron):
    def __init__(self, model_registrar, hyperparams, device):
        super(OnlineTrajectron, self).__init__(
            model_registrar=model_registrar,
            hyperparams=hyperparams,
            log_writer=False,
            device=device,
        )

        # We don't really care that this is a nn.ModuleDict, since
        # we want to index it by node object anyways.
        del self.node_models_dict
        self.node_models_dict = dict()

        # TODO: might not need the following variables
        self.node_data = dict()
        self.scene_graph = None

        self.rel_states = dict()
        self.removed_nodes = Counter()

    def __repr__(self):
        return f"OnlineTrajectron(# nodes: {len(self.nodes)}, device: {self.device}, hyperparameters: {str(self.hyperparams)}) "

    def _add_node_model(self, node):
        if node in self.nodes:
            raise ValueError("%s was already added to this graph!" % str(node))

        self.nodes.add(node)
        self.node_models_dict[node] = OnlineMultimodalGenerativeCVAE(
            self.env, node, self.model_registrar, self.hyperparams, self.device
        )

    def update_removed_nodes(self):
        for node in list(self.removed_nodes.keys()):
            if self.removed_nodes[node] >= len(self.hyperparams["edge_removal_filter"]):
                del self.node_data[node]
                del self.removed_nodes[node]

    def _remove_node_model(self, node):
        if node not in self.nodes:
            raise ValueError("%s is not in this graph!" % str(node))

        self.nodes.remove(node)
        del self.node_models_dict[node]
    
    def set_environment(self, agent_types, edge_types):
        self.node_models_dict.clear()

        for node_type in agent_types:
            # Only add a Model for NodeTypes we want to predict
            if node_type.name in self.pred_state.keys():
                self.node_models_dict[node_type.name] = OnlineMultimodalGenerativeCVAE(
                    node_type,
                    self.model_registrar,
                    self.hyperparams,
                    self.device,
                    edge_types
                )

    def incremental_forward(
        self,
        obs: AgentBatch,
        maps,
        prediction_horizon=0,
        num_samples=0,
        robot_present_and_future=None,
        z_mode=False,
        gmm_mode=False,
        full_dist=False,
        all_z_sep=False,
        run_models=True,
    ):
        # The way this function works is by appending the new datapoints to the
        # ends of each of the LSTMs in the graph. Then, we recalculate the
        # encoder's output vector h_x and feed that into the decoder to sample new outputs.
        mode = ModeKeys.PREDICT

        # No grad since we're predicting always, as evidenced by the line above.
        with torch.no_grad():
            self.node_models_dict.clear()
            for idx, agent_name in enumerate(obs.agent_name):
                edge_types = []
                agent_type = AgentType(obs.agent_type[idx].item())
                if obs.num_neigh[idx]>0:
                    edge_types = [(agent_type, agent_type)]
                self.node_models_dict[agent_name] = OnlineMultimodalGenerativeCVAE(
                    agent_type,
                    self.model_registrar,
                    self.hyperparams,
                    self.device,
                    edge_types
                )

            # This actually updates the node models with the newly observed data.
            if run_models:

                # We want tensors of shape (1, ph + 1, state_dim) where the first 1 is the batch size.
                if (
                    self.hyperparams["incl_robot_node"]
                    # and self.env.scenes[0].robot is not None
                    # and robot_present_and_future is not None
                ):
                    # TODO: not implemented
                    raise
                    # if len(robot_present_and_future.shape) == 2:
                    #     robot_present_and_future = robot_present_and_future[
                    #         np.newaxis, :
                    #     ]

                    # assert robot_present_and_future.shape[1] == prediction_horizon + 1
                    # robot_present_and_future = torch.tensor(
                    #     robot_present_and_future, dtype=torch.float, device=self.device
                    # )

                for agent_name in self.node_models_dict:
                    self.node_models_dict[agent_name].encoder_forward(obs, agent_name)

                # If num_predicted_timesteps or num_samples == 0 then do not run the decoder at all,
                # just update the encoder LSTMs.
                if prediction_horizon == 0 or num_samples == 0:
                    return

                return self.sample_model(
                    obs,
                    prediction_horizon,
                    num_samples,
                    robot_present_and_future=robot_present_and_future,
                    z_mode=z_mode,
                    gmm_mode=gmm_mode,
                    full_dist=full_dist,
                    all_z_sep=all_z_sep,
                )

    def _run_decoder(
        self,
        obs: AgentBatch,
        agent_name,
        num_predicted_timesteps,
        num_samples,
        robot_present_and_future=None,
        z_mode=False,
        gmm_mode=False,
        full_dist=False,
        all_z_sep=False,
    ):
        print(f"Agent: {agent_name}")
        model = self.node_models_dict[agent_name]
        idx = obs.agent_name.index(agent_name)
        dt = obs.dt[idx].unsqueeze(0)
        prediction_dist, predictions_uns = model.decoder_forward(
            dt,
            num_predicted_timesteps,
            num_samples,
            robot_present_and_future=robot_present_and_future,
            z_mode=z_mode,
            gmm_mode=gmm_mode,
            full_dist=full_dist,
            all_z_sep=all_z_sep,
        )

        predictions_np = predictions_uns.cpu().detach().numpy()

        # Return will be of shape (batch_size, num_samples, num_predicted_timesteps, 2)
        return prediction_dist, np.transpose(predictions_np, (1, 0, 2, 3))

    def sample_model(
        self,
        obs: AgentBatch,
        num_predicted_timesteps,
        num_samples,
        robot_present_and_future=None,
        z_mode=False,
        gmm_mode=False,
        full_dist=False,
        all_z_sep=False,
    ):
        # Just start from the encoder output (minus the
        # robot future) and get num_samples of
        # num_predicted_timesteps-length trajectories.
        if num_predicted_timesteps == 0 or num_samples == 0:
            return

        mode = ModeKeys.PREDICT

        # We want tensors of shape (1, ph + 1, state_dim) where the first 1 is the batch size.
        if (
            self.hyperparams["incl_robot_node"]
            and self.env.scenes[0].robot is not None
            and robot_present_and_future is not None
        ):
            # TODO: not implemented
            raise
            # if len(robot_present_and_future.shape) == 2:
            #     robot_present_and_future = robot_present_and_future[np.newaxis, :]

            # assert robot_present_and_future.shape[1] == num_predicted_timesteps + 1

        # No grad since we're predicting always, as evidenced by the line above.
        with torch.no_grad():
            predictions_dict = dict()
            prediction_dists = dict()
            for agent_name in obs.agent_name:
                # if node.is_robot:
                #     continue

                prediction_dists[agent_name], predictions_dict[agent_name] = self._run_decoder(
                    obs,
                    agent_name,
                    num_predicted_timesteps,
                    num_samples,
                    robot_present_and_future,
                    z_mode,
                    gmm_mode,
                    full_dist,
                    all_z_sep)

        return prediction_dists, predictions_dict

    def forward(
        self,
        init_env,
        init_timestep,
        input_dicts,  # After the initial environment
        num_predicted_timesteps,
        num_samples,
        robot_present_and_future=None,
        z_mode=False,
        gmm_mode=False,
        full_dist=False,
        all_z_sep=False,
    ):
        # This is the standard forward prediction function,
        # if you have some historical data and just want to
        # predict forward some number of timesteps.

        # Setting us back to the initial scene graph we had.
        self.set_environment(init_env, init_timestep)

        # Looping through and applying updates to the model.
        for i in range(len(input_dicts)):
            self.incremental_forward(input_dicts[i])

        return self.sample_model(
            num_predicted_timesteps,
            num_samples,
            robot_present_and_future=robot_present_and_future,
            z_mode=z_mode,
            gmm_mode=gmm_mode,
            full_dist=full_dist,
            all_z_sep=all_z_sep,
        )
