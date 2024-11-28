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

from typing import Dict

import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import linalg

from trajectron.model.components.gmm2d import GMM2D
from trajectron.utils import convert_to_world_tf


def plot_trajectories(
    ax,
    agent_type,
    prediction_dict,
    histories_dict,
    futures_dict,
    line_alpha=0.7,
    line_width=0.2,
    edge_width=2,
    circle_edge_width=0.5,
    node_circle_size=0.3,
    batch_num=0,
    kde=False,
):

    cmap = ["k", "b", "y", "g", "r"]

    for idx, agent_name in enumerate(list(histories_dict.keys())):
        history = histories_dict[agent_name]
        future = futures_dict[agent_name]
        predictions = prediction_dict[agent_name]

        if np.isnan(history[-1]).any():
            continue

        ax.plot(history[:, 0], history[:, 1], "k--")

        for sample_num in range(prediction_dict[agent_name].shape[1]):

            if kde and predictions.shape[1] >= 50:
                line_alpha = 0.2
                for t in range(predictions.shape[2]):
                    sns.kdeplot(
                        predictions[batch_num, :, t, 0],
                        predictions[batch_num, :, t, 1],
                        ax=ax,
                        shade=True,
                        shade_lowest=False,
                        color=np.random.choice(cmap),
                        alpha=0.8,
                    )

            ax.plot(
                predictions[batch_num, sample_num, :, 0],
                predictions[batch_num, sample_num, :, 1],
                color=cmap[agent_type[idx]],
                linewidth=line_width,
                alpha=line_alpha,
            )

            ax.plot(
                future[:, 0],
                future[:, 1],
                "w--",
                path_effects=[
                    pe.Stroke(linewidth=edge_width, foreground="k"),
                    pe.Normal(),
                ],
            )

            # Current Node Position
            circle = plt.Circle(
                (history[-1, 0], history[-1, 1]),
                node_circle_size,
                facecolor="g",
                edgecolor="k",
                lw=circle_edge_width,
                zorder=3,
            )
            ax.add_artist(circle)

    ax.axis("equal")


def visualize_prediction(
    ax, obs, prediction_output_dict, robot_node=None, map=None, **kwargs
):
    # TODO: check if there are agents to predict
    prediction_dict = dict()
    histories_dict = dict()
    futures_dict = dict()

    for idx, agent_name in enumerate(prediction_output_dict):
        history, future, prediction = convert_to_world_tf(prediction_output_dict, obs, idx)

        histories_dict[agent_name] = history
        futures_dict[agent_name] = future
        prediction_dict[agent_name] = prediction

    agent_type = obs.agent_type.cpu().numpy()

    if map is not None:
        ax.imshow(map.as_image(), origin="lower", alpha=0.5)

    plot_trajectories(ax, agent_type, prediction_dict, histories_dict, futures_dict, *kwargs)


def visualize_distribution(
    ax: plt.Axes,
    prediction_distribution_dict: Dict[str, GMM2D],
    map=None,
    pi_threshold=0.05,
    **kwargs
):
    if map is not None:
        ax.imshow(map.as_image(), origin="lower", alpha=0.5)

    for agent_name in prediction_distribution_dict.keys():
        pred_dist = prediction_distribution_dict[agent_name]

        if pred_dist.mus.shape[:2] != (1, 1):
            return

        means = pred_dist.mus[0, 0].cpu().numpy()
        covs = pred_dist.get_covariance_matrix()[0, 0].cpu().numpy()
        pis = pred_dist.pis_cat_dist.probs[0, 0].cpu().numpy()

        ml_idx = np.argmax(pis[0]).item()

        for z_val in range(means.shape[1]):
            pi = pis[0, z_val]  # All pis are the same across time.

            if pi < pi_threshold:
                continue

            if z_val == ml_idx:
                color = "red"
            elif "AgentType.VEHICLE" in agent_name:
                color = "blue"
            else:
                color = "orange"

            alpha_val = pi / 10
            ax.plot(means[:, z_val, 0], means[:, z_val, 1], color=color, alpha=alpha_val)

            for timestep in range(means.shape[0]):
                mean = means[timestep, z_val]
                covar = covs[timestep, z_val]

                v, w = linalg.eigh(covar)
                v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
                u = w[0] / linalg.norm(w[0])

                # Plot an ellipse to show the Gaussian component
                angle = np.arctan2(u[1], u[0])
                angle = 180.0 * angle / np.pi  # convert to degrees
                ell = patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
                ell.set_edgecolor(None)
                ell.set_clip_box(ax.bbox)
                ell.set_alpha(alpha_val)
                ax.add_artist(ell)
