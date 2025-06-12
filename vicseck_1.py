#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
import numpy as np 
from alifebook_lib.visualizers import SwarmVisualizer

# Visualizerの初期化
visualizer = SwarmVisualizer()

# パラメータ設定
N = 256
SPEED = 0.02
NOISE = 0               # ノイズの強さ（ラジアン）
INTERACTION_RADIUS = 0.1  # 相互作用半径

# 初期位置と初期速度（ランダムな方向）
x = np.random.rand(N, 3) * 2 - 1
theta = np.random.rand(N) * 2 * np.pi
phi = np.random.rand(N) * np.pi
v = np.stack([
    SPEED * np.sin(phi) * np.cos(theta),
    SPEED * np.sin(phi) * np.sin(theta),
    SPEED * np.cos(phi)
], axis=1)

# メインループ
while visualizer:
    new_v = np.zeros_like(v)
    for i in range(N):
        xi = x[i]
        vi = v[i]

        # 自分以外の個体との相対距離
        diff = x - xi
        distances = np.linalg.norm(diff, axis=1)
        neighbors = (distances < INTERACTION_RADIUS)

        if np.sum(neighbors) > 1:
            # 自分を除いた近傍の速度の平均方向を取得
            mean_v = np.mean(v[neighbors], axis=0)
            norm_mean_v = mean_v / np.linalg.norm(mean_v)

            # ノイズを加える
            noise_vector = np.random.normal(0, NOISE, 3)
            norm_noise = noise_vector / np.linalg.norm(noise_vector)

            # 平均方向 + ノイズベクトルを正規化して新たな方向とする
            new_direction = norm_mean_v + norm_noise
            new_direction /= np.linalg.norm(new_direction)

            new_v[i] = SPEED * new_direction
        else:
            # 近傍がいなければそのまま
            new_v[i] = vi

    # 更新
    v = new_v
    x += v
    visualizer.update(x, v)
