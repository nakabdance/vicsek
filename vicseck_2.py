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
SPEED = 0.01
NOISE = 0
INTERACTION_RADIUS = 0.1
INTERACTION_ANGLE = 0.5
BOUNDARY_FORCE = 0.001  # 原点中心への弱いバウンダリーフォース

# 初期位置と速度
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

        # 近傍探索
        diff = x - xi
        distances = np.linalg.norm(diff, axis=1)
        angle = np.arccos(np.dot(vi, (diff).T) / (np.linalg.norm(vi) * np.linalg.norm((diff), axis=1)))
        neighbors = ((distances < INTERACTION_RADIUS) & (angle < INTERACTION_ANGLE))

        
        # 平均方向＋ノイズ
        if np.sum(neighbors) > 1:
            mean_v = np.mean(v[neighbors], axis=0)
            mean_dir = mean_v / np.linalg.norm(mean_v)

            # ノイズ
            noise = np.random.normal(0, NOISE, 3)
            noise /= np.linalg.norm(noise)

            direction = mean_dir + noise
        else:
            direction = vi

        # バウンダリーフォースを加える（外に出たら内側に引く）
        dist_from_center = np.linalg.norm(xi)
        if dist_from_center > 1.0:
            boundary_pull = -BOUNDARY_FORCE * xi * (dist_from_center - 1.0) / dist_from_center
            direction += boundary_pull

        # 正規化して一定速度に
        direction /= np.linalg.norm(direction)
        new_v[i] = SPEED * direction
        v = new_v
        x += v

    # トーラス境界ではなく、強制的に原点中心に戻す方式（そのためラップなし）
    visualizer.update(x, v)
