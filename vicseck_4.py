#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from alifebook_lib.visualizers import SwarmVisualizer

# visualizerの初期化 (Appendix参照)
visualizer = SwarmVisualizer()

# シミュレーションパラメタ
N = 256
SPEED = 0.01
# 整列を妨げるノイズ
NOISE = 0.0001
# 力の働く距離
ALIGNMENT_DISTANCE = 0.1
# 力の働く角度
ALIGNMENT_ANGLE = np.pi / 3
# 速度の上限/下限
MIN_VEL = 0.005
MAX_VEL = 0.03
# 境界で働く力（0にすると自由境界）
BOUNDARY_FORCE = 0.001


# 位置と速度
x = np.random.rand(N, 3) * 2 - 1
v = (np.random.rand(N, 3) * 2 - 1 ) * MIN_VEL

# alignmentの力を代入する変数
dv_ali = np.empty((N,3))
# 境界で働く力を代入する変数
dv_boundary = np.empty((N,3))

while visualizer:
    for i in range(N):
        # ここで計算する個体の位置と速度
        x_this = x[i]
        v_this = v[i]
        # それ以外の個体の位置と速度の配列
        x_that = np.delete(x, i, axis=0)
        v_that = np.delete(v, i, axis=0)
        # 個体間の距離と角度
        distance = np.linalg.norm(x_that - x_this, axis=1)
        angle = np.arccos(np.dot(v_this, (x_that-x_this).T) / (np.linalg.norm(v_this) * np.linalg.norm((x_that-x_this), axis=1)))
        # 各力が働く範囲内の個体のリスト
        ali_agents_v = v_that[ (distance < ALIGNMENT_DISTANCE) & (angle < ALIGNMENT_ANGLE) ]
        #　平均角度の計算
        if len(ali_agents_v) > 0:
            # ノイズの計算
            noise = np.random.uniform(-NOISE/2, NOISE/2, 3)
            # 各個体の向いている方向（単位ベクトル）を求める
            normalized_directions = ali_agents_v / np.linalg.norm(ali_agents_v, axis=1, keepdims=True)
            # 向きの平均方向を計算
            mean_direction = np.mean(normalized_directions, axis=0)
            print(mean_direction)
            # 自分の向きとの差（整列力）として代入
            dv_ali[i] = noise + mean_direction - v_this / np.linalg.norm(v_this)
        else:
            dv_ali[i] = 0
        #　バウンダリーフォース
        dist_center = np.linalg.norm(x_this) # 原点からの距離
        dv_boundary[i] = - BOUNDARY_FORCE * x_this * (dist_center - 1) / dist_center if (dist_center > 1) else 0
    # 速度のアップデートと上限/下限のチェック
    for i in range(N):
        direction = v[i] + SPEED * dv_ali[i] + dv_boundary[i]
        direction /= np.linalg.norm(direction)
        v[i] = direction
    # 位置のアップデート
    x += v
    visualizer.update(x, v)
    
