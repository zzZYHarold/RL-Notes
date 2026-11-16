"""RL-3C：用一次 semi-gradient 更新演示线性 Q 值近似。

Q(s, a) = theta · features
td_error = target - Q(s, a)
theta <- theta + alpha * td_error * features
本例直接给定 target=5.0。
"""

import numpy as np


if __name__ == "__main__":
    theta = np.array([2.0, -1.0])
    features = np.array([2.0, 1.0])
    alpha = 0.1
    target = 5.0

    # 当前 Q-value 预测
    q_pred = np.dot(theta, features)
    td_error = target - q_pred

    # 线性近似的梯度是 features，因此每个权重按对应特征更新。
    theta += alpha * td_error * features

    new_q = np.dot(theta, features)

    print("Updated theta:", theta.tolist())
    print("New Q-value:", new_q)


# 原标准库手写版本，保留供公式对照。
# theta = [2.0, -1.0]
# features = [2.0, 1.0]
#
# alpha = 0.1
# target = 5.0
#
# # 当前 Q-value 预测
# q_pred = sum(weight * feature for weight, feature in zip(theta, features))
#
# # TD error
# td_error = target - q_pred
#
# # Semi-ghetaradient update
# for i, feature in enumerate(features):
#     theta[i] += alpha * td_error * feature
# new_q = sum(weight * feature for weight, feature in zip(t, features))
#
# from math import isclose
# assert isclose(q_pred, 3.0)
# assert isclose(td_error, 2.0)
# assert isclose(theta[0], 2.4) and isclose(theta[1], -0.8)
# assert isclose(new_q, 4.0)
#
# print("Updated theta:", theta)
# print("New Q-value:", new_q)
