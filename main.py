# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 假设y=2x+0
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(x):
    return x * w + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


w_list = []
b_list = []
mse_list = []
for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(-2, 2, 0.1):
        print('w=', w, ',b=', b)
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            print('\t', x_val, y_val, y_pred_val, loss_val)
        print('MSE=', l_sum / 3)
        w_list.append(w)
        b_list.append(b)
        mse_list.append(l_sum / 3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = np.array(w_list)
Y = np.array(b_list)
# X, Y = np.meshgrid(X, Y)

print("X维度信息", X.shape)
print("Y维度信息", Y.shape)

Z = np.array(mse_list)
print("Z轴数据维度", Z.shape)

#ax.plot_surface(X, Y, Z,cmap="rainbow")
surf = ax.plot_trisurf(X, Y, Z, cmap="rainbow")

ax.set_xlabel('w', color='b')
ax.set_ylabel('b', color='g')
ax.set_zlabel('Loss', color='r')
# 加上图例表
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.draw()
plt.show()

