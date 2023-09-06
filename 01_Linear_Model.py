import numpy as np
import matplotlib.pyplot as plt

# LinearModel 1 : y^ = x * w

# Prepare the train set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x): # 前馈函数
    return x * w

def loss(x, y): # loss function
    y_pred = forward(x) # y^
    return (y_pred - y) * (y_pred - y)

w_list = []
mse_list = [] # mean square error
for w in np.arange(0.0, 4.1, 0.1):
    print('w=', w)
    loss_sum = 0
    for x_val, y_val in zip(x_data, y_data): # zip()用法可参考 https://www.runoob.com/python/python-func-zip.html
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        loss_sum += loss_val
        print('t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', loss_sum / 3)
    w_list.append(w)
    mse_list.append(loss_sum / 3)

plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()

# LinearModel 2 : y^ = x * w + b

x_data = [1.0, 2.0, 3.0] 
y_data = [2.0, 4.0, 6.0]

def forward2(x):
    return x * W + B

def loss2(x, y):
    y_pred = forward2(x)
    return (y_pred - y) * (y_pred - y)

ww = np.arange(0.0, 4.1, 0.1)
bb = np.arange(-2.0, 2.5, 0.5)
W, B = np.meshgrid(ww, bb)

loss_sum = 0
for x_val, y_val in zip(x_data, y_data):
    y_pred_val = forward2(x_val)
    loss_val = loss2(x_val, y_val)
    loss_sum += loss_val
    MSE = loss_sum / 3
    print('t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', MSE)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(W, B, MSE,  cmap=plt.cm.coolwarm)
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_zlabel("cost value")
plt.show()
