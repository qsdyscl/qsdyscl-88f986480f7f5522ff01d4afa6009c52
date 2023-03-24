# 线性回归：矩阵分解法
# 通过分解矩阵的方法求解有时更高效并且数值稳定
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

ops.reset_default_graph()
# 创建会话
sess = tf.Session()
# 创建数据
x_vals = np.linspace(0, 10, 120)
y_vals = x_vals + np.random.normal(0, 1, 120)
# 创建设计矩阵
x_vals_column = np.transpose(np.matrix(x_vals))
ones_column = np.transpose(np.matrix(np.repeat(1, 120)))
A = np.column_stack((x_vals_column, ones_column))
# 矩阵矩阵b
b = np.transpose(np.matrix(y_vals))
# 创建张量
A_tensor = tf.constant(A)
b_tensor = tf.constant(b)
# 找到方阵的Cholesky矩阵分解
tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
# TensorFlow的cholesky（）函数仅仅返回矩阵分解的下三角矩阵，
# 因为上三角矩阵是下三角矩阵的转置矩阵。
L = tf.cholesky(tA_A)
# Solve L*y=t(A)*b
tA_b = tf.matmul(tf.transpose(A_tensor), b)
sol1 = tf.matrix_solve(L, tA_b)
# Solve L' * y = sol1
sol2 = tf.matrix_solve(tf.transpose(L), sol1)
solution_eval = sess.run(sol2)
# 抽取系数
slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]
print('slope: ' + str(slope))
print('y_intercept: ' + str(y_intercept))
# 获得最适合的线
best_fit = []
for i in x_vals:
    best_fit.append(slope * i + y_intercept)
# 绘制结果
plt.plot(x_vals, y_vals, 'rs', label='Data')
plt.plot(x_vals, best_fit, 'k-', label='Best fit line', linewidth=2.5)
plt.legend(loc='upper left')
plt.show()
