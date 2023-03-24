# lasso回归和岭回归
import matplotlib.pyplot as plt
import sys
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops

# 指定“岭”或“lasso”
regression_type = 'LASSO'
# 清除旧图
ops.reset_default_graph()
# 创建会话
sess = tf.Session()
# 载入数据
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])
# 设置模型参数
# 声明批量大小
batch_size = 50
# 初始化占位符
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
# 使结果可重现
seed = 13
np.random.seed(seed)
tf.set_random_seed(seed)
# 为线性回归创建变量
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))
# 声明模型操作
model_output = tf.add(tf.matmul(x_data, A), b)
# Loss函数
# 根据回归类型选择适当的损失函数
if regression_type == 'LASSO':
    # 声明 Lasso 损失函数
    # 增加损失函数，其为改良过的连续阶跃函数，lasso回归的截止点设为0.9。
    # 这意味着限制斜率系数不超过0.9
    # Lasso 损失= L2_Loss + heavyside_step,
    # Where heavyside_step ~ 0 if A < constant, otherwise ~ 99
    lasso_param = tf.constant(0.9)
    heavyside_step = tf.truediv(1., tf.add(1.,
                                         tf.exp(tf.multiply(-50., tf.subtract(A, lasso_param)))))
    regularization_param = tf.multiply(heavyside_step, 99.)
    loss = tf.add(tf.reduce_mean(tf.square(y_target - model_output)), regularization_param)
elif regression_type == 'Ridge':
    # 声明“岭”损失函数
    # 岭损失 = L2_loss + L2 范数
    ridge_param = tf.constant(1.)
    ridge_loss = tf.reduce_mean(tf.square(A))
    loss = tf.expand_dims(tf.add(tf.reduce_mean
                                 (tf.square(y_target - model_output)), tf.multiply(ridge_param, ridge_loss)), 0)
else:
    print('Invalid regression_type parameter value', file=sys.stderr)
# 优化
# 声明优化器
my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss)
# 运行回归
# 初始化变量
init = tf.global_variables_initializer()
sess.run(init)
# 训练循环
loss_vec = []
for i in range(1500):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss[0])
    if (i + 1) % 300 == 0:
        print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))
        print('\n')
# 提取回归结果
# 获得最佳系数
[slope] = sess.run(A)
[y_intercept] = sess.run(b)
# 获取最优拟合直线
best_fit = []
for i in x_vals:
    best_fit.append(slope * i + y_intercept)
# 结果绘图
# 根据数据点绘制回归线
plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()
# 随时间变化绘制损失曲线
plt.plot(loss_vec, 'k-')
plt.title(regression_type + ' Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
