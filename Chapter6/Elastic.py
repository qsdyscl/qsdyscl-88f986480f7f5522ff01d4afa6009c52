# 用tensorflow实现弹性网络算法(多变量)
# 使用鸢尾花数据集，后三个特征作为特征，用来预测第一个特征。
# 导入必要的编程库，创建计算图，加载数据集
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn import datasets
from tensorflow.python.framework import ops

ops.get_default_graph()
sess = tf.Session()
iris = datasets.load_iris()
# 加载数据集。这次x_vals数据将是三列值的数组
x_vals = np.array([[x[1], x[2], x[3]] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])
# 声明学习率，批量大小，占位符和模型变量,模型输出
learning_rate = 0.001
batch_size = 50
x_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)  # 占位符大小为3
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[3, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))
model_output = tf.add(tf.matmul(x_data, A), b)
# 对于弹性网络回归算法，损失函数包括L1正则和L2正则
elastic_param1 = tf.constant(1.)
elastic_param2 = tf.constant(1.)
l1_a_loss = tf.reduce_mean(abs(A))
l2_a_loss = tf.reduce_mean(tf.square(A))
e1_term = tf.multiply(elastic_param1, l1_a_loss)
e2_term = tf.multiply(elastic_param2, l2_a_loss)
loss = tf.expand_dims(tf.add(tf.add(tf.reduce_mean(tf.square(y_target - model_output)), e1_term), e2_term), 0)
# 初始化变量， 声明优化器， 然后遍历迭代运行， 训练拟合得到参数
init = tf.global_variables_initializer()
sess.run(init)
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)
loss_vec = []
for i in range(1000):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    if (i + 1) % 250 == 0:
        print('Step#' + str(i + 1) + 'A = ' + str(sess.run(A)) + 'b=' + str(sess.run(b)))
        print('Loss= ' + str(temp_loss))
    # 现在能观察到， 随着训练迭代后损失函数已收敛。
plt.plot(loss_vec, 'k--')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
