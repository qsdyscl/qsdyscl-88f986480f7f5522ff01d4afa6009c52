import numpy as np
import tensorflow as tf
from tensorflow_core.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
from mnist import MNIST

mndata = MNIST('MNIST_data')
mndata.gz = True
sess = tf.Session()
x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b
y_ = tf.placeholder("float", [None, 10])
# 使用Tensorflow自带的交叉熵函数
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(500)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
images, labels = mndata.load_testing()
num = 9000
image = images[num]
label = labels[num]
# 打印图片
print(mndata.display(image))
print('这张图片的实际数字是: ' + str(label))
# 测试新图片，并输出预测值
a = np.array(image).reshape(1, 784)
y = tf.nn.softmax(y)  # 为了打印出预测值，我们这里增加一步通过softmax函数处理后来输出一个向量
result = sess.run(y, feed_dict={x: a})  # result是一个向量，通过索引来判断图片数字
print('预测值为：')
print(result)
