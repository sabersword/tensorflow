# load MNIST data

import numpy as np

from collections import namedtuple




images = np.arange(9).reshape(3,3)
perm = np.arange(images.shape[0])
np.random.shuffle(perm)
images = images[[1,2],[2,2]]

websites = [
    ('Sohu', 'http://www.google.com/', u'张朝阳'),
    ('Sina', 'http://www.sina.com.cn/', u'王志东'),
    ('163', 'http://www.163.com/', u'丁磊')
]

Website = namedtuple('Website', ['name', 'url', 'founder'])

for website in websites:
    website = Website._make(website)
    print(website.url)

list = np.arange(100)
x = list.reshape(2,5,10)
print(x)
print(x[1,2])

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# start tensorflow interactiveSession
import tensorflow as tf
sess = tf.InteractiveSession()

# weight initialization
# tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
# 生产正态分布,均值为0 标准差为0.1
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    # 声明一个变量
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# convolution
'''
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一
第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，SAME的话卷积核中心可以在输入图像边缘, VALID的话卷积核边缘最多与输入图像边缘重叠
第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
'''
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# pooling
'''
tf.nn.max_pool(value, ksize, strides, padding, name=None)
参数是四个，和卷积很类似：
第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'
返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
'''
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Create the model
# placeholder
# 占位符,在session运行的时候通过feed_dict输入训练样本,与variable不同,不用事先指定数据
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# softmax就是将每个值以e为底计算指数,并归一化
y = tf.nn.softmax(tf.matmul(x,W) + b)

# first convolutinal layer
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# 重新调整张量的维度,如下-1表示不计算,其余3个维度调整为28,28,1的四维张量
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 计算修正线性单元(非常常用)：max(features, 0).并且返回和feature一样的形状的tensor。
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
w_conv2 = weight_variable([5, 5, 32, 32])
b_conv2 = bias_variable([32])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
w_fc1 = weight_variable([7*7*32, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout
'''
tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
x            :  输入tensor
keep_prob    :  float类型，每个元素被保留下来的概率
noise_shape  : 一个1维的int32张量，代表了随机产生“保留/丢弃”标志的shape。
seed         : 整形变量，随机数种子。
name         : 名字，没啥用。
'''
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# train and evaluate the model
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
#train_step = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)
# 最小化这个的一个操作
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
# tf.argmax,它能给出某个tensor对象在某一维上的其数据最大值所在的索引值
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
        print("step %d, train accuracy %g" %(i, train_accuracy))
    train_step.run(session=sess, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

print("test accuracy %g" % accuracy.eval(session=sess, feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))
