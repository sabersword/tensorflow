import tensorflow as tf


# 解析一个TFRecord的方法。
def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'pixels': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        })
    decoded_images = tf.decode_raw(features['image_raw'],tf.uint8)
    retyped_images = tf.cast(decoded_images, tf.float32)
    images = tf.reshape(retyped_images, [784])
    labels = tf.cast(features['label'],tf.int32)
    # pixels = tf.cast(features['pixels'],tf.int32)
    return images, labels

# 从TFRecord文件创建数据集。这里可以提供多个文件。
input_files = ["output.tfrecords"]
dataset = tf.data.TFRecordDataset(input_files)

# map()函数表示对数据集中的每一条数据进行调用解析方法。
dataset = dataset.map(parser)

# 定义遍历数据集的迭代器。
iterator = dataset.make_one_shot_iterator()

# 读取数据，可用于进一步计算
image, label = iterator.get_next()
image2, label2 = iterator.get_next()

with tf.Session() as sess:
    for i in range(10):
        x, y = sess.run([image, label])
        print(y)

with tf.Session() as sess:
    for i in range(10):
        x, y = sess.run([image2, label2])
        print(y)