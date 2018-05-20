import numpy as np
import os
import tensorflow as tf
from captcha.captcha_write_tfrecord import TRAIN_DATA_NUM

tf.logging.set_verbosity(tf.logging.DEBUG)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def lenet(x, is_training):
    x = tf.reshape(x, shape=[-1, 40, 40, 1])

    conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
    conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

    conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
    conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

    fc1 = tf.contrib.layers.flatten(conv2)
    fc1 = tf.layers.dense(fc1, 1024)
    fc1 = tf.layers.dropout(fc1, rate=0.4, training=is_training)
    return tf.layers.dense(fc1, 10)


def model_fn(features, labels, mode, params):
    predict = lenet(features["image"], mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"result": tf.argmax(predict, 1)})

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict, labels=labels))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"])

    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    eval_metric_ops = {"accuracy": tf.metrics.accuracy(tf.argmax(predict, 1), labels)}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def read_images_and_labels_by_estimator(image_root_dir):
    image_dirs = os.listdir(image_root_dir)
    image_array = np.zeros(shape=(200, 40, 40, 1))
    label_array = np.zeros(shape=200, dtype=np.int32)
    i = 0
    for image_dir in image_dirs:
        label = int(image_dir)
        for image_file in os.listdir(os.path.join(image_root_dir, image_dir)):
            image_raw_data = tf.gfile.FastGFile(os.path.join(image_root_dir, image_dir, image_file), 'rb').read()
            # 灰度图像可以收敛 奇怪
            img_data = tf.image.rgb_to_grayscale(tf.image.decode_jpeg(image_raw_data)).eval() / 255
            # img_data = tf.image.decode_jpeg(image_raw_data).eval() / 255
            image_array[i] = img_data
            label_array[i] = label
            i += 1
            print("i=", i)
    image_array = tf.cast(image_array, tf.float32).eval()
    # label_array = tf.cast(label_array, tf.float32).eval()
    return image_array, label_array


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    # 读取图片训练集
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(["captcha.train.tfrecords"])
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })

    image = tf.decode_raw(features['image'], tf.float32)
    label = features['label']

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(TRAIN_DATA_NUM):
        image_view = sess.run(image)
        label_view = sess.run(label)

    # model_params = {"learning_rate": 0.0001}
    # estimator = tf.estimator.Estimator(model_fn=model_fn, params=model_params, model_dir="captcha_models")

    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"image": images},
    #     y=labels.astype(np.int32),
    #     num_epochs=None,
    #     batch_size=10,
    #     shuffle=True)

    # 训练模型
    # estimator.train(input_fn=train_input_fn, steps=10000)

