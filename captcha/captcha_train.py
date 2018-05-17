import numpy as np
import tensorflow as tf
from captcha.captcha_write_tfrecord import TRAIN_DATA_NUM, TEST_DATA_NUM

tf.logging.set_verbosity(tf.logging.DEBUG)


def lenet(x, is_training):
    x = tf.reshape(x, shape=[-1, 40, 40, 3])

    conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
    conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

    conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
    conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

    fc1 = tf.contrib.layers.flatten(conv2)
    fc1 = tf.layers.dense(fc1, 1024)
    fc1 = tf.layers.dropout(fc1, rate=0.4, training=is_training)
    return tf.layers.dense(fc1, 36)


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


def read_dataset(tfPath, num):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([tfPath])
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

    images = np.zeros(shape=[num, 40, 40, 3])
    labels = np.zeros(shape=[num])

    for i in range(num):
        images[i] = sess.run(image).reshape([-1, 40, 40, 3])
        labels[i] = sess.run(label)

    images = images.astype(np.float32)
    labels = labels.astype(np.int32)
    return images, labels

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    train_images, train_labels = read_dataset("captcha.train.noise.tfrecords", TRAIN_DATA_NUM)
    test_images, test_labels = read_dataset("captcha.login.tfrecords", TEST_DATA_NUM)

    # 读取训练集
    # reader = tf.TFRecordReader()
    # filename_queue = tf.train.string_input_producer(["captcha.train.tfrecords"])
    # _, serialized_example = reader.read(filename_queue)
    # features = tf.parse_single_example(
    #     serialized_example,
    #     features={
    #         'image': tf.FixedLenFeature([], tf.string),
    #         'label': tf.FixedLenFeature([], tf.int64)
    #     })
    #
    # image = tf.decode_raw(features['image'], tf.float32)
    # label = features['label']
    #
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #
    # train_images = np.zeros(shape=[TRAIN_DATA_NUM, 40, 40, 1])
    # train_labels = np.zeros(shape=[TRAIN_DATA_NUM])
    #
    # for i in range(TRAIN_DATA_NUM):
    #     train_images[i] = sess.run(image).reshape([-1, 40, 40, 1])
    #     train_labels[i] = sess.run(label)
    #
    # train_images = train_images.astype(np.float32)
    # train_labels = train_labels.astype(np.int32)

    model_params = {"learning_rate": 0.01}
    estimator = tf.estimator.Estimator(model_fn=model_fn, params=model_params, model_dir="models/noise")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"image": train_images},
        y=train_labels,
        num_epochs=None,
        batch_size=50,
        shuffle=True)

    # 训练模型
    # estimator.train(input_fn=train_input_fn, steps=6000)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
          x={"image": test_images},
          y=test_labels,
          num_epochs=1,
          batch_size=100,
          shuffle=False)

    test_results = estimator.evaluate(input_fn=test_input_fn)
    accuracy_score = test_results["accuracy"]
    print("\nTest accuracy: %g %%" % (accuracy_score*100))

    # predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    #       x={"image": mnist.test.images[:10]},
    #       num_epochs=1,
    #       shuffle=False)
    #
    # predictions = estimator.predict(input_fn=predict_input_fn)
    # for i, p in enumerate(predictions):
    #     print("Prediction %s: %s" % (i + 1, p["result"]))
