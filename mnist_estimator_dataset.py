import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.DEBUG)


def lenet(x, is_training):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
    conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

    conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
    conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

    fc1 = tf.contrib.layers.flatten(conv2)
    fc1 = tf.layers.dense(fc1, 1024)
    fc1 = tf.layers.dropout(fc1, rate=0.4, training=is_training)
    return tf.layers.dense(fc1, 10)


def model_fn(features, labels, mode, params):
    predict = lenet(features, mode == tf.estimator.ModeKeys.TRAIN)

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


mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

model_params = {"learning_rate": 0.01}
estimator = tf.estimator.Estimator(model_fn=model_fn, params=model_params, model_dir="models/mnist_example")

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"image": mnist.train.images},
    y=mnist.train.labels.astype(np.int32),
    num_epochs=None,
    batch_size=128,
    shuffle=True)


def dataset_input_fn(file_path):
    # 输入数据使用本章第一节（1. TFRecord样例程序.ipynb）生成的训练和测试数据。# 输入数据使
    train_files = tf.train.match_filenames_once(file_path)

    # 解析一个TFRecord的方法。
    def parser(record):
        features = tf.parse_single_example(
            record,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'pixels': tf.FixedLenFeature([], tf.int64),
                'label': tf.FixedLenFeature([], tf.int64)
            })
        decoded_images = tf.decode_raw(features['image_raw'], tf.uint8)
        retyped_images = tf.cast(decoded_images, tf.float32)
        images = tf.reshape(retyped_images, [784])
        labels = tf.cast(features['label'], tf.int32)
        # pixels = tf.cast(features['pixels'],tf.int32)
        return images, labels

    # 定义读取训练数据的数据集。
    dataset = tf.data.TFRecordDataset(train_files)
    dataset = dataset.map(parser)

    # 对数据进行shuffle和batching操作。这里省略了对图像做随机调整的预处理步骤。
    dataset = dataset.shuffle(10000).batch(100)
    dataset = dataset.repeat(10)

    # 定义数据集迭代器。
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()
    # with tf.Session() as sess:
    #     sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
    #     sess.run(iterator.initializer)

    return image_batch, label_batch

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# first_batch = sess.run(dataset_input_fn())
# print(first_batch)

# 训练模型
estimator.train(input_fn=lambda: dataset_input_fn("output.tfrecords"), steps=1000)


test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"image": mnist.test.images},
      y=mnist.test.labels.astype(np.int32),
      num_epochs=1,
      batch_size=128,
      shuffle=False)

test_results = estimator.evaluate(input_fn=test_input_fn)
accuracy_score = test_results["accuracy"]
print("\nTest accuracy: %g %%" % (accuracy_score*100))


predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"image": mnist.test.images[:10]},
      num_epochs=1,
      shuffle=False)

predictions = estimator.predict(input_fn=predict_input_fn)
for i, p in enumerate(predictions):
    print("Prediction %s: %s" % (i + 1, p["result"]))
