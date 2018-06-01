import numpy as np
import tensorflow as tf
import operator
# from captcha.captcha_write_tfrecord import TRAIN_DATA_NUM, TEST_DATA_NUM

tf.logging.set_verbosity(tf.logging.DEBUG)


def lenet(x, is_training):
    x = tf.reshape(x, shape=[-1, 40, 35, 3])

    conv1 = tf.layers.conv2d(x, 96, 5, activation=tf.nn.relu)
    conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

    conv2 = tf.layers.conv2d(conv1, 192, 3, activation=tf.nn.relu)
    conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

    fc1 = tf.contrib.layers.flatten(conv2)
    fc1 = tf.layers.dense(fc1, 4096)
    fc1 = tf.layers.dropout(fc1, rate=0.5, training=is_training)
    return tf.layers.dense(fc1, 35)


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


def recognize(img):
    if not operator.eq(img.shape, (40, 100, 3)):
        raise TypeError
    regions = np.zeros(shape=(4, 40, 35, 3), dtype=np.float32)
    regions[0] = img[:, 0:35, :]
    regions[1] = img[:, 20:55, :]
    regions[2] = img[:, 45:80, :]
    regions[3] = img[:, 65:100, :]

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
          x={"image": regions},
          num_epochs=1,
          shuffle=False)

    predictions = estimator.predict(input_fn=predict_input_fn)
    captcha = ''
    for i, p in enumerate(predictions):
        print("Prediction %s: %s" % (i + 1, p["result"]))
        if int(p["result"]) > 9:
            captcha += chr(p["result"] + 65 - 10)
        else:
            captcha += str(p["result"])
    return captcha

estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir="D:/pythonProject/tfcaptcha/models/captcha")
