import numpy as np
import os
import PIL.Image as Image
import matplotlib.pyplot as pyplot
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.DEBUG)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_images_and_labels_by_estimator(image_root_dir, number, channel=1, is_fill=False):
    image_dirs = os.listdir(image_root_dir)
    image_array = np.zeros(shape=(number, 40, 35, channel))
    label_array = np.zeros(shape=number, dtype=np.int32)
    i = 0
    for image_dir in image_dirs:
        label = int(image_dir)
        for image_file in os.listdir(os.path.join(image_root_dir, image_dir)):
            image_raw_data = tf.gfile.FastGFile(os.path.join(image_root_dir, image_dir, image_file), 'rb').read()
            # 灰度图像可以收敛 奇怪
            if channel == 1:
                img_data = tf.image.rgb_to_grayscale(tf.image.decode_jpeg(image_raw_data)).eval() / 255
            if channel == 3:
                img_data = tf.image.decode_jpeg(image_raw_data).eval() / 255
            if is_fill:
                fill = np.zeros(shape=[40, 5, channel])
                if channel == 1:
                    fill[:, :, :] = np.median(img_data)
                else:
                    fill[:, :, 0] = np.median(img_data[:, :, 0])
                    fill[:, :, 1] = np.median(img_data[:, :, 1])
                    fill[:, :, 2] = np.median(img_data[:, :, 2])
                image_array[i] = np.hstack((img_data, fill))
                # 打印图片
                # image_array[i] = image_array[i] * 255
                # r = Image.fromarray(image_array[i, :, :, 0]).convert('L')
                # g = Image.fromarray(image_array[i, :, :, 1]).convert('L')
                # b = Image.fromarray(image_array[i, :, :, 2]).convert('L')
                # image_show = Image.merge("RGB", (r, g, b))
                # pyplot.imshow(image_show)
                # pyplot.show()
            else:
                image_array[i] = img_data
            label_array[i] = label
            i += 1
            print("i=", i)
    image_array = tf.cast(image_array, tf.float32).eval()
    # label_array = tf.cast(label_array, tf.float32).eval()
    return image_array, label_array

TRAIN_DATA_NUM = 5830
TEST_DATA_NUM = 310


if __name__ == '__main__':

    sess = tf.InteractiveSession()
    # 读取图片训练集
    # images, labels = read_images_and_labels_by_estimator("train_data", TRAIN_DATA_NUM, 3)
    # writer = tf.python_io.TFRecordWriter("captcha.train.tfrecords")
    # for index in range(TRAIN_DATA_NUM):
    #     image = images[index].tostring()
    #     example = tf.train.Example(features=tf.train.Features(feature={
    #         'label': _int64_feature(labels[index]),
    #         'image': _bytes_feature(image)}))
    #     writer.write(example.SerializeToString())
    # writer.close()

    # 读取图片测试集
    images, labels = read_images_and_labels_by_estimator("test_data", TEST_DATA_NUM, 3)
    writer = tf.python_io.TFRecordWriter("captcha.test.tfrecords")
    for index in range(TEST_DATA_NUM):
        image = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(labels[index]),
            'image': _bytes_feature(image)}))
        writer.write(example.SerializeToString())
    writer.close()

    # 读取登录集
    # images, labels = read_images_and_labels_by_estimator("login_data", TEST_DATA_NUM, 3, TRUE)
    # writer = tf.python_io.TFRecordWriter("captcha.login.tfrecords")
    # for index in range(TEST_DATA_NUM):
    #     image = images[index].tostring()
    #     example = tf.train.Example(features=tf.train.Features(feature={
    #         'label': _int64_feature(labels[index]),
    #         'image': _bytes_feature(image)}))
    #     writer.write(example.SerializeToString())
    # writer.close()

