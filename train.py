import tensorflow as tf
from dispnet import DispNet
from PIL import Image
import numpy as np
import re
import os
import sys


# We
# set β1 = 0.9 and β2 = 0.999 as in Kingma et al. [14]. As
# learning rate we used λ = 1e − 4 and divided it by 2 every
# 200 000 iterations starting from iteration 400 000.
# Due to the depth of the networks and the direct connections
# between contracting and expanding layers (see Table
# 2), lower layers get mixed gradients if all six losses are
# active. We found that using a loss weight schedule can be
# beneficial: we start training with a loss weight of 1 assigned
# to the lowest resolution loss loss6 and a weight of 0 for
# all other losses (that is, all other losses are switched off).
# During training, we progressively increase the weights of
# losses with higher resolution and deactivate the low resolution
# losses. This enables the network to first learn a coarse
# representation and then proceed with finer resolutions without
# losses constraining intermediate features

def train(batch_size, epochs, summary_dir=None, load_file=None, save_file=None):
    report_frequency = 500
    save_frequency = 50000
    test_frequency = 10000
    learning_rate = 1e-4
    weight_decay = 0.0004
    train_file = "FlyingThings3D_release_TRAIN.list"
    test_file = "FlyingThings3D_release_TEST.list"
    loss_weights = np.array([1.0, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)

    dispnet = DispNet()

    train_dataset = tf.contrib.data.TextLineDataset(train_file)
    test_dataset = tf.contrib.data.TextLineDataset(test_file)
    train_dataset = train_dataset.map(data_map)
    test_dataset = test_dataset.map(data_map)
    train_dataset = train_dataset.map(data_augment)
    test_dataset = test_dataset.map(data_crop)
    train_dataset.shuffle(buffer_size=22390)
    train_dataset.repeat(epochs)
    train_dataset.batch(batch_size)
    test_dataset.batch(batch_size)

    # dataset augmentation needs to happen here
    # Despite the large training set, we
    # chose to perform data augmentation to introduce more diversity
    # into the training data at almost no extra cost12. We
    # perform spatial transformations (rotation, translation, cropping,
    # scaling) and chromatic transformations (color, contrast,
    # brightness), and we use the same transformation for
    # all 2 or 4 input images.
    # For disparity, introducing any rotation or vertical shift
    # would break the epipolar constraint. Horizontal shifts
    # would lead to negnegative disparities or shifting infinity towards
    # the camera.

    adam_learning_rate = tf.placeholder(tf.float32, [1], name='learning_rate')
    train_op = tf.train.AdamOptimizer(learning_rate=adam_learning_rate).minimize(dispnet.loss)
    summaries_op = tf.summary.merge_all()

    if summary_dir:
        summary_writer = tf.summary.FileWriter(summary_dir, tf.get_default_graph())

    with tf.Session() as sess:
        step = 0
        steps_since_lr_udpate = None
        steps_since_last_report = 0
        steps_since_last_save = 0
        steps_since_last_test = 0
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if load_file != None:
            load_network(load_file)

        iterator = train_dataset.make_initializable_iterator()
        get_next = iterator.get_next()
        sess.run(iterator.initializer)
        while True:
            try:
                batch = sess.run(get_next)
                batch_size = batch["img1"].shape()[0]

                feed_dict = {
                    adam_learning_rate: learning_rate,
                    dispnet.weight_decay: weight_decay,
                    dispnet.img1: batch["img1"],
                    dispnet.img2: batch["img2"],
                    dispnet.disp: batch["disp"],
                    dispnet.loss_weights: loss_weights
                }

                _, loss, summary = sess.run([train_op, dispnet.loss, summaries_op], feed_dict=feed_dict)

                step += batch_size
                summary_writer.add_summary(summary, step)

                steps_since_last_report += batch_size
                if (steps_since_last_report >  report_frequency):
                    steps_since_last_report -= report_frequency
                    print("train step {} loss {}".format(step, loss))
                    sys.stdout.flush()

                steps_since_last_save += batch_size
                if (steps_since_last_save >  save_frequency):
                    steps_since_last_save -= save_frequency
                    if save_file:
                        save_network(save_file)

                steps_since_last_test += batch_size
                if(steps_since_last_test > test_frequency):
                    steps_since_last_test -= test_frequency
                    test(dispnet, sess, test_dataset)

                if step > 400000:
                    if steps_since_lr_udpate == None:
                        steps_since_lr_update = step - 400000
                        learning_rate /= 2
                        print("new learning rate {}".format(learning_rate))
                        sys.stdout.flush()
                    else:
                        steps_since_lr_update += batch_size

                        if steps_since_lr_udpate > 200000:
                            steps_since_lr_update -= 200000
                            learning_rate /= 2
                            print("new learning rate {}".format(learning_rate))
                            sys.stdout.flush()

            except tf.errors.OutOfRangeError:
                break

def test(dispnet, sess, test_dataset):
    loss_weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    iterator = test_dataset.make_initializable_iterator()
    get_next = iterator.get_next()
    sess.run(iterator.initializer)
    total_loss = 0.0
    step = 0

    while True:
        try:
            batch = sess.run(get_next)
            batch_size = batch["img1"].shape()[0]

            feed_dict = {
                dispnet.weight_decay: 0.0,
                dispnet.img1: batch["img1"],
                dispnet.img2: batch["img2"],
                dispnet.disp: batch["disp"],
                dispnet.loss_weights: loss_weights
            }

            loss = sess.run([dispnet.loss], feed_dict=feed_dict)
            total_loss += loss

            step += batch_size
        except tf.errors.OutOfRangeError:
            break

        print("average loss on test set is {}".format(total_loss / float(step)))

def load_network(name):
    print("loading {}".format(name))
    name += "/"
    saver = tf.train.Saver(tf.trainable_variables())
    saver.restore(tf.get_default_session(), name)

def save_network(name):
    print("saving {}".format(name))
    name += "/"
    os.makedirs(os.path.dirname(name), exist_ok=True)
    saver = tf.train.Saver(tf.trainable_variables())
    saver.save(tf.get_default_session(), name)

def load_image(file):
    image_string = tf.read_file(file)
    image_decoded = tf.image.decode_image(image_string, channels=3)
    image = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
    return image

def load_pfm(name):
    with open(name, "rb") as file:
        header = file.readline().decode('utf-8').rstrip()
        if header == 'PF':
            raise Exception("expecting non color PFM")
        elif header != 'Pf':
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().decode('utf-8').rstrip())
        if scale < 0: # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>' # big-endian

        data = np.fromfile(file, endian + 'f')

    shape = (height, width, 1)
    return np.reshape(data, shape) * scale

def data_map(s):
    s = tf.string_split([s], delimiter="\t")

    example = dict()
    example["img1"] = load_image(s.values[0])
    example["img2"] = load_image(s.values[1])
    example["disp"] = tf.py_func(load_pfm, [s.values[2]], tf.float32)

    return example

def data_crop(d):
    d["img1"] = tf.reshape(tf.image.resize_image_with_crop_or_pad(d["img1"], 768, 384), [-1, 768, 384, 3])
    d["img2"] = tf.reshape(tf.image.resize_image_with_crop_or_pad(d["img2"], 768, 384), [-1, 768, 384, 3])
    d["disp"] = tf.reshape(tf.image.resize_image_with_crop_or_pad(d["disp"], 768, 384), [-1, 768, 384, 1])

    return d

def data_augment(d):
    return data_crop(d)

if __name__ == '__main__':
    train(32, 100, summary_dir="summaries")