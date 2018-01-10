import tensorflow as tf
from dispnet import DispNet
from PIL import Image
import numpy as np
import re
import os


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

def train(num_loss, examples_file, batch_size, epochs, summary_dir=None, load_file=None, save_file=None):
    report_frequency = 500
    save_frequency = 5000
    learning_rate = 1e-4

    dispnet = DispNet()

    dataset = tf.contrib.data.TextLineDataset(examples_file)
    dataset = dataset.map(data_map)
    dataset = dataset.map(data_crop)
    dataset.shuffle(buffer_size=22390)
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

    dataset.repeat(epochs)
    dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    get_next = iterator.get_next()

    # adam_learning_rate = tf.placeholder(tf.float32, [1], name='learning_rate')
    adam_learning_rate = learning_rate
    train_op = tf.train.AdamOptimizer(learning_rate=adam_learning_rate).minimize(dispnet.loss)
    summaries_op = tf.summary.merge_all()

    if summary_dir:
        summary_writer = tf.summary.FileWriter(summary_dir, tf.get_default_graph())

    weights = np.zeros([6], dtype=np.float32)
    weights[num_loss] = 1.0


    with tf.Session() as sess:
        step = 1
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if load_file != None:
            load_network(load_file)

        sess.run(iterator.initializer)
        while True:
            try:
                batch = sess.run(get_next)

                feed_dict = {
                    adam_learning_rate: learning_rate,
                    dispnet.img1: batch["img1"],
                    dispnet.img2: batch["img2"],
                    dispnet.disp: batch["disp"],
                    dispnet.weights: weights
                }

                _, loss, summary = sess.run([train_op, dispnet.loss, summaries_op], feed_dict=feed_dict)

                summary_writer.add_summary(summary, step)
                step += 1
                if (step % report_frequency == 0):
                    print("train step {} loss {}".format(step, loss))

                if (step % save_frequency == 0) and save_file:
                    save_network(save_file)

            except tf.errors.OutOfRangeError:
                break

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

    shape = (height, width)
    return np.reshape(data, shape) * scale

def data_map(s):
    s = tf.string_split([s], delimiter="\t")

    example = dict()
    example["img1"] = tf.expand_dims(load_image(s.values[0]), axis=0)
    example["img2"] = tf.expand_dims(load_image(s.values[1]), axis=0)
    example["disp"] = tf.expand_dims(tf.expand_dims(tf.py_func(load_pfm, [s.values[2]], tf.float32), axis=0), axis=-1)

    return example

def data_crop(d):
    d["img1"] = tf.image.resize_image_with_crop_or_pad(d["img1"], 768, 364)
    d["img2"] = tf.image.resize_image_with_crop_or_pad(d["img2"], 768, 364)
    d["disp"] = tf.image.resize_image_with_crop_or_pad(d["disp"], 768, 364)

    return d

if __name__ == '__main__':
    train(0, "FlyingThings3D_release_TRAIN.list", 32, 10, summary_dir="summaries")