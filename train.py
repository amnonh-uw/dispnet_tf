import tensorflow as tf
from dispnet import DispNet
from PIL import Image
import numpy as np
import re
import os
import sys
import time

dataset_w = 960
dataset_h = 540
dispnet_w = 768
dispnet_h = 384

# We
# set β1 = 0.9 and β2 = 0.999 as in Kingma et al. [14]. As
# learning rate we used λ = 1e − 4 and divided it by 2 every
# 200 000 iterations starting from iteration 400 000.

def train(batch_size, epochs, summary_dir=None, load_file=None, save_file=None):
    report_frequency = 500
    save_frequency = 50000
    test_frequency = 25000
    learning_rate = 1e-4
    weight_decay = 0.0004
    train_file = "FlyingThings3D_release_TRAIN.list"
    test_file = "FlyingThings3D_release_TEST.list"

    loss_weights_update_steps = [50000, 100000, 150000, 250000, 350000, 4500000]
    loss_weights_index = 0
    loss_weights_updates = [[0, 0, 0, 0, 0.5, 1.0],
                            [0, 0, 0, 0.2, 1., 0.5],
                            [0, 0, 0.2, 1, 0.5, 0],
                            [0, 0.2, 1., 0.5, 0, 0],
                            [0.2, 1, 0.5, 0, 0, 0],
                            [1.0, 0.5, 0, 0, 0, 0],
                            [1.0, 0, 0, 0, 0, 0]]

    loss_weights = np.array(loss_weights_updates[0], dtype=np.float32)
    print("weights update: {}".format(loss_weights))

    dispnet = DispNet()

    print("creating datasets")
    sys.stdout.flush()
    train_dataset = create_train_dataset(train_file, epochs, batch_size)
    print("train_dataset done")
    sys.stdout.flush()
    test_dataset = create_test_dataset(test_file, batch_size)
    print("test_dataset done")
    sys.stdout.flush()

    adam_learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
    train_op = tf.train.AdamOptimizer(learning_rate=adam_learning_rate).minimize(dispnet.loss)
    summaries_op = tf.summary.merge_all()

    if summary_dir:
        summary_writer = tf.summary.FileWriter(summary_dir, tf.get_default_graph())

    with tf.Session() as sess:
        step = 0
        steps_since_lr_update = None
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

        print("Training starting")
        sys.stdout.flush()
        while True:
            try:
                batch = time_fn(sess.run, get_next)
                batch_size = batch["img_left"].shape[0]

                feed_dict = {
                    adam_learning_rate: learning_rate,
                    dispnet.weight_decay: weight_decay,
                    dispnet.img_left: batch["img_left"],
                    dispnet.img_right: batch["img_right"],
                    dispnet.disp: batch["disp"],
                    dispnet.loss_weights: loss_weights
                }

                _, loss, summary = sess.run([train_op, dispnet.loss, summaries_op], feed_dict=feed_dict)

                loss /= np.sum(loss_weights)
                step += batch_size
                summary_writer.add_summary(summary, step)

                steps_since_last_report += batch_size
                if (steps_since_last_report >= report_frequency):
                    steps_since_last_report -= report_frequency
                    print("train step {} loss {}".format(step, loss))
                    sys.stdout.flush()

                steps_since_last_save += batch_size
                if (steps_since_last_save >= save_frequency):
                    steps_since_last_save -= save_frequency
                    if save_file:
                        save_network(save_file)

                steps_since_last_test += batch_size
                if (steps_since_last_test >= test_frequency):
                    steps_since_last_test -= test_frequency
                    test(dispnet, sess, test_dataset)

                if loss_weights_index < len(loss_weights_update_steps):
                    if step > loss_weights_update_steps[loss_weights_index]:
                        loss_weights_index += 1
                        loss_weights = np.array(loss_weights_updates[loss_weights_index], dtype=np.float32)
                        print("weights update: {}".format(loss_weights))

                if step >= 400000:
                    if steps_since_lr_update == None:
                        steps_since_lr_update = step - 400000
                        learning_rate /= 2
                        print("step {} new learning rate {}".format(step, learning_rate))
                        sys.stdout.flush()
                    else:
                        steps_since_lr_update += batch_size

                        if steps_since_lr_update >= 200000:
                            steps_since_lr_update -= 200000
                            learning_rate /= 2
                            print("step {} new learning rate {}".format(step, learning_rate))
                            sys.stdout.flush()

            except tf.errors.OutOfRangeError:
                break

        print("finished {} steps".foramt(step))
        sys.stdout.flush()
        if save_file:
            save_network(save_file)


def create_train_dataset(file, epochs, batch_size):
    train_dataset = (tf.contrib.data.TextLineDataset(file)
                     .repeat(epochs)
                     .shuffle(buffer_size=22390)
                     .map(data_map)
                     .map(data_augment)
                     .prefetch(batch_size * 2)
                     .batch(batch_size))
    return train_dataset


def create_test_dataset(file, batch_size):
    test_dataset = (tf.contrib.data.TextLineDataset(file)
                    .map(data_map)
                    .map(center_crop)
                    .batch(batch_size))
    return test_dataset


def test(dispnet, sess, test_dataset):
    loss_weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    iterator = test_dataset.make_initializable_iterator()
    get_next = iterator.get_next()
    sess.run(iterator.initializer)
    total_loss = 0.0
    count = 0

    while True:
        try:
            batch = sess.run(get_next)

            feed_dict = {
                dispnet.weight_decay: 0.0,
                dispnet.img_left: batch["img_left"],
                dispnet.img_right: batch["img_right"],
                dispnet.disp: batch["disp"],
                dispnet.loss_weights: loss_weights
            }

            loss = sess.run([dispnet.loss], feed_dict=feed_dict)
            total_loss += loss[0]

            count += 1
        except tf.errors.OutOfRangeError:
            break

    print("average loss on test set is {}".format(total_loss / float(count)))


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

    img_mean = tf.constant([120.16955729, 116.97606771, 106.57792824], dtype=tf.float32) / 255.0
    img_mean = tf.reshape(img_mean, [1, 1, 3])

    image = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
    image -= img_mean
    image = tf.reshape(image, [dataset_h, dataset_w, 3], name="load_image")

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
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')

    nans = np.count_nonzero(np.isnan(data))
    if nans != 0:
        print("load_pfm: warning {} nans encountered".format(nan))

    shape = (height, width, 1)
    data = np.reshape(data, shape) * scale
    data = np.flipud(data)

    return data

def data_map(s):
    s = tf.string_split([s], delimiter="\t")

    example = dict()
    example["img_left"] = load_image(s.values[0])
    example["img_right"] = load_image(s.values[1])
    disp = tf.py_func(load_pfm, [s.values[2]], tf.float32)
    example["disp"] = tf.reshape(disp, [dataset_h, dataset_w, 1], name="load_pfm")

    return example

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

def center_crop_im(im):
    im = tf.image.resize_image_with_crop_or_pad(im, dispnet_h, dispnet_w)
    return tf.reshape(im, [dispnet_h, dispnet_w, 3], name="center_crop_im")

def center_crop_disp(im):
    im = tf.image.resize_image_with_crop_or_pad(im, dispnet_h, dispnet_w)
    return tf.reshape(im, [dispnet_h, dispnet_w, 1], name="center_crop_disp")

def center_crop(d):
    d["img_left"] = center_crop_im(d["img_left"])
    d["img_right"] = center_crop_im(d["img_right"])
    d["disp"] = center_crop_disp(d["disp"])

    return d

def gen_spatial_params(w, h):
    # squeeze
    zoom_h = tf.exp(tf.random_uniform([], minval=-0.3, maxval=+0.3))
    # zoom_h = tf.constant(1., dtype=tf.float32)
    new_h = tf.to_int32(tf.round(tf.to_float(h) * zoom_h))

    # translate
    spare_w = tf.to_int32((w - dispnet_w) / 2)
    spare_h = tf.to_int32((new_h - dispnet_h) / 2)

    dw = tf.random_uniform([], minval=-spare_w, maxval=spare_w, dtype=tf.int32)
    dh = tf.random_uniform([], minval=-spare_h, maxval=spare_h, dtype=tf.int32)

    o_w = spare_w + dw
    o_h = spare_h + dh

    return  o_w, o_h, new_h

def spatial_augment_img(im, w, h, o_w, o_h, new_h):
    # squeeze
    im = tf.image.resize_images(im, [new_h, w])

    # crop
    im = tf.image.crop_to_bounding_box(im, o_h, o_w, dispnet_h, dispnet_w)

    return im

def data_augment(d):
    shape = tf.shape(d["img_left"])
    h = shape[0]
    w = shape[1]

    o_w, o_h, new_h = gen_spatial_params(w,h)

    d["img_left"] = spatial_augment_img(d["img_left"], w, h, o_w, o_h, new_h)
    d["img_right"] = spatial_augment_img(d["img_right"],  w, h, o_w, o_h, new_h)
    d["disp"] = spatial_augment_img(d["disp"],  w, h, o_w, o_h, new_h)

    return d


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    # Distort the color of a Tensor image.
    #     Each color distortion is non-commutative and thus ordering of the color ops
    #     matters. Ideally we would randomly permute the ordering of the color ops.
    #     Rather then adding that level of complication, we select a distinct ordering
    #     of color ops for each preprocessing thread.
    #     Args:
    #         image: 3-D Tensor containing single image in [0, 1].
    #         color_ordering: Python int, a type of distortion (valid values: 0-3).
    #         fast_mode: Avoids slower ops (random_hue and random_contrast)
    #         scope: Optional scope for name_scope.
    # Returns:
    #     3-D Tensor color-distorted image on range [0, 1]
    # Raises:
    #     ValueError: if color_ordering not in [0, 3]

  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def time_fn(fn, *args, **kwargs):
    start = time.clock()
    results = fn(*args, **kwargs)
    end = time.clock()
    fn_name = fn.__module__ + "." + fn.__name__
    print(fn_name + ": " + str(end - start) + "s")
    return results


if __name__ == '__main__':
    train(32, 1000, summary_dir="summaries_2", save_file="save_2")
