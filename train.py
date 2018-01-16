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

def train(batch_size, epochs, summary_dir=None, load_file=None, save_file=None):
    report_frequency = 500
    save_frequency = 50000
    test_frequency = 25000
    learning_rate = 1e-4
    weight_decay = 0.0004
    train_file = "FlyingThings3D_release_TRAIN.list"
    test_file = "FlyingThings3D_release_TEST.list"

    loss_weights_update_steps = [50,000, 100000, 150000, 250000, 350000, 4500000]
    loss_weights_index = 0
    loss_weights_updates = [[0, 0, 0, 0, 0, 1.0],                                          
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
    train_dataset = (tf.contrib.data.TextLineDataset(train_file)
                                    .repeat(epochs)
                                    .shuffle(buffer_size=22390)
                                    .map(data_map)
                                    .map(data_augment)
                                    .batch(batch_size))

    test_dataset = (tf.contrib.data.TextLineDataset(test_file) 
                                   .map(data_map)
                                   .map(data_crop)
                                   .batch(batch_size))

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
                batch = sess.run(get_next)
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
                if (steps_since_last_report >=  report_frequency):
                    steps_since_last_report -= report_frequency
                    print("train step {} loss {}".format(step, loss))
                    sys.stdout.flush()

                steps_since_last_save += batch_size
                if (steps_since_last_save >=  save_frequency):
                    steps_since_last_save -= save_frequency
                    if save_file:
                        save_network(save_file)

                steps_since_last_test += batch_size
                if(steps_since_last_test >= test_frequency):
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
            batch_size = batch["img_left"].shape[0]

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

    nans = np.count_nonzero(np.isnan(data))
    if nans != 0:
        print("load_pfm: warning {} nans encountered".format(nan))

    shape = (height, width, 1)
    data =  np.reshape(data, shape) * scale
    data = np.flipud(data)

    return data

def data_map(s):
    s = tf.string_split([s], delimiter="\t")

    example = dict()
    example["img_left"] = load_image(s.values[0])
    example["img_right"] = load_image(s.values[1])
    example["disp"] = tf.py_func(load_pfm, [s.values[2]], tf.float32)

    return example

def data_crop(d):
    d["img_left"] = tf.reshape(tf.image.resize_image_with_crop_or_pad(d["img_left"], 384, 768), [384, 768, 3])
    d["img_right"] = tf.reshape(tf.image.resize_image_with_crop_or_pad(d["img_right"], 384, 768), [ 384, 768, 3])
    d["disp"] = tf.reshape(tf.image.resize_image_with_crop_or_pad(d["disp"], 384, 768), [384, 768, 1])

    return d

def data_augment(d):
    return data_crop(d)

if __name__ == '__main__':
    train(32, 1000, summary_dir="summaries", save_file="save")
