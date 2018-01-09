import tensorflow as tf
from dispnet import DispNet
from PIL import Image
import numpy as np
import re
import os

def train(num_loss, examples_file, batch_size, epochs, learning_rate, summary_dir=None, load_file=None, save_file=None):
    dispnet = DispNet()

    dataset = tf.data.TextLineDataset(examples_file)
    dataset = dataset.map(data_map)
    dataset.shuffle(buffer_size=22390)
    # dataset augmentation needs to happen here
    dataset.repeat(epochs)
    dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    get_next = iterator.get_next()

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(dispnet.loss)
    summaries_op = tf.summary.merge_all()

    if summary_dir:
        summary_writer = tf.summary.FileWriter(summary_dir, tf.get_default_graph())

    weights = np.zeros([6], dtype=np.float32)
    weights[num_loss] = 1.0
    report_frequency = 500
    save_frequency = 5000

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
                    dispnet.img1: batch["img1"],
                    dispnet.img2: batch["img2"],
                    dispnet.disp: batch["disp"],
                    dispnet.weights: weights
                }

                _, loss, summary = sess.run([train_op, loss, summaries_op], feed_dict=feed_dict)

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
    image = tf.cast(image_decoded, tf.float32)
    return image

def load_pfm(file):
  header = file.readline().rstrip()
  if header == 'PF':
    raise Exception("expecting non color PFM")
  elif header != 'Pf':
    raise Exception('Not a PFM file.')

  dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
  if dim_match:
    width, height = map(int, dim_match.groups())
  else:
    raise Exception('Malformed PFM header.')

  scale = float(file.readline().rstrip())
  if scale < 0: # little-endian
    endian = '<'
    scale = -scale
  else:
    endian = '>' # big-endian

  data = np.fromfile(file, endian + 'f')
  shape = (height, width)
  return np.reshape(data, shape) * scale

def data_map(s):
    s = tf.string_split([s], delimiter="\t", skip_empty=False)

    example = dict()
    example["img1"] = load_image(s.values[0])
    example["img2"] = load_image(s.values[1])
    example["disp"] = tf.py_func(load_pfm, [s.values[2]], tf.float32)

    return example

if __name__ == '__main__':
    train(0, "FlyingThings3D_release_TRAIN.list", 32, 10, 1e-5, summary_dir="summaries")