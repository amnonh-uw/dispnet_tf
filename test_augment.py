import tensorflow as tf
from train import create_test_dataset, save_image, copy_image
from PIL import Image
import numpy as np
import re
import os
import sys
import time


def test_augment(save_dir):
    train_file = "FlyingThings3D_release_TRAIN.list"
    test_file = "FlyingThings3D_release_TEST.list"

    test_dataset = create_test_dataset(test_file, 1)
    print("test_dataset done")
    sys.stdout.flush()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        iterator = test_dataset.make_initializable_iterator()
        get_next = iterator.get_next()
        sess.run(iterator.initializer)

        count = 0
        while True:
            try:
                batch = sess.run(get_next)
                batch_size = batch["img_left"].shape[0]

                count += batch_size
                save_image(batch["img_left"], save_dir + '/' + str(count) + "aug_left.png")
                save_image(batch["img_right"], save_dir + '/' + str(count) + "aug_right.png")
                copy_image(batch["img_left_file"], save_dir + '/' + str(count) + "orig_left.png")
                copy_image(batch["img_right_file"], save_dir + '/' + str(count) + "orig_right.png")

                break
            except tf.errors.OutOfRangeError:
                break

if __name__ == '__main__':
    test_augment("test_aug_out")
