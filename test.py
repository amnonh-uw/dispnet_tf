import tensorflow as tf
from dispnet import DispNet
import sys
from train import load_network, test, create_test_dataset

def run_test(batch_size, load_file):
    test_file = "FlyingThings3D_release_TEST.list"

    dispnet = DispNet()

    print("creating dataset")
    sys.stdout.flush()

    test_dataset = create_test_dataset(test_file, batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        load_network(load_file)
        test(dispnet, sess, test_dataset)

if __name__ == '__main__':
    run_test(32, "save")
