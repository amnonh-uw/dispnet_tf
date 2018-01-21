import tensorflow as tf
import numpy as np
from dispnet import DispNet
import sys
from train import load_network, test, create_test_dataset, create_train_dataset

def run_test(batch_size, load_file, test_file, summary_dir=None):
    dispnet = DispNet()
    if summary_dir:
        summary_writer = tf.summary.FileWriter(summary_dir, tf.get_default_graph())
        summaries_op = tf.summary.merge_all()
    else:
        summary_writer = None
        summaries_op = None

    sys.stdout.flush()

    test_dataset = create_test_dataset(test_file, batch_size)
    # test_dataset = create_train_dataset(test_file, 1, batch_size)
    loss_weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    print("testing {} with weights {}".format(test_file, loss_weights))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        load_network(load_file)
        test(dispnet, sess, test_dataset, loss_weights, verbose=True, summaries_op=summaries_op, summary_writer=summary_writer)

if __name__ == '__main__':
    test_file = "FlyingThings3D_release_TEST.list"
    # run_test(1, "save", test_file, summary_dir="test_summary")
    # run_test(1, "no_color_aug_save", test_file, summary_dir="no_color_aug_test_summary")
    run_test(1, "save", test_file, summary_dir="test_summary")
