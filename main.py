import os
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
from numpy.random import seed
# from tensorflow import set_random_seed
from time import strftime
from PIL import Image
# https://stackoverflow.com/questions/59823283/could-not-load-dynamic-library-cudart64-101-dll-on-tensorflow-cpu-only-install
# https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#download-cuda-software
# (venv) tensorboard --logdir=C:\Users\JMAT\PycharmProjects\HandWrittingRecognize\tensorboard_mnist_digit_logs

tf.disable_v2_behavior()
num_examples = 0
index_in_epoch = 0


def next_batch(batch_size, data, labels):
    global num_examples
    global index_in_epoch

    start = index_in_epoch
    index_in_epoch += batch_size

    if index_in_epoch > num_examples:
        start = 0
        index_in_epoch = batch_size

    end = index_in_epoch

    return data[start:end], labels[start:end]


def setup_layer(input, weight_dim, bias_dim, name):
    with tf.name_scope(name):
        # initial_w = tf.truncated_normal(shape=weight_dim, stddev=0.1, seed=42)
        initial_w = tf.random.truncated_normal(shape=weight_dim, stddev=0.1, seed=42)
        w = tf.Variable(initial_value=initial_w, name='W')

        initial_b = tf.constant(value=0.0, shape=bias_dim)
        b = tf.Variable(initial_value=initial_b, name='B')

        layer_in = tf.matmul(input, w) + b

        if name == 'out':
            layer_out = tf.nn.softmax(layer_in)
        else:
            layer_out = tf.nn.relu(layer_in)

        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)

        return layer_out


def hand_writing_fnt(name):

    X_TRAIN_PATH = 'MNIST/digit_xtrain.csv'
    X_TEST_PATH = 'MNIST/digit_xtest.csv'
    Y_TRAIN_PATH = 'MNIST/digit_ytrain.csv'
    Y_TEST_PATH = 'MNIST/digit_ytest.csv'

    LOGGING_PATH = 'tensorboard_mnist_digit_logs/'

    NR_CLASSES = 10
    VALIDATION_SIZE = 10000
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28
    CHANNELS = 1
    TOTAL_INPUTS = IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS

    y_train_all = np.loadtxt(Y_TRAIN_PATH, delimiter=',', dtype=int)
    x_train_all = np.loadtxt(X_TRAIN_PATH, delimiter=',', dtype=int)

    x_test = np.loadtxt(X_TEST_PATH, delimiter=',', dtype=int)
    y_test = np.loadtxt(Y_TEST_PATH, delimiter=',', dtype=int)

    # Re-scale
    x_train_all, x_test = x_train_all / 255.0, x_test / 255.0

    # convert target values to one-hot encoding
    y_train_all = np.eye(NR_CLASSES)[y_train_all]
    y_test = np.eye(NR_CLASSES)[y_test]

    # create validation dataset from training data

    x_val = x_train_all[:VALIDATION_SIZE]
    y_val = y_train_all[:VALIDATION_SIZE]

    x_train = x_train_all[VALIDATION_SIZE:]
    y_train = y_train_all[VALIDATION_SIZE:]

    # Setup Tensorflow Graph
    # X = tf.placeholder(tf.float32, shape=[None, TOTAL_INPUTS], name='X')
    # Y = tf.placeholder(tf.float32, shape=[None, NR_CLASSES], name='labels')

    X = tf.compat.v1.placeholder(tf.float32, shape=[None, TOTAL_INPUTS])
    Y = tf.compat.v1.placeholder(tf.float32, shape=[None, NR_CLASSES])

    # X = tf.Variable(tf.ones(shape=[None, TOTAL_INPUTS]), dtype=tf.float32, name='X')
    # Y = tf.Variable(tf.ones(shape=[None, NR_CLASSES]), dtype=tf.float32, name='labels')

    nr_epochs = 50
    learning_rate = 1e-3

    n_hidden1 = 512
    n_hidden2 = 64

    layer_1 = setup_layer(X, weight_dim=[TOTAL_INPUTS, n_hidden1], bias_dim=[n_hidden1], name='layer_1')
    # layer_drop = tf.nn.dropout(layer_1, keep_prob=0.8, name='dropout_layer')
    # layer_drop = tf.compat.v1.nn.dropout(layer_1, keep_prob=0.8, name='dropout_layer')

    # rate = 1 - keep_prob
    layer_drop = tf.compat.v1.nn.dropout(layer_1, rate=0.2, name='dropout_layer')

    layer_2 = setup_layer(layer_drop, weight_dim=[n_hidden1, n_hidden2], bias_dim=[n_hidden2], name='layer_2')
    output = setup_layer(layer_2, weight_dim=[n_hidden2, NR_CLASSES], bias_dim=[NR_CLASSES], name='out')

    LR = np.int(learning_rate*1000)
    model_name = f'{n_hidden1}_DO_{n_hidden2}_LR{LR}K_E{nr_epochs}'

    folder_name = f'{model_name}_at_{strftime("%H-%M")}'
    directory = os.path.join(LOGGING_PATH, folder_name)
    # os.path.join(LOGGING_PATH, 'train')
    # Tensorboard Setup
    try:
        os.makedirs(directory)
    except OSError as exception:
        print("1:", exception.strerror)
        print("2:", str(exception))
        print("3: ", exception.args)
    else:
        print('Successfully created directories!')

    # Folder for Tensorboard
    with tf.name_scope('loss_calc'):
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output)) # ok but warnings
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=output))

    # Defining Optimizer
    with tf.name_scope('optimizer'):
        # optimizer = tf.optimizers.Adam(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(loss)

    # Accuracy Metric
    with tf.name_scope('accuracy_calc'):
        correct_pred = tf.equal(tf.argmax(output, axis=1), tf.argmax(Y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.name_scope('performance'):
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('cost', loss)

    with tf.name_scope('show_image'):
        x_image = tf.reshape(X, [-1, 28, 28, 1])
        tf.summary.image('image_input', x_image, max_outputs=4)

    # Run Session
    sess = tf.Session()

    # Setup Filewriter and Merge Summaries
    merged_summary = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(os.path.join(directory, 'train'))
    # train_writer = tf.summary.FileWriter(directory + '/train')
    train_writer.add_graph(sess.graph)

    validation_writer = tf.summary.FileWriter(os.path.join(directory, 'validation'))
    # validation_writer = tf.summary.FileWriter(directory + '/validation')

    # Initialise all the variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # Batching the Data
    size_of_batch = 1000
    global num_examples
    global index_in_epoch

    # global num_examples
    # global index_in_epoch

    num_examples = y_train.shape[0]
    nr_iterations = int(num_examples / size_of_batch)

    index_in_epoch = 0

    for epoch in range(nr_epochs):

        # ============= Training Dataset =========
        for i in range(nr_iterations):
            batch_x, batch_y = next_batch(batch_size=size_of_batch, data=x_train, labels=y_train)

            feed_dictionary = {X: batch_x, Y: batch_y}

            sess.run(train_step, feed_dict=feed_dictionary)

        s, batch_accuracy = sess.run(fetches=[merged_summary, accuracy], feed_dict=feed_dictionary)

        train_writer.add_summary(s, epoch)

        print(f'Epoch {epoch} \t| Training Accuracy = {batch_accuracy}')

        # ================== Validation ======================

        summary = sess.run(fetches=merged_summary, feed_dict={X: x_val, Y: y_val})
        validation_writer.add_summary(summary, epoch)

    print('Done training!')

    # Make a Prediction
    img = Image.open('MNIST/test_img.png')

    # black and white
    bw = img.convert('L')
    img_array = np.invert(bw)
    print('Array.shape', img_array.shape)
    test_img = img_array.ravel()

    print('Flattened Array.shape', test_img.shape)

    prediction = sess.run(fetches=tf.argmax(output, axis=1), feed_dict={X: [test_img]})
    print(f'Prediction for test image is {prediction}')

    # Testing and Evaluation
    test_accuracy = sess.run(fetches=accuracy, feed_dict={X: x_test, Y: y_test})
    print(f'Accuracy on test set is {test_accuracy:0.2%}')

    # Reset for the Next Run
    train_writer.close()
    validation_writer.close()
    sess.close()
    tf.reset_default_graph()



    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    seed(888)
    tf.set_random_seed(404)
    # tf.random.set_seed(404)

    tf.compat.v1.disable_eager_execution()

    hand_writing_fnt('Finished app!')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
