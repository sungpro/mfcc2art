# Sung Hah Hwang
# sunghahh@gmail.com
# Jan. 18, 2018

import os
import time
import numpy as np
import tensorflow as tf

from models import *


# print logging messages to stdout
tf.logging.set_verbosity(tf.logging.INFO)

PREFIX = '/home/innocompass/speech_inversion/'
DATA_DIR = os.path.join(PREFIX, 'data')
TRAIN_FILE = '/home/innocompass/speech_inversion/MFCC_ART.TFR'
LOG_DIR = os.path.join(PREFIX, 'log')
CKPT_DIR = os.path.join(PREFIX, 'ckpt')
RESULT_DIR = os.path.join(PREFIX, 'result')

INPUT_SIZE = 39
OUTPUT_SIZE = 16
NUM_EPOCH = 100 # number of training epochs
BATCH_SIZE = 128
LEARN_RATE = 0.001 # learning rate
RANDOM_SEED = 55555 # make shuffling & sampling deterministic

SUMMARY_FREQ_TRAIN = 2000 # summary-writing frequency while training
SUMMARY_FREQ_TEST = 10 # summary-writing frequency while testing

np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)


########################################
# Input pipeline
########################################


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
      features = {
        'MFCC': tf.FixedLenFeature([39], tf.float32),
        'ART': tf.FixedLenFeature([16], tf.float32)
      }
    )
  MFCC = features['MFCC']
  ART = features['ART']
  return MFCC, ART



def input_pipeline(filenames, batch_size, num_epochs):
  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer([filenames], num_epochs = num_epochs, name = 'shuffle_files')
#     dequeue_many = filename_queue.dequeue_many(len(filenames) * num_epochs, name = 'dequeue_many')
#     dequeue = filename_queue.dequeue()
    MFCC, ART = read_and_decode(filename_queue)
    min_after_dequeue = 1000
    capacity = batch_size * 3 + min_after_dequeue
    MFCC_Batch, ART_Batch = tf.train.shuffle_batch(
      [MFCC, ART], batch_size = batch_size, seed = 777, num_threads = 4,
      capacity = capacity, min_after_dequeue = min_after_dequeue
    )
    return MFCC_Batch, ART_Batch


########################################
# TRAINING
########################################

def train(TRAIN_FILE, INPUT_SIZE, OUTPUT_SIZE,
          RESULT_DIR, LOG_DIR, CKPT_DIR):

    # reset graph
    tf.reset_default_graph()

    # define a list of CSV files
    train_x_batch, train_y_batch = input_pipeline(TRAIN_FILE, BATCH_SIZE, NUM_EPOCH)


    # build model
    keep_prob = tf.Variable(tf.constant(0.8))

    hidden1 = nn_layer(train_x_batch, INPUT_SIZE, 300, 'layer1')
    hidden2 = nn_layer(hidden1, 300, 500, 'layer2')
    hidden3 = nn_layer(hidden2, 500, 500, 'layer3')
    hidden4 = nn_layer(hidden3, 500, 300, 'layer4')
    dropped1 = tf.nn.dropout(hidden4, keep_prob)

    hidden5 = nn_layer(dropped1, 300, 100, 'layer5')
    dropped2 = tf.nn.dropout(hidden5, keep_prob)

    pred = nn_layer(dropped2, 100, OUTPUT_SIZE, 'out_layer', act = tf.identity)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(pred - train_y_batch))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('ADAM_optimizer'):
        eta = tf.Variable(LEARN_RATE, dtype=tf.float32)
        optmzr = tf.train.AdamOptimizer(eta)
        tf.summary.scalar('l_rate', eta)
  
    train_op = optmzr.minimize(loss)

    # define log writer
    merged_summary = tf.summary.merge_all()
    sess_train = tf.InteractiveSession()
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess_train.graph)
    saver = tf.train.Saver(max_to_keep=3)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    init_op.run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess_train, coord=coord)

    iter = 1
    train_start = time.time()
    batch_start = train_start

    try:
        while not coord.should_stop():
            _, summary, cost = sess_train.run([train_op, merged_summary, loss])

            if iter % SUMMARY_FREQ_TRAIN == 0:
                duration = time.time() - batch_start
                print('Iter: {:9d}    Cost: {:7.4f}    Elapsed: {:7.4f}'.format(
                    iter, cost, duration))
                save_path = saver.save(sess_train, os.path.join(CKPT_DIR, 'tmp'))
                batch_start = time.time()

            train_writer.add_summary(summary, iter)
            iter += 1

    except tf.errors.OutOfRangeError:
        print('\nTraining Complete - Reached the end of the final epoch\n')

    finally:
        # when complete, ask the threads to stop
        coord.request_stop()

    # Wait for threads to finish
    coord.join(threads, stop_grace_period_secs = 5)
    sess_train.close()


def perfs(message):
    print('\n' +
          '=' * 60 + \
          '\n' + str(message) + '\n' + \
          '=' * 60 + \
          '\n')


def main():
    # train
    train(TRAIN_FILE, INPUT_SIZE, OUTPUT_SIZE,
          RESULT_DIR=RESULT_DIR,
          LOG_DIR=LOG_DIR,
          CKPT_DIR=CKPT_DIR)



if __name__ == '__main__':
    main()
