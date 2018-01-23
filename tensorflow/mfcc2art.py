# Sung Hah Hwang
# EMCS Laboratories, Korea University
# March 3, 2017

import tensorflow as tf
import flag # customizable class
import time

FLAG = flag.FLAG()

FLAG.TRAIN_DIR = '/home/inchon26/tensorflow/exp/TrainSet.TFR'
FLAG.TEST_DIR = '/home/inchon26/tensorflow/exp/TestSet.TFR'
FLAG.CKPT_DIR = '/home/inchon26/tensorflow/exp/ckpt'
FLAG.LOG_DIR = '/home/inchon26/tensorflow/exp/log'
FLAG.TRAIN_EXAMPLES = 3264252
FLAG.TEST_EXAMPLES= 816063
FLAG.BATCH_SIZE = 100
FLAG.SUMMARY_FREQ = 1000 # summary-writing frequency
FLAG.NUM_THREADS = 8

# raise error if DIR's are unset
assert FLAG.ready()

# print state
FLAG.state()


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


def input_pipeline(filenames, batch_size, num_epochs, seed = None, num_threads = 4):
  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer([filenames], num_epochs = num_epochs, name = 'file_shuffling')
#     dequeue_many = filename_queue.dequeue_many(len(filenames) * num_epochs, name = 'dequeue_many')
#     dequeue = filename_queue.dequeue()
    MFCC, ART = read_and_decode(filename_queue)
    min_after_dequeue = 1000
    capacity = batch_size * 3 + min_after_dequeue
    MFCC_Batch, ART_Batch = tf.train.shuffle_batch(
      [MFCC, ART], batch_size = batch_size, seed = seed, num_threads = num_threads,
      capacity = capacity, min_after_dequeue = min_after_dequeue,
      allow_smaller_final_batch = True, name = 'batch_shuffling'
    )
    return MFCC_Batch, ART_Batch


def weight_variable(shape):
  """Create a weight variable with appropriate initialization."""
  initial = tf.get_variable('weights', shape, initializer = tf.truncated_normal_initializer(stddev = 0.1))
  return initial


def bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.get_variable('biases', shape, initializer = tf.constant_initializer(0.1))
  return initial


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act = tf.nn.relu):
  with tf.name_scope(layer_name) as scope:
    weights = weight_variable([input_dim, output_dim])
    biases = bias_variable([output_dim])
    variable_summaries(weights)
    variable_summaries(biases)
#     tf.add_to_collection(layer_name, weights)
#     tf.add_to_collection(layer_name, biases)
    with tf.name_scope('Wx_plus_b'):
      pre_activations = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', pre_activations)
    activations = act(pre_activations, name = 'activation')
    tf.summary.histogram('activations', activations)
  return activations


def inference(input):
  ''' build graph '''
  with tf.variable_scope('hid_layer1') as scope:
    hidden1 = nn_layer(input, 39, 100, 'hid_layer1')
  with tf.variable_scope('hid_layer2') as scope:
    hidden2 = nn_layer(hidden1, 100, 100, 'hid_layer2')
  with tf.variable_scope('hid_layer3') as scope:
    hidden3 = nn_layer(hidden2, 100, 100, 'hid_layer3')
  with tf.variable_scope('out_layer') as scope:
    output = nn_layer(hidden3, 100, 16, 'out_layer', act = tf.identity)
  return output


def cost(output, answer):
  with tf.name_scope('cost'):
    cost = tf.sqrt(tf.reduce_mean(tf.square(output - answer)))
    tf.summary.scalar('cost', cost)
  return cost


def train(cost, eta = 1e-6):
  with tf.name_scope('train'):
    eta = tf.Variable(eta)
    train_op = tf.train.AdamOptimizer(eta).minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(eta)
    tf.summary.scalar('l-rate', eta)
  return train_op



