# Sung Hah Hwang
# EMCS Laboratories, Korea University
# Jan 23, 2017

'''Convert ASCII data to TFRecords'''

import numpy as np
import tensorflow as tf


def write_TFR(input, output, fname):
    input_size = input.shape[0]
    output_size = output.shape[0]

    if input_size != output_size:
        raise ValueError('Input and output sizes of set do not match [{}] <-> [{}]'.format(input_size, output_size))

    writer = tf.python_io.TFRecordWriter(fname)

    # iterate over each example & construct the Example proto object
    for idx in range(input_size):
        x = input[idx]
        y = output[idx]

        # Example contains a Features proto object; Features contain a map of string to Feature proto objects
        example = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    'MFCC': tf.train.Feature( float_list = tf.train.FloatList(value = x) ),
                    'ART': tf.train.Feature( float_list = tf.train.FloatList(value = y) )
                }
            )
        )
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        writer.write(serialized)



# Specify filenames
x_train = np.loadtxt('MFCC.train', dtype = 'float')
y_train = np.loadtxt('ART.train', dtype = 'float')
x_test = np.loadtxt('MFCC.test', dtype = 'float')
y_test = np.loadtxt('ART.test', dtype = 'float')

write_TFR(x_train, y_train, 'TrainSet.TFR')
write_TFR(x_test, y_test, 'TestSet.TFR')