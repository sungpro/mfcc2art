# Sung Hah Hwang
# sunghahh@gmail.com


import tensorflow as tf
import tensorflow.contrib.rnn


def weight_variable(shape, mean=0.0, stddev=0.1, dtype=tf.float32):
    """
        Create a weight variable
    """
    weight = tf.truncated_normal(shape, mean=mean,
                                        stddev=stddev,
                                        dtype=dtype)
    return tf.Variable(weight)


def bias_variable(shape, dtype=tf.float32):
    """
        Create a bias variable
    """
    bias = tf.constant(0.1, shape=shape, dtype=dtype)
    return tf.Variable(bias)


def variable_summaries(var, name):
    """
        Attach summaries to a variable
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        min_val = tf.reduce_min(var)
        max_val = tf.reduce_max(var)

        tf.summary.histogram('histogram/' + name, var)
        tf.summary.scalar('mean/' + name, mean)
        tf.summary.scalar('stddev/' + name, stddev)
        tf.summary.scalar('min/' + name, min_val)
        tf.summary.scalar('max/' + name, max_val)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """
        Reusable code for building a simple neural net layer
    """
    lname = layer_name
    with tf.name_scope(lname):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights, lname + '/weights')

        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases, lname + '/biases')

        with tf.name_scope('Wx_plus_b'):
            pre_activations = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram(lname + '/pre_activations', pre_activations)

        if act == tf.nn.softmax:
            activations = act(pre_activations, -1, 'activations')
        else:
            activations = act(pre_activations, 'activations')

        tf.summary.histogram(lname + '/activations', activations)
    return activations




class RNN(object):
    pass




def conv1d_layer(input_tensor, in_channels, out_channels,
                 conv_patch, pool_patch=None, layer_name=None,
                 conv_stride=1, pool_stride=3, use_pooling=True):
    """
        Reusable code for building a 1D convolutional neural net layer
    """
    lname = layer_name
    with tf.name_scope(lname):
        with tf.name_scope('weights'):
            weights = weight_variable([conv_patch, in_channels, out_channels])
            variable_summaries(weights, lname + '/weights')

        with tf.name_scope('biases'):
            biases = bias_variable([out_channels])
            variable_summaries(biases, lname + '/biases')

        with tf.name_scope('conv1d'):
            conv_1d = tf.nn.conv1d(input_tensor, weights, stride=conv_stride,
                                                          padding='SAME')
            pre_activations = conv_1d + biases
            tf.summary.histogram(lname + '/pre_activations', pre_activations)

        activations = tf.nn.relu(pre_activations, 'activations')
        tf.summary.histogram(lname + '/activations', activations)

        if use_pooling:
            conv_out = tf.nn.pool(activation, window_shape=[pool_patch],
                                              pooling_type="MAX",
                                              strides=[pool_stride],
                                              padding='SAME')
        else:
            conv_out = activations
        return conv_out


def rnn_layer(input_tensor, timesteps, input_size, output_size, hidden_size,
              act=tf.nn.relu):

    weights = weight_variable([hidden_size, output_size])
    biases = bias_variable([output_size])

    # unstack to get a list of 'TIMESTEPS' tensors of shape: (BATCH_SIZE, INTPUT_SIZE)
    input_tensor = tf.unstack(input_tensor, timesteps, 1)

    # define a LSTM cell
    lstm_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
    # fetch LSTM cell output
    outputs, states = rnn.static_rnn(lstm_cell, input_tensor, dtype=tf.float32)

    # activation using RNN inner loop last output
    rnn_out = act(tf.matmul(outputs[-1], weights) + biases)
    return rnn_out


def bidirectional_lstm_layer(input, output_dim, params,
                             foward_cell_size=128,
                             backward_cell_size=128):

    cell_list = [foward_cell_size, backward_cell_size]

    with tf.name_scope('deep_bidirectional_lstm'):
        # forward direction cells
        fw_c_list = [rnn.BasicLSTMCell(n, forget_bias=1.0) for n in cell_list]
        # backward direction cells
        bw_c_list = [rnn.BasicLSTMCell(n, forget_bias=1.0) for n in cell_list]

        lstm_net, _, _ = rnn.stack_bidirectional_dynamic_rnn(fw_c_list,
                                                             bw_c_list,
                                                             input,
                                                             dtype=tf.float32)

        # dropout layer
        lstm_net = tf.nn.dropout(lstm_net, keep_prob=params.keep_prob)

        with tf.name_scope('reshape_rnn'):
            shape = lstm_net.get_shape().as_list() # [batch, width, 2 * n_hidden]
            rnn_reshaped = tf.reshape(lstm_net, [-1, shape[-1]]) # [batch * width 2 * n_hidden]

        with tf.name_scope('fully_connected_layer'):

            with tf.name_scope('weights'):
                weights = weight_variable([list_n_hidden[-1]*2, output_dim])
                variable_summaries(weights, 'weights')

            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases, 'biases')

            with tf.name_scope('Wx_plus_b'):
                fc_out = tf.matmul(rnn_reshaped, weights) + biases
                tf.summary.histogram('fc_out', fc_out)

        lstm_out = tf.reshape(fc_out, [shape[0], -1, output_dim], name='reshape_out') # [batch, width, output_dim]

        # transpose batch and time axes
        lstm_out = tf. transpose(lstm_out, [1, 0, 2], name='transpose_time_major') # [width(time), batch, output_dim]

    return lstm_out
