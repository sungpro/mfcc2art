# mfcc2art

## Tensorflow implementation of speech inversion

### write_TFR.py
- Converts an ASCII file to a `TFRecords` file which follows the standard TensorFlow format containing `tf.train.Example` `protocol buffers` which hold `features` as a field
- Thus written TFRecords were read and fed into the graph as tensors with the use of `tf.TFRecordReader` and `tf.parse_single_example`

### run_reg.py
- Trains a linear regression model
- Writes summaries for tensorboard visualization
- Supports threading, queues, batch training

### run\_reg_3L.py
- Trains a three hidden layer non-linear regression model
- Uses ReLU activation function in hidden layers as default
- Writes summaries for tensorboard visualization
- Supports Threading, queues, batch training

