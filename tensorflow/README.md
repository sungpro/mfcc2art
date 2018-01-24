# mfcc2art

## Tensorflow implementation of speech inversion

### ANN5.py
- Up-to-date version (Tensorflow 1.4.0) 
- ANN with 5 hidden layers

### write_TFR.py
- Converts an ASCII file to a `TFRecords` file which follows the standard TensorFlow format containing `tf.train.Example` `protocol buffers` which hold `features` as a field
- Thus written TFRecords were read and fed into the graph as tensors with the use of `tf.TFRecordReader` and `tf.parse_single_example`

### flag.py
- A lightweight global container/accessor for flags and their values

### mfcc2art.py
- Contains functions which builds the mfcc2art network

### mfcc2art_train.py
- Trains a three-hidden-layer neural network
- Uses ReLU activation function in hidden layers as default
- Writes summaries for tensorboard visualization
- Supports threading, queues, batch training

### mfcc2art_eval.py
- Loads and evaluates a trained model

