# Sung Hah Hwang
# EMCS Laboratories, Korea University
# March 3, 2017

import tensorflow as tf
import flag
import mfcc2art
import time

FLAG = flag.FLAG()

FLAG.TRAIN_DIR = '/home/inchon26/tensorflow/exp/TrainSet.TFR'
FLAG.TEST_DIR =  '/home/inchon26/tensorflow/exp/TestSet.TFR'
FLAG.CKPT_DIR = '/home/hdd2tb/sunghah/XRMB/'
FLAG.LOG_DIR = '/home/hdd2tb/sunghah/XRMB/log/'
FLAG.NUM_EPOCH = 1
FLAG.BATCH_SIZE = 100
FLAG.SUMMARY_FREQ = 100 # summary-writing frequency
FLAG.NUM_THREADS = 8

# raise error if DIR's are unset
assert FLAG.ready()

# print state
FLAG.state()


def train():
  with tf.Graph().as_default():

    MFCC_TrainBatch, ART_TrainBatch = mfcc2art.input_pipeline(
      FLAG.TRAIN_DIR, FLAG.BATCH_SIZE, FLAG.NUM_EPOCH, FLAG.SEED, FLAG.NUM_THREADS
    )

    # inference model
    output = mfcc2art.inference(MFCC_TrainBatch)

    # calculate loss
    cost = mfcc2art.cost(output, ART_TrainBatch)

    # build a graph that trains the model
    train_op = mfcc2art.train(cost, FLAG.LEARN_RATE)

    sess = tf.InteractiveSession()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    init_op.run()

    saver = tf.train.Saver(max_to_keep=10)

    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAG.LOG_DIR, sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

    try:
      iter = 1
      train_start = time.time()
      disp_start = train_start
      while not coord.should_stop():
        summary, _, cost_val = sess.run([merged_summary, train_op, cost])

        if iter % FLAG.SUMMARY_FREQ == 0:
          duration = time.time() - disp_start
          print 'Iter: {:d}  Loss: {:.3f} (MSE)  Elapsed time: {:.3f}'.format(iter, cost_val, duration)
          save_path = saver.save(sess, FLAG.CKPT_DIR + 'tmp', global_step = iter)
          disp_start = time.time()

        train_writer.add_summary(summary, iter)
        iter += 1
        
        

        
        

    except tf.errors.OutOfRangeError:
      duration = time.time() - train_start
      print '\n' + '=' * 100
      print 'Done training for {:d} epochs of {:d} iterations (batch size = {:d})'.format(
        FLAG.NUM_EPOCH, iter, FLAG.BATCH_SIZE
      )
      print 'Training time: {:.2f} sec.'.format(duration)
      print '=' * 100 + '\n'

    except tf.errors.CancelledError:
      print 'Caught an exception: CancelledError'

    finally:
      coord.request_stop()

    coord.join(threads, stop_grace_period_secs = 5)

    train_writer.close()


if __name__ == '__main__':
  train()
