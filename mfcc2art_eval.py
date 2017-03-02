# Sung Hah Hwang
# EMCS Laboratories, Korea University
# March 3, 2017

import tensorflow as tf
import flag # customizable class
import mfcc2art
import datetime, time
import math

FLAG = flag.FLAG()

FLAG.TRAIN_DIR = '/home/inchon26/tensorflow/exp/TrainSet.TFR'
FLAG.TEST_DIR =  '/home/inchon26/tensorflow/exp/TestSet.TFR'
FLAG.CKPT_DIR = '/home/hdd2tb/sunghah/XRMB/'
FLAG.LOG_DIR = '/home/hdd2tb/sunghah/XRMB/log/'
FLAG.TRAIN_EXAMPLES = 3264252
FLAG.TEST_EXAMPLES= 816063
FLAG.BATCH_SIZE = 1


assert FLAG.ready()

FLAG.state()


def evaluate():
  """Eval XRMB for a number of steps."""

  with tf.Graph().as_default() as g:

    MFCC_TestBatch, ART_TestBatch = xrmb.input_pipeline(
      FLAG.TEST_DIR, FLAG.BATCH_SIZE, FLAG.NUM_EPOCH, FLAG.SEED, FLAG.NUM_THREADS
    )

    # Build a Graph that computes the predictions from the inference model
    output = xrmb.inference(MFCC_TestBatch)

    cost = xrmb.cost(output, ART_TestBatch)

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
      init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
      init_op.run()
      ckpt = tf.train.get_checkpoint_state(FLAG.CKPT_DIR)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        print 'No checkpoint file found' 
        return


    # Build the summary operation based on the TF collection of Summaries.
#     summary_op = tf.summary.merge_all()

#     summary_writer = tf.summary.FileWriter(FLAG.TEST_DIR, g)


      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess = sess, coord = coord)


      try:
        iter = 0
        RMS = 0
        eval_start = time.time()
        while not coord.should_stop():
          print sess.run(cost)
          RMS += sess.run(cost)
          iter += 1

          print sess.run(MFCC_TestBatch)
        
          print sess.run(ART_TestBatch)
        
          print MFCC_TestBatch.get_shape()
        
          print ART_TestBatch.get_shape()
          
          break

      except tf.errors.OutOfRangeError:
        duration = time.time() - eval_start
        print '\n' + '=' * 100
        print 'Done evaluating for {:d} epochs of {:d} iterations (batch size = {:d})'.format(
          FLAG.NUM_EPOCH, iter, FLAG.BATCH_SIZE
        )
        print 'Eval time: {:.2f} sec.'.format(duration)
        print '=' * 100 + '\n'

      except tf.errors.CancelledError:
        print 'Caught an exception: CancelledError'

      finally:
        coord.request_stop()

      coord.join(threads, stop_grace_period_secs = 5)


      RMS = RMS / iter
      print '%s: RMS = %.3f' % (datetime.datetime.now(), RMS)


if __name__ == '__main__':
  evaluate()

