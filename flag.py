# Sung Hah Hwang
# EMCS Laboratories, Korea University
# March 3, 2017

class FLAG():

  def __init__(self):
    self.TRAIN_DIR = None # training file directory
    self.TEST_DIR = None # test file directory
    self.LOG_DIR = None # summary directory
    self.CKPT_DIR = None # checkpoint directory
    self.SEED = 77777 # set to None for random seed
    self.BATCH_SIZE = 100
    self.SUMMARY_FREQ = 1000 # summary-writing frequency
    self.NUM_EPOCH = 1
    self.LEARN_RATE = 1e-3
    self.NUM_THREADS = 8


  def state(self):
    attrs = vars(self)
    print '=' * 35
    print '\n'.join('%12s : %10s' % field for field in sorted(attrs.items(), key=lambda x: (x[1], x[0]), reverse=True))
    print '=' * 35 + '\n'


  def ready(self):
    if self.TRAIN_DIR and self.TEST_DIR and self.LOG_DIR and self.CKPT_DIR is not None:
      return True
    else:
      return False