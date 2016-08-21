import numpy as np
import os
import logging
import re
import string
import bitarray
import time
from multiprocessing import Process, Queue
#from Queue import Queue

from common_utils import CommonUtiler

logger = logging.getLogger('TfDataProvider')
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)4s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


class Batch(object):
  def __init__(self, batch_size, max_seq_len, vf_size, bos_ind):
    self.batch_size = batch_size
    self.max_seq_len = max_seq_len
    self.vf_size = vf_size
    self.bos_ind = bos_ind
    self.empty()
      
  def empty(self):
    self.x = np.zeros([self.batch_size, self.max_seq_len], dtype=np.int32)
    self.y = np.zeros([self.batch_size, self.max_seq_len], dtype=np.int32)
    self.vf = np.zeros([self.batch_size, self.vf_size], dtype=np.float32)
    self.fg = np.zeros([self.batch_size, self.max_seq_len], dtype=np.float32)
    self.sl = np.zeros([self.batch_size], dtype=np.int32)
    self.num_feed = 0
      
  def feed_and_vomit(self, visual_features, sentence):
    i = self.num_feed
    # feed sentence
    self.x[i, 0] = self.bos_ind
    if len(sentence) > self.max_seq_len - 1:
      self.x[i, 1:] = sentence[:self.max_seq_len-1]
      self.y[i, :self.max_seq_len-1] = sentence[:self.max_seq_len-1]
      self.y[i, self.max_seq_len-1] = self.bos_ind
      self.fg[i, :] = np.ones([self.max_seq_len], dtype=np.float32)
      self.sl[i] = self.max_seq_len
    else:
      l = len(sentence)
      self.x[i, 1:l+1] = sentence
      self.y[i, :l] = sentence
      self.y[i, l] = self.bos_ind
      self.fg[i, :l+1] = np.ones([l+1], dtype=np.float32)
      self.sl[i] = l + 1
    # feed visual feature
    assert visual_features.shape[0] == self.vf_size
    self.vf[i, :] = visual_features
    self.num_feed += 1
    assert self.num_feed <= self.batch_size
    # vomit if necessary
    if self.num_feed == self.batch_size:
      return (self.x, self.y, self.vf, self.fg, self.sl)
    return None


class mRNNCocoBucketDataProvider(object):
  """mRNN TensorFlow Data Provider with Buckets on MS COCO."""
  def __init__(self, anno_files_path, vocab_path, vocab_size, vf_dir, vf_size,
      flag_shuffle=True):
    self.cu = CommonUtiler()
    self.anno_files_path = anno_files_path
    self.vocab_path = vocab_path
    self.vocab, _ = self.cu.load_vocabulary(vocab_path)
    assert len(self.vocab) == vocab_size
    assert self.vocab['<pad>'] == 0
    self.vf_dir = vf_dir
    self.vf_size = vf_size
    self.flag_shuffle = flag_shuffle
      
  def generate_batches(self, batch_size, buckets):
    """Return a list generator of mini-batches of training data."""
    # create Batches
    batches = []
    for max_seq_len in buckets:
      batches.append(
          Batch(batch_size, max_seq_len, self.vf_size, self.vocab['<bos>']))

    # Create prefetch process
    self._data_queue = Queue(10)
    self._prefetch_process = TaskAllocator(self._data_queue, self, flag_shuffle = self.flag_shuffle)
    self._prefetch_process.start()
    def cleanup():
      print 'Terminating DataFetcher'
      self._prefetch_process.terminate()
      self._prefetch_process.join()
    import atexit
    atexit.register(cleanup)
    # shuffle if necessary
    #if self.flag_shuffle:
    #  np.random.shuffle(self._data_pointer)
    # scan data queue
    while True:
      data = self._data_queue.get()
      if data == None:
        break
      for ind_s in range(len(data['sentences'])):
        sentence = data['sentences'][ind_s]
        visual_features = data['visual_features']
        if len(sentence) >= buckets[-1]:
          feed_res = batches[-1].feed_and_vomit(visual_features, sentence)
          ind_buc = len(buckets) - 1
        else:
          for (ind_b, batch) in enumerate(batches):
            if len(sentence) < batch.max_seq_len:
              feed_res = batches[ind_b].feed_and_vomit(visual_features, sentence)
              ind_buc = ind_b
              break
        if feed_res:
          yield (ind_buc,) + feed_res
          batches[ind_buc].empty()

    print('End of a epoch')
    logger.info('End of a epoch')

    self._prefetch_process.terminate()
    self._prefetch_process.join()
    del self._prefetch_process
    del self._data_queue

class TaskAllocator(Process):
  def __init__(self, data_queue, provider, n_threads=8, flag_shuffle=True):

    super(TaskAllocator, self).__init__()
    self._task_queue = Queue(10)
    self._data_queue = data_queue
    self.flag_shuffle = flag_shuffle
    self.provider = provider
    self.anno_files_path = provider.anno_files_path
    self.n_threads = n_threads
    self.threads = []

    def cleanup():
      print 'Terminating DataFetchers'
      for t in self.threads:
        t.terminate()
        t.join()
    import atexit
    atexit.register(cleanup)

  def run(self):
    self.threads = []
    for i in range(self.n_threads):
      self.threads.append(DataFetcher(self._task_queue, self._data_queue, self.provider))
      self.threads[i].start()

    for anno_file_path in self.anno_files_path:
      
      annos = np.load(anno_file_path).tolist()
      if self.flag_shuffle:
        perm = np.random.permutation(np.arange(len(annos)))
      else:
        perm = np.arange(len(annos))
      for ind in range(len(perm)):
        ind_a = perm[ind]
        anno = annos[ind_a]
      #for (ind_a, anno) in enumerate(annos):
        self._task_queue.put(anno)

    for i in range(len(self.threads)*2):
      self._task_queue.put(None)

    for t in self.threads:
      t.join()

    self._data_queue.put(None)

class DataFetcher(Process):
    """Experimental class for prefetching data in a separate process."""
    def __init__(self, task_queue, data_queue, provider):

        super(DataFetcher, self).__init__()
        self._task_queue = task_queue
        self._data_queue = data_queue
        self.vf_dir = provider.vf_dir
        self.cu = provider.cu
        self.vocab = provider.vocab

    def run(self):
        print 'DataFetcher started'
        logger.info('Loading data')
        vocab = self.vocab
        while True:
          anno = self._task_queue.get()
          if anno == None:
            break

          data = {}
          # Load visual features
          feat_path = os.path.join(self.vf_dir, anno['file_path'],
              anno['file_name'].split('.')[0] + '.txt')
          if os.path.exists(feat_path):
            vf = np.loadtxt(feat_path)
          else:
            continue
          data['visual_features'] = vf
          # Encode sentences
          data['sentences'] = []
          for (ind_s, sentence) in enumerate(anno['sentences']):
            sentence_encode = self.cu.encode_sentence(sentence, vocab, 
                flag_add_bos=False)
            data['sentences'].append(np.array(sentence_encode))
              
          self._data_queue.put(data)

