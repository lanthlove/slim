import os
from six.moves import xrange
filenames = [os.path.join('.\data','data_batch_%d' % i) for i in xrange(1, 6)]
print(filenames)