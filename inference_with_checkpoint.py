import tensorflow as tf
from datasets import dataset_factory
from preprocessing import preprocessing_factory
from nets import nets_factory
import numpy as np

tf.app.flags.DEFINE_string('dataset_name','cifar10','')
tf.app.flags.DEFINE_string('dataset_dir','D:\Documents\AI\Database\cifar10','')
tf.app.flags.DEFINE_string('model_name','cifarnet','')
tf.app.flags.DEFINE_string('checkpoint_path','D:\Documents\AI\CSDN\Week08\Code\models','')
tf.app.flags.DEFINE_string('pic_path','timg.jpg','')

FLAGS = tf.app.flags.FLAGS

is_training = False
preprocessing_name = FLAGS.model_name

graph = tf.Graph().as_default()

dataset = dataset_factory.get_dataset(FLAGS.dataset_name,'train',FLAGS.dataset_dir)

imge_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name,is_training=False)
network_fn = nets_factory.get_network_fn(FLAGS.model_name,num_classes=(dataset.num_classes),is_training=is_training)

if hasattr(network_fn,'default_image_size'):
    image_size = network_fn.default_image_size
else:
    image_size = FLAGS.default_image_size

palceholder = tf.placeholder(name='input',dtype=tf.string)
image = tf.image.decode_jpeg(palceholder,channels=3)
image = imge_preprocessing_fn(image,image_size,image_size)
image = tf.expand_dims(image,0)
logit,end_points = network_fn(image)

sess_config = tf.ConfigProto(allow_soft_placement=True)
saver = tf.train.Saver()
sess = tf.Session(config=sess_config)
checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
saver.restore(sess,checkpoint_path)
image_value = open(FLAGS.pic_path,'rb').read()
logit_value = sess.run([logit],feed_dict={palceholder:image_value})
print(logit_value)
print(np.argmax(logit_value))
