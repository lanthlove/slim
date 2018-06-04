""" densenet"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

def densenet_arg_scope(weight_decay=4e-5):
    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=slim.variance_scaling_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d],padding='SAME') as sc:
            return sc

def composition_layer(net,keep_proba,grow_rate = 12,scope=None):
    with tf.variable_scope(scope,'composition_layer',[net]):
        #batch norm --- relu --- conv3x3 --- drop out
        end_point = scope + 'batch_norm'
        net = slim.batch_norm(net,scope=end_point)
        end_point = scope + 'relu'
        net = tf.nn.relu(net,name=end_point)
        end_point = scope + 'conv_3x3'
        net = slim.conv2d(net, grow_rate, [3, 3], stride=1, scope=end_point,activation_fn=None)
        end_point = scope + 'dropout2'
        net = slim.dropout(net, keep_prob=keep_proba, scope=end_point)
        return net

def bottleneck_layer(net,keep_proba,grow_rate = 12,scope = None):
    with tf.variable_scope(scope,'bottleneck_layer',[net]):
		#batch norm --- relu --- conv1x1 --- drop out
        end_point = scope + 'batch_norm'
        net = slim.batch_norm(net,scope=end_point)
        end_point = scope + 'relu'
        net = tf.nn.relu(net,name=end_point)
        end_point = scope + 'conv_1x1'
        net = slim.conv2d(net, 4*grow_rate, [1, 1], stride=1, scope=end_point,activation_fn=None)
        end_point = scope + 'dropout1'
        net = slim.dropout(net, keep_prob=keep_proba, scope=end_point)
        return net

def transition_layer(net,keep_proba,theta = 0.5,scope = None):
    with tf.variable_scope(scope,'transition_layer',[net]):
        #batch norm --- conv 1x1(theta) --- drop out --- pooling(2x2,stride=2)
        end_point = scope + 'batch_norm'
        net = slim.batch_norm(net,scope=end_point)
        end_point = scope + 'conv1x1_compression'
        net_shape = net.get_shape().as_list()
        output_depth = int(theta*net_shape[-1])
        net = slim.conv2d(net, output_depth, [1, 1], stride=1, scope=end_point)
        end_point = scope + 'dropout'
        net = slim.dropout(net, keep_prob=keep_proba, scope=end_point)
        end_point = scope + 'pooling_layer'
        net = slim.avg_pool2d(net, [2,2], stride=2, scope=end_point, padding='same')
        return net

#input_depth,
#grow_rate=12,
def densenet_block(net,keep_proba,layers = 12,scope=None,use_BC=True):
    #output_depth = input_depth
    with tf.variable_scope(scope,'desenet_block',[net]):
        # put input net to net list
        net_list = list()
        net_list.append(net)
        for i in range(layers):
            
            if use_BC == True:
                #apply bottle neck layer
                end_point = scope + 'bottleneck_layer' + str(i)
                net = bottleneck_layer(net,keep_proba,grow_rate=12,scope=end_point)
            
            #apply composition layer
            end_point = scope + 'composition_layer' + str(i)
            net = composition_layer(net,keep_proba,grow_rate=12,scope=end_point)
            net_list.append(net)
            #output_depth = output_depth + grow_rate
            
            #do merge net list
            end_point = scope + 'layer' +str(i)
            net = tf.concat(values=net_list,axis=3,name=end_point)
        return net

def densenet(inputs,
             num_classes,
             is_training,
             num_block_list,
             keep_proba,
             scope='densenet',
             grow_rate = 12,
             prediction_fn=slim.softmax,
             batch_norm_decay = 0.997,
             batch_norm_epsilon = 0.01,
             use_BC = True
             ):

    with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=is_training):
        with slim.arg_scope([slim.batch_norm],
                            decay=batch_norm_decay,
                            epsilon=batch_norm_epsilon,
                            updates_collections=tf.GraphKeys.UPDATE_OPS,
                            fused=None):
            with tf.variable_scope(scope, 'Densenet-40', [inputs]):
                #define batch norm params
                
                end_points = {}
                if False:
                    end_point = scope + 'input_conv7x7'
                    net = slim.conv2d(inputs,grow_rate*2,[7,7],stride=2,scope=end_point)
                    end_points[end_point] = net

                    end_point = scope + 'maxpool_3x3'
                    net = slim.max_pool2d(net,[3,3],stride=2, padding='same',scope=end_point)
                    end_points[end_point] = net
                else:
                    end_point = scope + 'input_conv3x3'
                    net = slim.conv2d(inputs,grow_rate*2,[3,3],stride=2, padding='same',scope=end_point)
                    end_points[end_point] = net

                for i,len_block in enumerate(num_block_list):
                    # dense block
                    end_point = scope + 'denseblock_' + str(i) + '_'
                    net = densenet_block(net,keep_proba,layers=len_block,scope=end_point,use_BC=use_BC)
                    end_points[end_point] = net
                    
                    #the last dense block needn't to connect a transition layer
                    if i < len(num_block_list)-1:
                    #transition layer
                        end_point = scope + 'transition_' + str(i) + '_'
                        if use_BC == True:
                            net = transition_layer(net,keep_proba,theta=0.5,scope=end_point)
                        else:
                            net = transition_layer(net,keep_proba,theta=1,scope=end_point)
                        end_points[end_point] = net

                end_points['denseblock'] = net

                end_point = scope + 'globalpool'
                net_shape = net.get_shape().as_list()
                net = slim.avg_pool2d(net,[net_shape[1],net_shape[2]],stride=1, padding='valid',scope=end_point)
                end_points['globalpool'] = net

                end_point = scope + 'fc'
                net = tf.contrib.layers.flatten(net)
                logits = slim.fully_connected(net,num_classes,scope=end_point,activation_fn=None)
                end_points['logit'] = net
                predictions = prediction_fn(logits, scope='Predictions')
                end_points['prediction'] = predictions

                return logits,end_points



def densenet_bc100(inputs,
                  num_classes,
                  is_training=True,
                  dropout_keep_prob=0.8,
                  prediction_fn=slim.softmax):
    
    block_list= [16,16,16]
    logits,end_points = densenet(inputs,
                                 num_classes,
                                 is_training,
                                 block_list,
                                 scope='densenet_bc100',
                                 grow_rate=12,
                                 keep_proba = dropout_keep_prob,
                                 prediction_fn=slim.softmax
                                )
    return logits,end_points

def densenet_bc250(inputs,
                 num_classes,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax):
    
    block_list= [41,41,41]
    logits,end_points = densenet(inputs,
                                 num_classes,
                                 is_training,
                                 block_list,
                                 scope='densenet_bc250',
                                 grow_rate=12,
                                 keep_proba = dropout_keep_prob,
                                 prediction_fn=slim.softmax
                                )
    return logits,end_points

def densenet_40(inputs,
               num_classes,
               is_training=True,
               dropout_keep_prob=0.8,
               prediction_fn=slim.softmax):
    
    num_block_list= [8,8,8]
    logits,end_points = densenet(inputs,
                                 num_classes,
                                 is_training,
                                 num_block_list,
                                 scope='densenet_40',
                                 grow_rate=12,
                                 keep_proba = dropout_keep_prob,
                                 prediction_fn=slim.softmax,
                                 )
    return logits,end_points

densenet.default_image_size = 32

densenet_40.default_image_size = densenet.default_image_size
densenet_bc100.default_image_size = densenet.default_image_size
densenet_bc250.default_image_size = densenet.default_image_size