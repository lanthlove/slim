""" densenet"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

def densenet_arg_scope(weight_decay=1e-4):
    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=slim.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d],padding='same',activation_fn=None) as sc:
            return sc

def composition_layer(net,keep_proba,grow_rate = 12,scope=None):
    with tf.variable_scope(scope,'composition_layer',[net]):
        #batch norm --- relu --- conv3x3 --- drop out
        end_point = scope + 'bn'
        net = slim.batch_norm(net,scope=end_point)
        end_point = scope + 'relu'
        net = tf.nn.relu(net,name=end_point)
        end_point = scope + 'conv3x3'
        net = slim.conv2d(net, grow_rate, [3, 3], stride=1, scope=end_point,activation_fn=None)
        end_point = scope + 'drop'
        net = slim.dropout(net, keep_prob=keep_proba, scope=end_point)
        return net

def bottleneck_layer(net,keep_proba,grow_rate = 12,scope = None):
    with tf.variable_scope(scope,'bottleneck',[net]):
		#batch norm --- relu --- conv1x1 --- drop out
        end_point = scope + 'bn'
        net = slim.batch_norm(net,scope=end_point)
        end_point = scope + 'relu'
        net = tf.nn.relu(net,name=end_point)
        end_point = scope + 'conv1x1'
        net = slim.conv2d(net, 4*grow_rate, [1, 1], stride=1, scope=end_point,activation_fn=None)
        end_point = scope + 'drop'
        net = slim.dropout(net, keep_prob=keep_proba, scope=end_point)
        return net

def transition_layer(net,keep_proba,theta = 0.5,scope = None):
    with tf.variable_scope(scope,'transition_layer',[net]):
        #batch norm --- conv 1x1(theta) --- drop out --- pooling(2x2,stride=2)
        end_point = scope + 'bn'
        net = slim.batch_norm(net,scope=end_point)
        end_point = scope + 'compress'
        net_shape = net.get_shape().as_list()
        output_depth = int(theta*net_shape[-1])
        net = slim.conv2d(net, output_depth, [1, 1], stride=1, scope=end_point)
        end_point = scope + 'drop'
        net = slim.dropout(net, keep_prob=keep_proba, scope=end_point)
        end_point = scope + 'pool'
        net = slim.avg_pool2d(net, [2,2], stride=2, scope=end_point, padding='same')
        return net

#input_depth,
#grow_rate=12,
def densenet_block(net,keep_proba,grow_rate,layers = 12,scope=None,use_BC=True):
    #output_depth = input_depth
    with tf.variable_scope(scope,'block',[net]):
        # put input net to net list
        net_list = list()
        net_list.append(net)
        for i in range(layers):
            
            if use_BC == True:
                #apply bottle neck layer
                end_point = scope + 'bottleneck' + str(i)
                net = bottleneck_layer(net,keep_proba,grow_rate,scope=end_point)
            
            #apply composition layer
            end_point = scope + 'composition' + str(i)
            net = composition_layer(net,keep_proba,grow_rate,scope=end_point)
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
             batch_norm_epsilon = 1e-5,
             use_BC = True,
             large_mem = False
             ):

    with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=is_training):
        with slim.arg_scope([slim.batch_norm],
                            decay=batch_norm_decay,
                            epsilon=batch_norm_epsilon,
                            updates_collections=tf.GraphKeys.UPDATE_OPS,
                            fused=None):
            with tf.variable_scope(scope, 'densenet', [inputs]):
                #define batch norm params
                
                end_points = {}
                if large_mem == True:
                    max_size = 400
                else:
                    max_size = 270
                
                inputs_shape = inputs.get_shape().as_list()
                if inputs_shape[1] >max_size:
                    end_point = scope + 'conv9x9'
                    net = slim.conv2d(inputs,grow_rate*2,[9,9],stride=3,scope=end_point)
                    end_points[end_point] = net

                    end_point = scope + 'mpool3x3'
                    net = slim.max_pool2d(net,[5,5],stride=2, padding='same',scope=end_point)
                    end_points[end_point] = net
                elif inputs_shape[1] >100:
                    end_point = scope + 'conv7x7'
                    net = slim.conv2d(inputs,grow_rate*2,[7,7],stride=2,scope=end_point)
                    end_points[end_point] = net

                    end_point = scope + 'mpool3x3'
                    net = slim.max_pool2d(net,[3,3],stride=2, padding='same',scope=end_point)
                    end_points[end_point] = net
                else:
                    end_point = scope + 'conv3x3'
                    net = slim.conv2d(inputs,grow_rate*2,[3,3],stride=2, padding='same',scope=end_point)
                    end_points[end_point] = net

                for i,len_block in enumerate(num_block_list):
                    # dense block
                    end_point = scope + 'block_' + str(i)
                    net = densenet_block(net,keep_proba,grow_rate,layers=len_block,scope=end_point,use_BC=use_BC)
                    end_points[end_point] = net
                    
                    #the last dense block needn't to connect a transition layer
                    if i < len(num_block_list)-1:
                    #transition layer
                        end_point = scope + 'tans' + str(i) 
                        if use_BC == True:
                            net = transition_layer(net,keep_proba,theta=0.5,scope=end_point)
                        else:
                            net = transition_layer(net,keep_proba,theta=1,scope=end_point)
                        end_points[end_point] = net

                end_points['denseblock'] = net

                end_point = scope + 'gpool'
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


def densenet_bc88(inputs,
                  num_classes,
                  is_training=True,
                  dropout_keep_prob=0.8,
                  prediction_fn=slim.softmax):
    
    block_list= [14,14,14]
    logits,end_points = densenet(inputs,
                                 num_classes,
                                 is_training,
                                 block_list,
                                 scope='dense_bc88',
                                 grow_rate=12,
                                 keep_proba = dropout_keep_prob,
                                 prediction_fn=slim.softmax
                                )
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
                                 scope='dense_bc100',
                                 grow_rate=12,
                                 keep_proba = dropout_keep_prob,
                                 prediction_fn=slim.softmax
                                )
    return logits,end_points


def densenet_bc124(inputs,
                  num_classes,
                  is_training=True,
                  dropout_keep_prob=0.8,
                  large_mem = True,
                  prediction_fn=slim.softmax):
    
    block_list= [20,20,20]
    logits,end_points = densenet(inputs,
                                 num_classes,
                                 is_training,
                                 block_list,
                                 scope='dense_bc100',
                                 grow_rate=12,
                                 large_mem = large_mem,
                                 keep_proba = dropout_keep_prob,
                                 prediction_fn=slim.softmax
                                )
    return logits,end_points


def densenet_bc250(inputs,
                 num_classes,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 large_mem = True,
                 prediction_fn=slim.softmax):
    
    block_list= [41,41,41]
    logits,end_points = densenet(inputs,
                                 num_classes,
                                 is_training,
                                 block_list,
                                 scope='dense_bc250',
                                 grow_rate=12,
                                 large_mem = large_mem,
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
                                 scope='dense_40',
                                 grow_rate=12,
                                 keep_proba = dropout_keep_prob,
                                 prediction_fn=slim.softmax,
                                 )
    return logits,end_points

densenet.default_image_size = 320

densenet_40.default_image_size = densenet.default_image_size
densenet_bc100.default_image_size = densenet.default_image_size
densenet_bc124.default_image_size = densenet.default_image_size
densenet_bc250.default_image_size = densenet.default_image_size