# -*- coding: utf-8 -*-
"""
@author: kk
"""

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope
from skimage import io, transform
import glob
import os
import numpy as np
import time


def Relu(x):
    return tf.nn.relu(x)

def conv_layer(input, filter, kernel, stride=1, padding='SAME', layer_name="conv", activation=True):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        if activation :
            network = Relu(network)
        return network
    
def Sigmoid(x):
    return tf.nn.sigmoid(x)

def Max_pooling(x, pool_size=[3,3], stride=2, padding='VALID') :
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Dropout(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def attention_1(inputs,layer_name):
    m = inputs.shape[1]
    n = inputs.shape[2]
    w = inputs.shape[3]
    with tf.name_scope(layer_name):
        layer_a = tf.layers.average_pooling2d(inputs,[3,3],strides=2,padding='Valid',name="avg_pool2d_1")
        layer_b = tf.layers.average_pooling2d(layer_a,[3,3],strides=2,padding='Valid',name="avg_pool2d_2")
        layer_c = conv_layer(layer_b,w,[1,1],stride=1,padding='SAME',activation=True,layer_name='conv2d_1')
        layer_d = conv_layer(layer_c,w,[1,1],stride=1,padding='SAME',activation=False,layer_name='conv2d_2')
        layer_d = Sigmoid(layer_d)
        
        layer_e = tf.image.resize_bilinear(layer_d, size=[m,n])
        
        layer_f = tf.multiply(layer_e, inputs, name="fuse_mul")
        
        layer_g = tf.add(layer_f, inputs, name="fuse_add")
        
        return layer_g
    
def attention_2(inputs,layer_name):
    m = inputs.shape[1]
    n = inputs.shape[2]
    w = inputs.shape[3]
    with tf.name_scope(layer_name):
        layer_a = tf.layers.average_pooling2d(inputs,[3,3],strides=2,padding='Valid',name="avg_pool2d_2_1")
        layer_b = conv_layer(layer_a,w,[1,1],stride=1,padding='SAME',activation=True,layer_name='conv2d_2_1')
        layer_c = conv_layer(layer_b,w,[1,1],stride=1,padding='SAME',activation=False,layer_name='conv2d_2_2')
        layer_d = Sigmoid(layer_c)
               
        layer_e = tf.image.resize_bilinear(layer_d, size=[m,n])
        
        layer_f = tf.multiply(layer_e, inputs, name="fuse_mul_2")
        
        layer_g = tf.add(layer_f, inputs, name="fuse_add_2")
        
        return layer_g


class Inception_attention():
    def __init__(self, x, training):
        self.training = training
        self.model = self.Build_Inception_attention(x)

    def Stem(self, x, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=16, kernel=[7,7], stride=2, padding='VALID', layer_name=scope+'_conv1')
            x = Max_pooling(x)            
            x = conv_layer(x, filter=32, kernel=[3,3], stride=1, padding='SAME', layer_name=scope+'_conv2')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            #37
            return x
        
    def Inception_A(self, x, scope):
        with tf.name_scope(scope):
            
            split_conv_x1 = conv_layer(x, filter=8, kernel=[1,1], layer_name=scope+'_split_conv1')
            split_conv_x1 = conv_layer(split_conv_x1, filter=32, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv1_1')

            split_conv_x2 = conv_layer(x, filter=8, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=32, kernel=[5,5], layer_name=scope+'_split_conv2_1')
            split_conv_x2 = conv_layer(split_conv_x2, filter=32, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv2_2')

            split_conv_x3 = conv_layer(x, filter=8, kernel=[1,1], layer_name=scope+'_split_conv3')
            split_conv_x3 = conv_layer(split_conv_x3, filter=32, kernel=[7,7], layer_name=scope+'_split_conv3_1')
            split_conv_x3 = conv_layer(split_conv_x3, filter=32, kernel=[3,3], stride=2, padding='VALID',layer_name=scope+'_split_conv3_2')
            
            split_avgpool = tf.layers.average_pooling2d(x, [3,3], strides=2, padding='valid',name=scope+'split_avgpool')
            split_avgpool = conv_layer(split_avgpool, filter=32, kernel=[1,1], layer_name=scope+'_split_conv3')
            
            net = tf.concat([split_conv_x1, split_conv_x2, split_conv_x3, split_avgpool], 3)
            net = Batch_Normalization(net, training=self.training, scope=scope+'_batch1')
            
            net = attention_1(net,layer_name="attention_a")
            #36
            return net
        
    def Inception_B(self, x, scope):
        with tf.name_scope(scope):
            
            split_conv_x1 = conv_layer(x, filter=64, kernel=[1,1], layer_name=scope+'_split_conv1')
            split_conv_x1 = conv_layer(split_conv_x1, filter=128, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv1_1')

            split_conv_x2 = conv_layer(x, filter=64, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=64, kernel=[5,5], layer_name=scope+'_split_conv2_1')
            split_conv_x2 = conv_layer(split_conv_x2, filter=128, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv2_2')

            split_conv_x3 = conv_layer(x, filter=64, kernel=[1,1], layer_name=scope+'_split_conv3')
            split_conv_x3 = conv_layer(split_conv_x3, filter=64, kernel=[7,7], layer_name=scope+'_split_conv3_1')
            split_conv_x3 = conv_layer(split_conv_x3, filter=128, kernel=[3,3], stride=2, padding='VALID',layer_name=scope+'_split_conv3_2')
            
            split_avgpool = tf.layers.average_pooling2d(x, [3,3], strides=2, padding='valid',name=scope+'split_avgpool')
            split_avgpool = conv_layer(split_avgpool, filter=128, kernel=[1,1], layer_name=scope+'_split_conv3')
            
            net = tf.concat([split_conv_x1, split_conv_x2, split_conv_x3, split_avgpool], 3)
            net = Batch_Normalization(net, training=self.training, scope=scope+'_batch2')
            
            net = attention_2(net,layer_name="attention_b")
            #17
            return net
        
    def Inception_C(self, x, scope):
        with tf.name_scope(scope):
            
            split_conv_x1 = conv_layer(x, filter=128, kernel=[1,1], layer_name=scope+'_split_conv1')
            split_conv_x1 = conv_layer(split_conv_x1, filter=256, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv1_1')

            split_conv_x2 = conv_layer(x, filter=128, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=128, kernel=[5,5], layer_name=scope+'_split_conv2_1')
            split_conv_x2 = conv_layer(split_conv_x2, filter=256, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv2_2')

            split_conv_x3 = conv_layer(x, filter=128, kernel=[1,1], layer_name=scope+'_split_conv3')
            split_conv_x3 = conv_layer(split_conv_x3, filter=128, kernel=[7,7], layer_name=scope+'_split_conv3_1')
            split_conv_x3 = conv_layer(split_conv_x3, filter=256, kernel=[3,3], stride=2, padding='VALID',layer_name=scope+'_split_conv3_2')
            
            split_avgpool = tf.layers.average_pooling2d(x, [3,3], strides=2, padding='valid',name=scope+'split_avgpool')
            split_avgpool = conv_layer(split_avgpool, filter=256, kernel=[1,1], layer_name=scope+'_split_conv3')
            
            net = tf.concat([split_conv_x1, split_conv_x2, split_conv_x3, split_avgpool], 3)
            #8
            return net
            
    def Build_Inception_attention(self, input_x):
        
        x = self.Stem(input_x, scope='stem') #
        
        x_a = self.Inception_A(x,scope='Inception_a')
        x_b = self.Inception_B(x_a,scope='Inception_b')
        x_c = self.Inception_C(x_b,scope='Inception_c')
        
        aux_logits = x_b
        aux_logits = conv_layer(aux_logits, filter=512, stride=2, kernel=[3,3],padding='valid',layer_name='aux_conv_1')
        aux_logits = tf.layers.average_pooling2d(aux_logits,[8,8],strides=1,padding='valid',name='aux_avgpool')
        aux_logits = conv_layer(aux_logits,filter=4,kernel=[1,1],activation=False,layer_name='aux_conv_2')
        aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze_1')
        
        
        logits = x_c
        logits = tf.layers.average_pooling2d(logits,[8,8],strides=1,padding='valid',name='aux_avgpool')
        logits = Dropout(logits, rate=0.5, training=self.training)
        logits = conv_layer(logits,filter=2,kernel=[1,1],activation=False,layer_name='logits_conv_1')
        logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze_2')
        
        prediction = tf.nn.softmax(logits, name='Predictions')
        
        return aux_logits,logits,prediction
        


w = 299
h = 299
c = 3

weight_decay = 0.0005
momentum = 0.9

x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')


training_flag = tf.placeholder(tf.bool,name='training_flag')


aux_logits,logits,prediction = Inception_attention(x,training_flag).model

loss1 = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=aux_logits)
loss2 = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

cost = 0.35 * loss1 + loss2 
optimizer = tf.train.MomentumOptimizer(learning_rate=0.00001, momentum=momentum, use_nesterov=True)
train = optimizer.minimize(cost + l2_loss * weight_decay)

correct_prediction = tf.equal(tf.cast(tf.argmax(prediction, 1), tf.int32), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]  # 列表生成式
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpeg'):
            print('reading the images:%s' % (im))
            img = io.imread(im)
            img = transform.resize(img, (w, h, c))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


path = 'D:/deeplearningwork/changjing/'
data, label = read_img(path)


num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]


ratio = 0.8
s = np.int(num_example * ratio)
x_train = data[:s]
y_train = label[:s]
x_val = data[s:]
y_val = label[s:]


def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


total_epochs = 3000
batch_size = 32


sess = tf.InteractiveSession()
saver = tf.train.Saver(tf.global_variables())


    
ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

summary_writer = tf.summary.FileWriter('./logs', sess.graph)

for epoch in range(1, total_epochs + 1):

    start_time = time.time()
    sum_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _, err, ac = sess.run([train, cost, accuracy], feed_dict={x: x_train_a, y_: y_train_a,training_flag: True})
        sum_loss += err;
        train_acc += ac;
        n_batch += 1
            
    train_loss = sum_loss / n_batch
    train_acc = train_acc / n_batch

    print("   sum loss: %f" % (train_loss))
    print("   sum acc: %f" % (train_acc))
        
    train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                         tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])
   
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([cost, accuracy], feed_dict={x: x_val_a, y_: y_val_a,training_flag: False})
        val_loss += err;
        val_acc += ac;
        n_batch += 1
            
    test_loss = val_loss / n_batch
    test_acc = val_acc / n_batch
            
    print("   validation loss: %f" % (test_loss))
    print("   validation acc: %f" % (test_acc))
        
    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])
    
    summary_writer.flush()

    if val_acc > 0.994:
        saver.save(sess=sess, save_path='./model/UWNet.ckpt')

    print(epoch)
    end_time = time.time()
    print(end_time-start_time)
sess.close()