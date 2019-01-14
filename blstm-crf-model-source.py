# coding: utf-8

# In[1]:

import numpy as np
import math
import tensorflow as tf
from utils import read_pos_tagging_file
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import smtplib

# In[2]:

MODEL_PATH = 'model_bi_lstm_crf_laptops_220k_50_75'


# In[3]:

json_file_path = 'data/amazon_laptops_50_75_220k'
x_train, y_train, _, sequence_length_train = read_pos_tagging_file(json_file_path)

# In[4]:

def pad(sentence, max_length, is_label, input_size=0):
    pad_len = max_length - len(sentence)
    padding = np.zeros(pad_len) if is_label == True else np.zeros((pad_len, input_size))
    return np.concatenate((sentence, padding))

def pad_5_gram(sentence, max_length, input_size=0):
    pad_len = max_length - len(sentence)
    padding = np.zeros((pad_len, 5, input_size)) # 5-gram
    return np.concatenate((sentence, padding))

def batch(data, labels, sequence_lengths, batch_size, input_size):
    n_batch = int(math.ceil(len(data) / batch_size))
    index = 0
    for _ in range(n_batch):
        batch_sequence_lengths = np.array(sequence_lengths[index: index + batch_size])
        batch_length = np.array(max(batch_sequence_lengths)) # max length in batch
        batch_data = np.array([pad(x, batch_length, False, input_size) for x in data[index: index + batch_size]]) # pad data
        batch_labels = np.array([pad(x, batch_length, True) for x in labels[index: index + batch_size]]) # pad labels
        index += batch_size
        
        # Reshape input data to be suitable for LSTMs.
        batch_data = batch_data.reshape(-1, batch_length, input_size)
        
        
        yield batch_data, batch_labels, batch_length, batch_sequence_lengths


# In[9]:

# Bidirectional LSTM + CRF model.
learning_rate = 0.001
input_size = 300
batch_size = 128
num_units = 256 # the number of units in the LSTM cell
number_of_classes = 41

input_data = tf.placeholder(tf.float32, [None, None, input_size], name="input_data") # shape = (batch, batch_seq_len, input_size)
labels = tf.placeholder(tf.int32, shape=[None, None], name="labels") # shape = (batch, sentence)
batch_sequence_length = tf.placeholder(tf.int32) # max sequence length in batch
original_sequence_lengths = tf.placeholder(tf.int32, [None])

# Scope is mandatory to use LSTMCell (https://github.com/tensorflow/tensorflow/issues/799).
with tf.name_scope("BiLSTM"):
    with tf.variable_scope('forward'):
        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=1.0, state_is_tuple=True, name="fw_cell")
    with tf.variable_scope('backward'):
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=1.0, state_is_tuple=True, name="bw_cell")
    (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, 
                                                                     cell_bw=lstm_bw_cell, 
                                                                     inputs=input_data,
                                                                     sequence_length=original_sequence_lengths, 
                                                                     dtype=tf.float32,
                                                                     scope="BiLSTM")

# As we have a Bi-LSTM, we have two outputs which are not connected, so we need to merge them.
outputs = tf.concat([output_fw, output_bw], axis=2)

# Fully connected layer.
W = tf.get_variable(name="W", shape=[2 * num_units, number_of_classes],
                dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

b = tf.get_variable(name="b", shape=[number_of_classes], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

outputs_flat = tf.reshape(outputs, [-1, 2 * num_units])
pred = tf.matmul(outputs_flat, W) + b
scores = tf.reshape(pred, [-1, batch_sequence_length, number_of_classes])

# Linear-CRF.
log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(scores, labels, original_sequence_lengths)

loss = tf.reduce_mean(-log_likelihood)

# Compute the viterbi sequence and score (used for prediction and test time).
viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(scores, transition_params, original_sequence_lengths)

# Training ops.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

global_step = tf.Variable(0, name='global_step', trainable=False)

# Call this after declaring all tf.Variables.
saver = tf.train.Saver()


# In[15]:

training_epochs = 150

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2

# Training the model.
with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())
    
    ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path) # restore all variables


    start = global_step.eval() # get last global_step
    print("Start from:", start)
    
    for i in range(training_epochs):
        for batch_data, batch_labels, batch_seq_len, batch_sequence_lengths in batch(x_train, y_train, sequence_length_train, batch_size, input_size):
            tf_viterbi_sequence, _ = session.run([viterbi_sequence, train_op], 
                                                 feed_dict={input_data: batch_data,
                                                            labels: batch_labels, 
                                                            batch_sequence_length: batch_seq_len,
                                                            original_sequence_lengths: batch_sequence_lengths })

        # Show train accuracy.
        if i % 10 == 0:
            # Create a mask to fix input lengths.
            mask = (np.expand_dims(np.arange(batch_seq_len), axis=0) <
                np.expand_dims(batch_sequence_lengths, axis=1))
            total_labels = np.sum(batch_sequence_lengths)
            correct_labels = np.sum((batch_labels == tf_viterbi_sequence) * mask)
            accuracy = 100.0 * correct_labels / float(total_labels)
            print("Epoch: %d" % i, "Accuracy: %.2f%%" % accuracy)
    
        global_step.assign(i+1).eval() # set and update(eval) global_step with index, epoch_i

        # Save the variables to disk.
        saver.save(session, MODEL_PATH + "/model.ckpt", global_step=global_step-1)


