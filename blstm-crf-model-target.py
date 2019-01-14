import numpy as np
import math
import tensorflow as tf
from utils import read_xml_file
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd

# model with pre-trained weights that are going to be transfered (transfer learning)
MODEL_PATH = 'model_bi_lstm_crf_laptops_220k_50_75'

_, _, x_orig, y_orig, _, sequence_length_orig, original_orig_sentences = read_xml_file('data/Laptops_Train.xml')
_, _, x_test, y_test, _, sequence_length_test, original_test_sentences = read_xml_file('data/laptops-trial.xml')

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


# Bidirectional LSTM + CRF model.
learning_rate = 0.001
input_size = 300
num_units = 256 # the number of units in the LSTM cell
number_of_classes = 3

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


# transfer learning: load only part of the layers
variables_to_restore = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='BiLSTM/fw')
                        if var.name.startswith('BiLSTM/fw/fw_cell')] + [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='BiLSTM/bw')
                        if var.name.startswith('BiLSTM/bw/bw_cell')]

# https://stackoverflow.com/questions/45617026/tensorflow-transfer-learning-how-to-load-part-of-layers-from-one-checkpoint-file
# Save/restore variables from the old model (transfer learning).
saver = tf.train.Saver(variables_to_restore)



def get_valid_labels_predictions(batch_labels, tf_viterbi_sequence, batch_sequence_lengths):
    valid_labels = []
    valid_predictions = []
    for i in range(len(batch_labels)):
        l = batch_sequence_lengths[i]
        for j in range(l):
            valid_labels.append(batch_labels[i][j])
            valid_predictions.append(tf_viterbi_sequence[i][j])
            
    return valid_labels, valid_predictions


def count_errors(valid_labels, valid_predictions):

    # Post-processing
    for i in range(len(valid_labels)):
        if valid_labels[i] == 2:
            valid_labels[i-1] = 2
    
    zeros_idx = []
    singular_idx = []
    compound_idx = []

    i = 0
    while i < len(valid_labels):
        if valid_labels[i] == 0:
            zeros_idx.append(i)
            i += 1
        elif valid_labels[i] == 1:
            singular_idx.append(i)
            i += 1
        else:
            compound = []
            compound.append(i)
            i += 1
            while valid_labels[i] == 2:
                compound.append(i)
                i += 1
            compound_idx.append(compound)


    zeros_err = 0
    for i in range(len(zeros_idx)):
        if valid_predictions[zeros_idx[i]] != 0:
            zeros_err += 1

    singular_err = 0
    for i in range(len(singular_idx)):
        if valid_predictions[singular_idx[i]] == 0:
            singular_err += 1

    compound_err = 0
    for i in range(len(compound_idx)):
        for j in range(len(compound_idx[i])):
            if valid_predictions[compound_idx[i][j]] == 0:
                compound_err += 1
                break

    print("errors_count: ",zeros_err, singular_err, compound_err)
    total_aspects = len(singular_idx) + len(compound_idx)
    try:
        recall = (total_aspects - (singular_err + compound_err)) / total_aspects
        precision = (total_aspects - (singular_err + compound_err)) / ((total_aspects - (singular_err + compound_err)) + zeros_err)
        f1_score = 2 * (precision*recall) / (precision + recall)
        print("P: %.4f%%" % precision, "R: %.4f%%" % recall, "F1: %.4f%%" % f1_score)
    except ZeroDivisionError:
        print("zero division")


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2

training_epochs = 1030
runs = 10
batch_size = 128

# Training the model.
with tf.Session(config=config) as session:
    
    for curr_run in range(runs):
    
        session.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(session, ckpt.model_checkpoint_path) # restore all variables

        # Train
        for i in range(training_epochs):
            loss_total = 0
            loss_total_valid = 0
            for batch_data, batch_labels, batch_seq_len, batch_sequence_lengths in batch(x_orig, y_orig, sequence_length_orig, batch_size, input_size):
                tf_viterbi_sequence, tf_loss, _ = session.run([viterbi_sequence, loss, train_op], 
                                                     feed_dict={input_data: batch_data,
                                                                labels: batch_labels, 
                                                                batch_sequence_length: batch_seq_len,
                                                                original_sequence_lengths: batch_sequence_lengths })
                
                loss_total += tf_loss
            

            # Show train accuracy.
            if i % 10 == 0:
                # Create a mask to fix input lengths.
                mask = (np.expand_dims(np.arange(batch_seq_len), axis=0) <
                    np.expand_dims(batch_sequence_lengths, axis=1))
                total_labels = np.sum(batch_sequence_lengths)
                correct_labels = np.sum((batch_labels == tf_viterbi_sequence) * mask)
                accuracy = 100.0 * correct_labels / float(total_labels)
                print("Epoch: %d" % i, "Accuracy: %.2f%%" % accuracy, "Loss: %.6f%%" % loss_total)
        
                
                
        # Test    
        for batch_data, batch_labels, batch_seq_len, batch_sequence_lengths in batch(x_test, y_test, sequence_length_test, len(x_test), input_size):
                tf_viterbi_sequence, tf_valid_loss = session.run([viterbi_sequence, loss], feed_dict={input_data: batch_data,
                                                                labels: batch_labels, 
                                                                batch_sequence_length: batch_seq_len,
                                                                original_sequence_lengths: batch_sequence_lengths })
        
        # mask to correct input sizes
        mask = (np.expand_dims(np.arange(batch_seq_len), axis=0) <
            np.expand_dims(batch_sequence_lengths, axis=1))
        total_labels = np.sum(batch_sequence_lengths)
        correct_labels = np.sum((batch_labels == tf_viterbi_sequence) * mask)
        accuracy = 100.0 * correct_labels / float(total_labels)
        print("Test accuracy: %.2f%%" % accuracy)
        print("Test Loss: %.6f%%" % tf_valid_loss)

        valid_labels, valid_predictions = get_valid_labels_predictions(batch_labels.astype(int), tf_viterbi_sequence, batch_sequence_lengths)
        labels_df = pd.DataFrame(valid_labels)
        predictions_df = pd.DataFrame(valid_predictions)
        print(classification_report(valid_labels, valid_predictions, target_names=['O', 'B-ASPECT', 'I-ASPECT'], digits=3))
        print(confusion_matrix(valid_labels, valid_predictions, labels=[0, 1, 2]))

        count_errors(valid_labels, valid_predictions)
