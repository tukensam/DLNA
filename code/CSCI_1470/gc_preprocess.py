from cgi import test
from textwrap import indent
import numpy as np
import tensorflow as tf
from Bio import SeqIO
import random

def get_data(file_name):
    parsed = SeqIO.parse(open(file_name),'fasta')
    seqlist = []
    num_examples = 0
    for fasta in parsed:
        seqlist.append(str(fasta.seq))
        num_examples += 1

    return seqlist, num_examples

def get_full_data(file_names, train_test_split = .8):
    seqlist = []
    labels = []
    numexamples = 0
    for i in range(len(file_names)):
        oh_i, length = get_data(file_names[i])
        seqlist.append(oh_i)
        numexamples += length
        labels_i = tf.fill(length, i)
        labels.append(labels_i)

    labels = tf.concat(labels, axis = 0)

    all_data = [val for inner in seqlist for val in inner]
    all_labels = labels

    shuffler = np.arange(0, numexamples).tolist()
    np.random.shuffle(shuffler)

    all_data = [all_data[i] for i in shuffler]

    tensor_shuffler = tf.convert_to_tensor(shuffler)
    all_labels = tf.gather(labels, tensor_shuffler)

    split_ind = int(numexamples * train_test_split)
    

    train_data, test_data=  all_data[:split_ind], all_data[split_ind:]
    train_labels, test_labels = all_labels[:split_ind], all_labels[split_ind:]

    return train_data, test_data, train_labels, test_labels
