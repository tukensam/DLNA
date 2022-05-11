from cgi import test
from textwrap import indent
import numpy as np
import tensorflow as tf
from Bio import SeqIO


def get_data(file_name, seq_size = 300):
    '''
    Helper function. Do not call directly.

    Reads and parses a FASTA file. Creates one_hot encoding of n num_examples DNA
    sequences of length seq_size, where num_examples = num_seqs_in_fasta // seq_size

    :param file_name: path to .fasta file

    :return: (onehot float32 RaggedTensor, number of sequences)

    one_hot scheme:
    "A" : 0, "C" : 1, "G" : 2, "T" : 3, "N" : 4
    '''
    parsed = SeqIO.parse(open(file_name),'fasta')
    seqlist = []
    num_examples = 0
    for fasta in parsed:
        seqstr = str(fasta.seq)
        seqstr = seqstr[np.random.randint(0,300):]
        newseqstr = seqstr[: len(seqstr) - (len(seqstr) % seq_size)]
        for i in range(len(newseqstr) // seq_size):
            seqlist.append(newseqstr[ i * seq_size: (i + 1) * seq_size])
            num_examples += 1

    seq_tensor = tf.convert_to_tensor(seqlist)
    split_seq_tensor = tf.strings.bytes_split(seq_tensor)

    dnas = ["A", "C", "G", "T", "N"]

    matches = tf.stack([tf.equal(split_seq_tensor, s) for s in dnas], axis=-1)
    onehot = tf.cast(matches, tf.float32)

    onehot = onehot.to_tensor()

    print(tf.shape(onehot))

    return onehot, num_examples

def get_full_data(file_names, train_test_split = .8, seq_size = 300):
    '''
    Reads and parses a list of fasta FASTA files. Creates one_hot encoding of DNA sequence
    and tensor of labels. Sequences are of length seq_size and split into train/test by train_test_split.

    
    :param file_names: list of paths to .fasta files, each corresponding to all sequences from one virus
    :param train_test_split: float, proportion of sequences desired to be in train set
    :param seq_size: Length of desired sequences.

    :return: tuple of (train_data, train_labels, test_data, test_labels)
    :return train_data: One hot Tensor
    :return test_data: One hot Tensor
    :return train_labels: Integer tensor of labels (same length as train_data)
    :return test_labels: Integer tensor of labels (same length as test_data)

    dim(train_data) = (num_examples * train_test_split, len_longest_sequence, 5)
    dim(train_labels) = (num_examples * train_test_split, )

    one_hot Tensors follow the scheme in get_data()
    Sequences whose length is less than the longest sequence length (across all fasta files) 
    are filled with [0, 0, 0, 0, 0] encodings for dimensions to match.

    label scheme:
    file_names = [<covid.fasta>, <influenza.fasta>, <mers.fasta>, ...]
    labels covid -> 0
           influenza -> 1
           mers -> 2
           ... 

    Both sets of data are randomly shuffled

    call as:
    train_X, test_X, train_Y, test_Y = get_full_data([<virus0.fasta>, <virus1.fasta>, virus2.fasta,...], .8)
    
    Remember to batch and re-shuffle after every epoch with something like:

    shuffindices = tf.random.shuffle(list(range(nume_examples)))
    train_X = tf.gather(train_X, shuffindices)
    train_Y = tf.gather(train_Y, shuffindices)

    **We can use something like numpy.save(file_name), and load data in batches if memory gets too bad**

    '''
    onehots = []
    labels = []
    numexamples = 0
    for i in range(len(file_names)):
        oh_i, length = get_data(file_names[i], seq_size)
        onehots.append(oh_i)
        numexamples += length
        labels_i = tf.fill(length, i)
        labels.append(labels_i)

    labels = tf.concat(labels, axis = 0)
    filled_onehot = tf.concat(onehots, axis = 0)

    shuffindices = tf.random.shuffle(list(range(numexamples)))

    all_data = tf.gather(filled_onehot, shuffindices)
    all_labels = tf.gather(labels, shuffindices)

    split_ind = int(numexamples * train_test_split)
    

    train_data, test_data=  all_data[:split_ind], all_data[split_ind:]
    train_labels, test_labels = all_labels[:split_ind], all_labels[split_ind:]

    print(tf.shape(train_data))
    print(tf.shape(test_data))
    print(tf.shape(train_labels))
    print(tf.shape(test_labels))

    return train_data, test_data, train_labels, test_labels

#How to call in model:

sequencefiles = ["fasta_data_new/sarscov2.fasta",
             "fasta_data_new/mers.fasta", 
             "fasta_data_new/sarscov1.fasta", 
             "fasta_data_new/dengue.fasta", 
             "fasta_data_new/hepatitis.fasta",
             "fasta_data_new/influenza.fasta"
             ]
get_full_data(sequencefiles)

