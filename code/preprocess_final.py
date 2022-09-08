from cgi import test
from textwrap import indent
import numpy as np
import tensorflow as tf
from Bio import SeqIO

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

    train_labels = []
    test_labels = []
    data = []
    train_data = []
    test_data = []
    num_samples = []
    num_bps = []
    
    for i in range(len(file_names)):
        # Loads each file in
        parsed = list(SeqIO.parse(open(file_names[i]),'fasta'))
        data.append(parsed)
        num_bps.append(np.size(np.concatenate(parsed)))
        num_samples.append(len(parsed))
        print("THIS IS A TEST " + str(i))


    for i in range(len(file_names)):
        #TODO: ASK ABOUT THIS: We are subsampling by rounding the number of samples to provide
        #   roughly equivalent number of base pairs per virus species. Should we instead base
        #   solely on number of samples? Solely on base pairs, splitting samples?
        
        subsample_size = int(round(min(num_bps) * num_samples[i] / num_bps[i]))
        split_index = int(round(subsample_size * train_test_split))
        train_data.append(data[i][split_index:])
        test_data.append(data[i][:split_index - 1])
        
        train_data[i] = np.concatenate(train_data[i])
        test_data[i] = np.concatenate(test_data[i])

        train_trim = np.size(train_data[i]) % seq_size
        test_trim = np.size(test_data[i]) % seq_size
        
        train_data[i] = train_data[i][:-train_trim]
        test_data[i] = test_data[i][:-test_trim]
        
        train_data[i] = np.reshape(train_data[i], [-1, seq_size])
        test_data[i] = np.reshape(test_data[i], [-1, seq_size])

        train_labels.append(tf.fill(np.shape(train_data[i])[0], i))
        test_labels.append(tf.fill(np.shape(test_data[i])[0], i))
        print("THIS IS A TEST ALSO " + str(i))
    
    data = None
    parsed = None

    train_data = tf.convert_to_tensor(np.concatenate(train_data))
    #train_split_seq_tensor = tf.strings.bytes_split(train_data)

    test_data = tf.convert_to_tensor(np.concatenate(test_data))
    #test_split_seq_tensor = tf.strings.bytes_split(test_data)

    print(tf.shape(train_data))
    print(tf.shape(test_data))

    dnas = ["A", "C", "G", "T", "N"]

    train_data = tf.stack([tf.equal(train_data, s) for s in dnas], axis=-1)
    train_data = tf.cast(train_data, tf.float32)
    # train_data = train_data.to_tensor()

    test_data = tf.stack([tf.equal(test_data, s) for s in dnas], axis=-1)
    test_data = tf.cast(test_data, tf.float32)
    #test_data = test_data.to_tensor()

    train_labels = tf.concat(train_labels, axis = 0)
    test_labels = tf.concat(test_labels, axis = 0)
    # filled_onehot = tf.concat(onehots, axis = 0)

    print(tf.shape(train_data))
    print(tf.shape(test_data))
    print(tf.shape(train_labels))
    print(tf.shape(test_labels))

    return train_data, test_data, train_labels, test_labels

#How to call in model:

# sequencefiles = ["fasta_data_new/sarscov2.fasta",
#              "fasta_data_new/mers.fasta", 
#              "fasta_data_new/sarscov1.fasta", 
#              "fasta_data_new/dengue.fasta", 
#              "fasta_data_new/hepatitis.fasta",
#              "fasta_data_new/influenza.fasta"
#              ]
# get_full_data(sequencefiles)
