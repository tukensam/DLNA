import numpy as np
import tensorflow as tf
import pickle

from realpreprocess import get_full_data

class Model(tf.keras.Model):
    def __init__(self, model_type, num_classes):
        super(Model, self).__init__()
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)       
        self.batch_size = 1000
        self.embedding_size = 100
        self.model_type = model_type
        self.num_classes = num_classes
        self.rnn_size = 100
        self.h1 = 250
        self.h2 = 100

        self.layers1 = tf.keras.Sequential(layers=[
           # tf.keras.layers.Embedding(5, self.embedding_size),
            tf.keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu'),
            tf.keras.layers.MaxPool1D(),
            tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'),
            tf.keras.layers.MaxPool1D(),
            tf.keras.layers.Flatten()
        ])
        self.LSTM_layer = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)
        self.biLSTM_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)) 
        self.layers2 = tf.keras.Sequential(layers=[
            tf.keras.layers.Dense(self.h1, activation='relu'),
            tf.keras.layers.Dense(self.h2, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])

    def call(self, x):
        """
        Perform the forward pass
        :param x: the inputs to the model
        :return:  the batch element probabilities as a tensor
        """
        x = self.layers1(x)
        # If using a special model type, do that here
        if self.model_type == "LSTM":
            x, _, _ = self.LSTM_layer(x)
        elif self.model_type == "biLSTM":
            x, _, _ = self.biLSTM_layer(x)
        
        probs = self.layers2(x)
        return probs
    

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    """
    Runs through one epoch
    :param train_inputs:
    :param train_labels:
    :return: None
    """
    indices = tf.random.shuffle(np.arange(tf.shape(train_inputs)[0]))
    train_inputs = tf.gather(train_inputs, indices)
    train_labels = tf.gather(train_labels, indices)

    for b in range(0, len(train_labels), model.batch_size):
        with tf.GradientTape() as tape:
            # generate a batch
            (batch_inputs, batch_labels) = (train_inputs[b: b + model.batch_size], train_labels[b: b + model.batch_size])
            # forward pass
            probs = model.call(batch_inputs)
            # calculate loss
            loss = model.loss(batch_labels, probs)
            # calculate gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        # update trainable variables
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return None

def test(model, test_inputs, test_labels):
    """
    Runs through one epoch
    :param train_inputs: 
    :param train_labels:
    :return: Average accuracy across the epoch
    """
    for b in range(0, len(test_labels), model.batch_size):
        accuracies = []
        # generate a batch
        (batch_inputs, batch_labels) = (test_inputs[b: b + model.batch_size], test_labels[b: b + model.batch_size])
        # forward pass
        probs = model.call(batch_inputs)
        # get accuracies
        accuracies.append(model.accuracy(probs, batch_labels))
        # calculate gradients
    return tf.reduce_mean(accuracies)

def main():
    model = Model("CNN", 6)
    epochs = 1
    sequencefiles = ["C:/Users/moasi/Desktop/CSCI_1470/DLNA/tukensam DLNA main code/fasta_data_new/sarscov2.fasta",
             "C:/Users/moasi/Desktop/CSCI_1470/DLNA/tukensam DLNA main code/fasta_data_new/mers.fasta", 
             "C:/Users/moasi/Desktop/CSCI_1470/DLNA/tukensam DLNA main code/fasta_data_new/sarscov1.fasta", 
             "C:/Users/moasi/Desktop/CSCI_1470/DLNA/tukensam DLNA main code/fasta_data_new/dengue.fasta", 
             "C:/Users/moasi/Desktop/CSCI_1470/DLNA/tukensam DLNA main code/fasta_data_new/hepatitis.fasta",
             "C:/Users/moasi/Desktop/CSCI_1470/DLNA/tukensam DLNA main code/fasta_data_new/influenza.fasta"
             ]
    (train_data, test_data, train_labels, test_labels) = get_full_data(sequencefiles)
    
    # with open("pickeled_data.pk", 'rb') as fi:
    # # dump your data into the file
    #    #pickle.dump((train_data, train_labels, test_data, test_labels), fi)
    #    (train_data, test_data, train_labels, test_labels) = pickle.load(fi)

    for e in range(epochs):
        min = np.amin(train_labels)
        max = np.amax(train_labels)
        train(model, train_data, train_labels)
        
        accuracy = test(model, test_data, test_labels)
        print(accuracy)
    
    return


if __name__ == "__main__":
    main()
