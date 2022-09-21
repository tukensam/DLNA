import numpy as np
import tensorflow as tf
import pickle

from preprocess_final import get_full_data

class Model(tf.keras.Model):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
        self.categorical_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.auroc = tf.keras.metrics.AUC()
        self.recall = tf.keras.metrics.Recall()
        self.precision = tf.keras.metrics.Precision()
        self.batch_size = 1000
        self.embedding_size = 100
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
        probs = self.layers2(x)
        return probs

           
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
    confusion_matrix = np.zeros([model.num_classes, model.num_classes])
    for b in range(0, len(test_labels), model.batch_size):
        # generate a batch
        (batch_inputs, batch_labels) = (test_inputs[b: b + model.batch_size], test_labels[b: b + model.batch_size])
        # forward pass
        probs = model.call(batch_inputs)
        # get accuracies
        model.auroc.update_state(batch_labels, probs)
        model.recall.update_state(batch_labels, probs)
        model.precision.update_state(batch_labels, probs)
        model.categorical_accuracy.update_state(batch_labels, probs)
        confusion_matrix = confusion_matrix + tf.math.confusion_matrix(batch_labels.numpy().argmax(axis = 1), probs.numpy().argmax(axis = 1), model.num_classes)

    return confusion_matrix

def main():
    # Number of classes
    num_classes = 6
    # Number of epochs
    epochs = 1
    # Run preprocessing? If True, will dump preprocessed data in "pickled_data"
    # Otherwise, will read data from "pickled_data"
    run_preproc = True
    # File in which to store pickled data
    pickled_data = "pickeled_data.pk"
    # Directory of data files
    sequencefiles = ["C:/Users/moasi/Desktop/CSCI_1470/DLNA/tukensam DLNA main code/fasta_data_new/sarscov2.fasta",
             "C:/Users/moasi/Desktop/CSCI_1470/DLNA/tukensam DLNA main code/fasta_data_new/mers.fasta", 
             "C:/Users/moasi/Desktop/CSCI_1470/DLNA/tukensam DLNA main code/fasta_data_new/sarscov1.fasta", 
             "C:/Users/moasi/Desktop/CSCI_1470/DLNA/tukensam DLNA main code/fasta_data_new/dengue.fasta", 
             "C:/Users/moasi/Desktop/CSCI_1470/DLNA/tukensam DLNA main code/fasta_data_new/hepatitis.fasta",
             "C:/Users/moasi/Desktop/CSCI_1470/DLNA/tukensam DLNA main code/fasta_data_new/influenza.fasta"
             ]
     
    #===========================================================================
    model = Model(num_classes)

    if run_preproc:
        (train_data, test_data, train_labels, test_labels) = get_full_data(sequencefiles)
        train_labels = tf.one_hot(train_labels, 6)
        test_labels = tf.one_hot(test_labels, 6)

        with open(pickled_data, 'wb') as fi:
        # dump your data into the file
            pickle.dump((train_data, test_data, train_labels, test_labels), fi)
            #(train_data, test_data, train_labels, test_labels) = pickle.load(fi)
    else:
       with open(pickled_data, 'rb') as fi:
        # read from pickled file
            (train_data, test_data, train_labels, test_labels) = pickle.load(fi)
             
    for e in range(epochs):
        train(model, train_data, train_labels)
        # Print model metrics
        confusion_matrix = test(model, test_data, test_labels)
        print("Accuracy: " + str(model.categorical_accuracy.result().numpy()))
        print("AUROC: " + str(model.auroc.result().numpy()))
        print("Precision: " + str(model.precision))
        print("Recall: " + str(model.recall))
        print("F1 score: " + str(2 * model.recall*model.precision / (model.precision + model.recall)))
        print("Confusion Matrix: " + str(confusion_matrix))    
    return


if __name__ == "__main__":
    main()
