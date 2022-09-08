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

    def accuracy(self, labels, logits):
        logits_1 = []
        logits_2 = []
        logits_3 = []
        logits_4 = []
        logits_5 = []
        logits_6 = []
        labels_1 = []
        labels_2 = []
        labels_3 = []
        labels_4 = []
        labels_5 = []
        labels_6 = []
        argmax_labels = tf.argmax(labels, 1)
        for i in range(len(labels)):
            if argmax_labels[i] == 0:
                logits_1.append(logits[i])
                labels_1.append(labels[i])
            elif argmax_labels[i] == 1:
                logits_2.append(logits[i])
                labels_2.append(labels[i])
            elif argmax_labels[i] == 2:
                logits_3.append(logits[i])
                labels_3.append(labels[i])
            elif argmax_labels[i] == 3:
                logits_4.append(logits[i])
                labels_4.append(labels[i])
            elif argmax_labels[i] == 4:
                logits_5.append(logits[i])
                labels_5.append(labels[i])
            elif argmax_labels[i] == 5:
                logits_6.append(logits[i])
                labels_6.append(labels[i])
        corr_pred1 = tf.equal(tf.argmax(logits_1, 1), tf.argmax(labels_1, 1))
        corr_pred2 = tf.equal(tf.argmax(logits_2, 1), tf.argmax(labels_2, 1))
        corr_pred3 = tf.equal(tf.argmax(logits_3, 1), tf.argmax(labels_3, 1))
        corr_pred4 = tf.equal(tf.argmax(logits_4, 1), tf.argmax(labels_4, 1))
        corr_pred5 = tf.equal(tf.argmax(logits_5, 1), tf.argmax(labels_5, 1))
        corr_pred6 = tf.equal(tf.argmax(logits_6, 1), tf.argmax(labels_6, 1))

        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return np.array([tf.reduce_mean(tf.cast(corr_pred1, tf.float32)),
            tf.reduce_mean(tf.cast(corr_pred2, tf.float32)),
            tf.reduce_mean(tf.cast(corr_pred3, tf.float32)),
            tf.reduce_mean(tf.cast(corr_pred4, tf.float32)),
            tf.reduce_mean(tf.cast(corr_pred5, tf.float32)),
            tf.reduce_mean(tf.cast(corr_pred6, tf.float32)),
            tf.reduce_mean(tf.cast(correct_predictions, tf.float32))])

        
    
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
        accuracies = np.zeros([model.num_classes + 1])
        n = 0
        # generate a batch
        (batch_inputs, batch_labels) = (test_inputs[b: b + model.batch_size], test_labels[b: b + model.batch_size])
        # forward pass
        probs = model.call(batch_inputs)
        # get accuracies
        # accuracies += model.accuracy(batch_labels, probs)
        model.auroc.update_state(batch_labels, probs)
        model.categorical_accuracy.update_state(batch_labels, probs)
        confusion_matrix = confusion_matrix + tf.math.confusion_matrix(batch_labels.numpy().argmax(axis = 1), probs.numpy().argmax(axis = 1), model.num_classes)
        #n += 1

    return confusion_matrix #accuracies/n

def main():
    model = Model(6)

    epochs = 1
    sequencefiles = ["C:/Users/moasi/Desktop/CSCI_1470/DLNA/tukensam DLNA main code/fasta_data_new/sarscov2.fasta",
             "C:/Users/moasi/Desktop/CSCI_1470/DLNA/tukensam DLNA main code/fasta_data_new/mers.fasta", 
             "C:/Users/moasi/Desktop/CSCI_1470/DLNA/tukensam DLNA main code/fasta_data_new/sarscov1.fasta", 
             "C:/Users/moasi/Desktop/CSCI_1470/DLNA/tukensam DLNA main code/fasta_data_new/dengue.fasta", 
             "C:/Users/moasi/Desktop/CSCI_1470/DLNA/tukensam DLNA main code/fasta_data_new/hepatitis.fasta",
             "C:/Users/moasi/Desktop/CSCI_1470/DLNA/tukensam DLNA main code/fasta_data_new/influenza.fasta"
             ]
    # (train_data, test_data, train_labels, test_labels) = get_full_data(sequencefiles)
    # train_labels = tf.one_hot(train_labels, 6)
    # test_labels = tf.one_hot(test_labels, 6)

    with open("pickeled_data.pk", 'rb') as fi:
    # # dump your data into the file
        #pickle.dump((train_data, test_data, train_labels, test_labels), fi)
        (train_data, test_data, train_labels, test_labels) = pickle.load(fi)

    for e in range(epochs):
        train(model, train_data, train_labels)
        
        # accuracy = test(model, test_data, test_labels)
        confusion_matrix = test(model, test_data, test_labels)
        # print("SARS CoV 2 accuracy: " + str(accuracy[0]))
        # print("MERS accuracy: " + str(accuracy[1]))
        # print("SARS CoV 1 accuracy: " + str(accuracy[2]))
        # print("Dengue accuracy: " + str(accuracy[3]))
        # print("Hepatitis accuracy: " + str(accuracy[4]))
        # print("Influenza accuracy: " + str(accuracy[5]))
        # print("Total accuracy: " + str(accuracy[6]))
        print("Accuracy: " + str(model.categorical_accuracy.result().numpy()))
        print("AUROC: " + str(model.auroc.result().numpy()))
        print("Confusion Matrix: " + str(confusion_matrix))
        
    
    return


if __name__ == "__main__":
    main()
