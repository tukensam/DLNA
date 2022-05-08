import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from preprocess import get_full_data

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
    
        self.dense = tf.keras.Sequential()
        #self.dense.add(tf.keras.layers.Dense(6, activation='softmax'))
        #self.dense.add(tf.keras.layers.Dense(1000))
        self.dense.add(tf.keras.layers.Dense(1000))
        self.dense.add(tf.keras.layers.Dense(1000))
        self.dense.add(tf.keras.layers.Dense(6))

        self.loss_list = []
        self.epochs = 1000

        self.learning_rate = 1e-2
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    
    def call(self, inputs):
        probs = self.dense(inputs)
        return probs

    
    def loss(self, logits, labels):
        softmax = (tf.nn.softmax_cross_entropy_with_logits(labels, logits))
        return tf.math.reduce_mean(softmax)
        #loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
        #mean_loss = tf.math.reduce_mean(loss)
        #return mean_loss

    def accuracy(self, logits, labels):
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
        #return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return (tf.reduce_mean(tf.cast(corr_pred1, tf.float32)),
            tf.reduce_mean(tf.cast(corr_pred2, tf.float32)),
            tf.reduce_mean(tf.cast(corr_pred3, tf.float32)),
            tf.reduce_mean(tf.cast(corr_pred4, tf.float32)),
            tf.reduce_mean(tf.cast(corr_pred5, tf.float32)),
            tf.reduce_mean(tf.cast(corr_pred6, tf.float32)),
            tf.reduce_mean(tf.cast(correct_predictions, tf.float32)))


def train(model, train_inputs, train_labels):
    array = np.arange(len(train_inputs))
    np.random.shuffle(array)
    shuffled_inputs = tf.gather(train_inputs, array)
    shuffled_labels = tf.gather(train_labels, array)

    with tf.GradientTape() as tape:
        logits = model.call(shuffled_inputs)
        loss = model.loss(logits, shuffled_labels)
    model.loss_list.append(loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    logits = model.call(test_inputs)
    return model.accuracy(logits, test_labels)


def visualize_loss(losses): 
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()


def main():
    files = ["/Users/jacquesvonsteuben/Documents/Brown/CS/CS1470/DLNA/code/fasta_samples/mers.fasta",
            "/Users/jacquesvonsteuben/Documents/Brown/CS/CS1470/DLNA/code/fasta_samples/sarscov1.fasta",
            "/Users/jacquesvonsteuben/Documents/Brown/CS/CS1470/DLNA/code/fasta_samples/dengue.fasta",
            "/Users/jacquesvonsteuben/Documents/Brown/CS/CS1470/DLNA/code/fasta_samples/hepatitis.fasta",
            "/Users/jacquesvonsteuben/Documents/Brown/CS/CS1470/DLNA/code/fasta_samples/influenza.fasta",
            "/Users/jacquesvonsteuben/Documents/Brown/CS/CS1470/DLNA/code/fasta_samples/sarscov2.fasta"]
    train_data, test_data, train_labels, test_labels = get_full_data(files)

    train_labels = tf.one_hot(train_labels, 6)
    test_labels = tf.one_hot(test_labels, 6)
    train_inputs = []
    test_inputs = []

    for i in range(len(train_data)):
        x = train_data[i]
        boolean_mask = tf.cast(x, dtype=tf.bool)              
        no_zeros = tf.boolean_mask(x, boolean_mask, axis=0)
        length = no_zeros.get_shape().as_list()[0]
        train_inputs.append(length)

    for i in range(len(test_data)):
        x = test_data[i]
        boolean_mask = tf.cast(x, dtype=tf.bool)              
        no_zeros = tf.boolean_mask(x, boolean_mask, axis=0)
        length = no_zeros.get_shape().as_list()[0]
        test_inputs.append(length)

    model = Model()
    for epoch in range(model.epochs): #should I be running multiple epochs? Probably
        print(epoch)
        train(model, tf.expand_dims(tf.convert_to_tensor(train_inputs), 1), train_labels) #train called for each epoch; train dimensions here are: (batch_size, width, height, in_channels)
    acc1, acc2, acc3, acc4, acc5, acc6, total_accuracy = test(model, tf.expand_dims(tf.convert_to_tensor(test_inputs), 1), test_labels)
    print("ACCURACY1:")
    print(acc1)
    print("ACCURACY2:")
    print(acc2)
    print("ACCURACY3:")
    print(acc3)
    print("ACCURACY4:")
    print(acc4)
    print("ACCURACY5:")
    print(acc5)
    print("ACCURACY6:")
    print(acc6)
    print("TOTAL ACCURACY:")
    print(total_accuracy)
    visualize_loss(model.loss_list)


if __name__ == '__main__':
    main()