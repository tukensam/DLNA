import numpy as np
import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        # Layer (type), Output Shape, Param #
        # Embedding, (None, 1000, 8), 128
        # Conv1D_1, (None, 1000, 128), 3200
        # MaxPooling1D_1, (None, 500, 128), 0
        # Conv1D_2, (None, 500, 64), 24640
        # MaxPooling_2, (None, 250, 64), 0
        # Flatten, (None, 16), 0
        # Dense1, (None, 128), 2176
        # Dense2, (None, 64), 8256
        # Dense3, (None, 6), 390 remember to adjust 6 for however many different viruses we are considering

        self.model = tf.keras.Sequential(layers=[
            tf.keras.layers.Embedding(input_dim=16, output_dim=8, input_length=1000), # why is input_dim = 16?
            tf.keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu'), # param num should be 3200, right now 2176 (128 * 17), 3200 (128 * 25)
            # tf.keras.layers.MaxPool1D(),
            # tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'),
            # tf.keras.layers.MaxPool1D(),
            # tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(activation='relu'),
            # tf.keras.layers.Dense(activation='relu'),
            # tf.keras.layers.Dense(activation='softmax')
        ])

        self.model.summary()

model = Model()
