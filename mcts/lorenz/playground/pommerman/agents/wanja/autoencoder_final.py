import numpy as np

# loads the states saved by "embeddings.py" in the same folder
data = np.load("states.npy")

# from https://blog.keras.io/building-autoencoders-in-keras.html

import keras
from keras import layers

# This is the size of our encoded representations
# this parameter has to be finetuned, it highly influences the quality of the encoding
# encode state into 2-embedding and send as message??
encoding_dim = 32

# This is our input image
input_img = keras.Input(shape=(38*81,))
# "encoded" is the encoded representation of the input
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(38*81, activation='sigmoid')(encoded)

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)

# This model maps an input to its encoded representation
encoder = keras.Model(input_img, encoded)

# This is our encoded (32-dimensional) input
encoded_input = keras.Input(shape=(encoding_dim,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Create the decoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# train test split
x_train = data[:int(0.7*len(data))]
x_test = data[int(0.7*len(data)):]

autoencoder.fit(x_train, x_train,
                epochs=30, # tends to overfit with more epochs
                batch_size=128, #probably best. tested: 4,32,64,128,256
                shuffle=True,
                validation_data=(x_test, x_test))
