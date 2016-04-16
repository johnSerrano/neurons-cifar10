import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
import numpy as np
import sys

IMAGE_SIZE = 32
NUM_ITER = 10
BATCH_SIZE = 128
NUM_CATEGORIES = 10


# Extract data from pickle files
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


# Reshape image data for pyplot
def process_image(img):
    pixels = []
    image = []

    for i in range(IMAGE_SIZE*IMAGE_SIZE):
        pixel = [img[i], img[i+(IMAGE_SIZE*IMAGE_SIZE)],
                img[i+(IMAGE_SIZE*IMAGE_SIZE*2)]]
        pixels += [pixel]

    for i in range(IMAGE_SIZE):
        row = []
        for j in range(IMAGE_SIZE):
            row += [pixels[(IMAGE_SIZE*i)+j]]
        image += [row]

    return image


# check command line args for number of iterations
if (len(sys.argv) > 1):
    NUM_ITER = int(sys.argv[1])


# Extract image data from cifar-10 files
print("Processing data...")
data_1 = unpickle("cifar-10-batches-py/data_batch_1")
data_2 = unpickle("cifar-10-batches-py/data_batch_2")
data_3 = unpickle("cifar-10-batches-py/data_batch_3")
data_4 = unpickle("cifar-10-batches-py/data_batch_4")
data_5 = unpickle("cifar-10-batches-py/data_batch_5")
test = unpickle("cifar-10-batches-py/test_batch")
meta = unpickle("cifar-10-batches-py/batches.meta")


#combine datasets
data_all = np.vstack([data_1["data"], data_2["data"], data_3["data"], data_4["data"], data_5["data"]])
labels_all = data_1["labels"] + data_2["labels"] + data_3["labels"] + data_4["labels"] + data_5["labels"]


# Process data for training
test_data = test["data"]
test_labels = np_utils.to_categorical(np.array(test["labels"]), NUM_CATEGORIES)
labels = np_utils.to_categorical(np.array(labels_all), NUM_CATEGORIES)

data_all = data_all.reshape(data_all.shape[0], 3, 32, 32)
test_data = test_data.reshape(test_data.shape[0], 3, 32, 32)

data_all = data_all.astype('float32')
test_data = test_data.astype('float32')
data_all /= 255
test_data /= 255


# Define network
print("Generating model...")
model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode="same", input_shape=(3, 32, 32)))
model.add(Activation("relu"))
model.add(Convolution2D(46, 3, 3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.3))
model.add(Convolution2D(64, 3, 3, border_mode="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(.3))
model.add(Flatten())
model.add(Dense(500, input_dim=3072, init="glorot_uniform"))
model.add(Activation("relu"))
model.add(Dense(10, init="glorot_uniform"))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adadelta")


# Train network
model.fit(data_all, labels, nb_epoch=NUM_ITER, batch_size=BATCH_SIZE, show_accuracy=True, verbose=1, shuffle=True)


# Test network
result = model.evaluate(test_data, test_labels, batch_size=BATCH_SIZE, show_accuracy=True, verbose=0, sample_weight=None)
print('Test score:', result[0])
print('Test accuracy:', result[1])


# Predict results for test data
predictions = model.predict_classes(test_data, batch_size=BATCH_SIZE, verbose=0)


# Display 9 random results from test data
test_data = test_data.reshape(test_data.shape[0], 3072)
for i in range(9):
    j = random.randint(0, 10000)
    plt.subplot(3,3,i+1)

    image = process_image(test_data[j])

    plt.imshow(image, cmap='gray', interpolation='none')
    plt.title(meta["label_names"][test["labels"][j]] + " : " + meta["label_names"][predictions[j]])
plt.show()
