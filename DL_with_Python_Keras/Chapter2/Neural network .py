from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

# loading train data set
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preparing the image data
# * what is the function of reshape?
# print(train_images.shape)
train_images = train_images.reshape((train_images.shape[0], -1))
train_images = train_images.astype('float') / 255
# print(train_images.shape)

test_images = test_images.reshape((test_images.shape[0], -1))
test_images - test_images.astype('float') / 255

# Preparing the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Defining the model
network = models.Sequential()

# Adding layers
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28, )))
network.add(layers.Dense(10, activation='softmax'))

# summary
network.summary()

# Compiling
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Training
network.fit(train_images,
            train_labels,
            epochs=5,
            batch_size=256
            )

# Evaluating
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc', test_acc)
