from keras.datasets import mnist
from keras import models, layers, metrics, losses
from keras.utils import to_categorical

# loading train data set
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preparing the image data
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
train_images = train_images.astype('float') / 255
# print(train_images.shape)

test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
test_images = test_images.astype('float') / 255

# Preparing the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Model
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# Compiling the model
model.compile(optimizer='rmsprop',
              loss=losses.categorical_crossentropy,
              metrics=[metrics.categorical_accuracy])
# Training
model.fit(train_images,
          train_labels,
          epochs=5,
          batch_size=64)
# Evaluating
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc %f, test_loss %f' % (test_acc, test_loss))
