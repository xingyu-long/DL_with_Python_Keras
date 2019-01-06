import os
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

original_dataset_dir = '/home/xingyu/Downloads/Dogvscat/train'
base_dir = '/home/xingyu/Downloads/Dogvscat/train/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
train_cat_dir = os.path.join(train_dir, 'cats')
fnames = [os.path.join(train_cat_dir, fname) for fname in os.listdir(train_cat_dir)]

# pick one image
img_path = fnames[3]

# Read the image
img = image.load_img(img_path, target_size=(150, 150))

# Convert it to a Numpy array with shape(150, 150, 3)
x = image.img_to_array(img)

# Reshape it to (1, 150, 150, 3)
x = x.reshape((1, ) + x.shape)

i = 0

for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break

plt.show()
