import os, shutil

# original dataset path
original_dataset_dir = '/home/xingyu/Downloads/Dogvscat/train'

# create small dataset
base_dir = '/home/xingyu/Downloads/Dogvscat/train/cats_and_dogs_small'
# os.mkdir(base_dir)

# Directories for training, validation, test
train_dir = os.path.join(base_dir, 'train')
# os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
# os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
# os.mkdir(test_dir)

# Directory with our training cat pics
train_cat_dir = os.path.join(train_dir, 'cats')
# os.mkdir(train_cat_dir)

# Directory with our training dog pics
train_dog_dir = os.path.join(train_dir, 'dogs')
# os.mkdir(train_dog_dir)

# Directory with our validation cat pics
validation_cat_dir = os.path.join(validation_dir, 'cats')
# os.mkdir(validation_cat_dir)

# Directory with our validation dog pics
validation_dog_dir = os.path.join(validation_dir, 'dogs')
# os.mkdir(validation_dog_dir)

# Directory with our test cat pics
test_cat_dir = os.path.join(test_dir, 'cats')
# os.mkdir(test_cat_dir)

# Directory with our test dog pics
test_dog_dir = os.path.join(test_dir, 'dogs')
# os.mkdir(test_dog_dir)

# Copy first 1000 cat images to train_cat_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cat_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 500 cat images to validation_cat_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cat_dir, fname)
    shutil.copyfile(src, dst)

# Copy first 1000 cat images to test_cat_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cat_dir, fname)
    shutil.copyfile(src, dst)

# Copy first 1000 dog images to train_dog_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dog_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 500 dog images to validation_dog_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dog_dir, fname)
    shutil.copyfile(src, dst)

# Copy first 1000 dog images to test_dog_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dog_dir, fname)
    shutil.copyfile(src, dst)
