from keras.applications import VGG16
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


def deprocess_image(x):
    # normalize tensor: center on 0, std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern(layer_name, filter_index, size=150):
    layer_out = model.get_layer(layer_name).output
    loss = K.mean(layer_out[:, :, :, filter_index])

    # gradients and smooth it
    gradients = K.gradients(loss, model.input)[0]
    gradients /= (K.sqrt(K.mean(K.square(gradients))) + 1e-5)

    iterate = K.function([model.input], [loss, gradients])

    # We start from a gray image with some noise
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128

    # Run gradient ascent for 40 steps
    step = 1
    for i in range(40):
        # Compute the loss value and gradient value
        loss_value, gradients_value = iterate([input_img_data])
        # Adjust the input image in the direction that maximizes the loss
        input_img_data += gradients_value * step
    img = input_img_data[0]
    return deprocess_image(img)


model = VGG16(weights='imagenet',
              include_top=False)

layer_name = 'block1_conv1'
size = 64
margin = 5

# This a empty (black) image where we will store
results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

for i in range(8):  # iterate over the rows of our result grid
    for j in range(8):  # iterate over the columns of our results grid
        # Generate the pattern for filter `i + (j * 8)` in `layer_name`
        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)

        # Put the result in the square `(i, j)` of the results grid
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

plt.figure(figsize=(20, 20))
plt.imshow(results)
plt.show()