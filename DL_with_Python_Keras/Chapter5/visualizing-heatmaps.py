from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import cv2

model = VGG16(weights='imagenet')

img_path = 'creative_commons_elephant.jpg'

# resize
img = image.load_img(img_path, target_size=(224, 224))

# convert to Numpy array
x = image.img_to_array(img)

# reshape (1, 244, 244, 3)
x = np.expand_dims(x, axis=0)

# preprocess the batch (channel-wise color normalization)
predicts = model.predict(x)
print('Predicted:', decode_predictions(predicts, top=3)[0])
print(np.argmax(predicts[0]))

# ----Setting up the Grad-CAM algorithm----
# this is the 'african elephant' entry in the prediction vector
african_elephant_output = model.output[:, 386]

# This is the output feature map of the `block5_conv3` layer
last_conv_layer = model.get_layer('block5_conv3')

# gradients
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

# This is a vector of shape (512, ) where each entry is
# the mean intensity of the gradient over a specific feature map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# this function allows us to access the value
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
# plt.matshow(heatmap)
# plt.show()

origin = cv2.imread(img_path)

heatmap = cv2.resize(heatmap, (origin.shape[1], origin.shape[0]))

heatmap = np.uint8(255 * heatmap)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = heatmap * 0.4 + origin

cv2.imwrite('elephant_cam.jpg', superimposed_img)