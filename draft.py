import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

model = tf.keras.models.load_model(r'saved_model\1')
img = cv2.imread('5.png', cv2.IMREAD_GRAYSCALE)

img = img.reshape(img.shape[0], img.shape[1], 1)[None, ...]
img = tf.image.resize(img, (28, 28))

plt.imshow(img[0], 'gray')
plt.show()

print(np.argmax(model(img)))