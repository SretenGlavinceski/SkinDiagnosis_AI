import cv2
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

vgg_model = VGG16(weights='imagenet')
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

def extract_features(image):
    resized_img = cv2.resize(image, (224, 224))

    img_array = resized_img.astype('float32')
    img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
    img_array = preprocess_input(img_array)

    cnn_features = vgg_model.predict(img_array, verbose=0)

    return cnn_features.flatten()
