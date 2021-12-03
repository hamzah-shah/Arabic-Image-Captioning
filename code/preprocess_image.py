import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

VGG16_IMG_SIZE = (224,224)

def preprocess_image(img):
    '''
    Preprocesses an image according to VGG16 preprocessing.
    :param image: image to be processed
    :return image: VGG16 processed image
    '''
    img = image.img_to_array(image.load_img(img, target_size=VGG16_IMG_SIZE))
    return tf.keras.applications.vgg16.preprocess_input(img)


# img_file = os.path.join(ROOT, "data/Flicker8k_Dataset/667626_18933d713e.jpg")

