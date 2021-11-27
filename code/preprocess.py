import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import os

def get_image_list(filename):
    path = os.path.join(os.getcwd(), "data/Flickr8k_text", filename)
    with open(path) as f:
        imgs = f.read().splitlines()
    assert(len(imgs) == len(set(imgs))) # no image listed more than once
    return imgs
    

train_imgs = get_image_list("Flickr_8k.trainImages.txt") # 6000
val_imgs = get_image_list("Flickr_8k.devImages.txt") # 1000
test_imgs = get_image_list("Flickr_8k.testImages.txt") # 1000

# 8000 images
all_imgs = train_imgs + val_imgs + test_imgs
assert(len(all_imgs) == len(set(all_imgs))) # no image in more than one split category

# 8091 images in the Flicker_8k dataset
dir = os.path.join(os.getcwd(), "data/Flicker8k_Dataset")
file_list = os.listdir(dir)
assert(len(file_list) == len(set(file_list)))

# 91 images from Flicker_8k not used in train/val/test
diff = set(file_list).difference(set(all_imgs))
len(diff)




##### Pre-processing Images #####
# def process_images():
#     pass

# # root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# # img_file = os.path.join(root, "data/Flicker8k_Dataset/667626_18933d713e.jpg")

# img_file = os.path.join(os.getcwd(), "data/Flicker8k_Dataset/667626_18933d713e.jpg")

# img = image.img_to_array(image.load_img(img_file, target_size=(224,224)))

# # img = tf.keras.applications.vgg16.preprocess_input(img)

# print(img)

# print(img.shape)



##### Pre-processing Captions #####
# remove diacritics

# normalize 'hamza' 't marbuta' 'alif maqsura'

# remove punctuation

# remove non Arabic letters

# add start & end tokens 

# pad short captions with special token to ensure similar length captions