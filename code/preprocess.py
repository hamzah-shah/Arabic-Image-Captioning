import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
import re

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




# path = "/Users/khosrow/fall2021/dl/Arabic-Image-Captioning/data/Flickr8k_text/Flickr8k.arabic.full.txt"
# with open(path) as f:
#     imgs = f.read().splitlines()


def clean_caption(caption):
    # remove diacritics (taken from the source code)
    diacritics = re.compile("""
                                         ّ    | # Tashdid
                                         َ    | # Fatha
                                         ً    | # Tanwin Fath
                                         ُ    | # Damma
                                         ٌ    | # Tanwin Damm
                                         ِ    | # Kasra
                                         ٍ    | # Tanwin Kasr
                                         ْ    | # Sukun
                                         ـ     # Tatwil/Kashida
                                     """, re.VERBOSE)
    caption = re.sub(diacritics, '', caption)

    # normalize (taken from the source code)
    caption = re.sub("[إأآاٱ]", "ا", caption)
    caption = re.sub("ى", "ي", caption)
    caption = re.sub("ة", "ه", caption)  # replace ta2 marboota by ha2
    caption = re.sub("گ", "ك", caption)
    caption = re.sub("\u0640", '', caption)  # remove tatweel

    # remove punctuation
    punctuations = set(list('''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''))
    caption = ''.join([char for char in caption if char not in punctuations])

    # remove non-Arabic letters
    caption = re.sub(r'[a-zA-Z]+', '', caption)

    # remove numeric words
    caption = ' '.join([word for word in caption.split() if word.isalpha()])

    # add start and end tokens
    caption = "<start>" + ' ' + caption + ' ' + "<end>"

    # pad short captions with special token to ensure similar length captions
    # TODO

    return caption




def make_capition_dict(raw_data):
    output = {}

    for line in raw_data:
        
        # get the image name and caption
        if len(line.split('\t')) < 2:
            continue
        image_name, caption = line.split('\t')

        # clean the caption
        caption = clean_caption(caption)

        # remove the last 2 characters from the end of the image name
        image_name = image_name[:-2]

        # map the image to its caption
        if image_name in output:
            output[image_name].append(caption)
        else:
            output[image_name] = [caption]

    print(len(output), output)
    return output

make_capition_dict(imgs)