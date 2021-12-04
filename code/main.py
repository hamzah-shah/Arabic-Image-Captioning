import os
import pandas as pd
import numpy as np
from preprocess import get_data
from model import EndtoEnd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_image_list(filename):
    path = os.path.join(ROOT, "data/Flickr8k_text", filename)
    with open(path) as f:
        imgs = f.read().splitlines()
    assert(len(imgs) == len(set(imgs))) # no image listed more than once
    return imgs

def prep_data(file, image_list):
    data = pd.read_csv(file, header=None, delimiter="\t", on_bad_lines="skip", names=["image_caption", "caption"])
    data['image'] = data['image_caption'].str[:-2]
    
    x = (np.where(data['image'].str[-4:] != ".jpg"))[0][0]
    data = data.drop([x])

    def is_unique(s):
        a = s.to_numpy() # s.values (pandas<0.24)
        return (".jpg" == a).all()

    assert(is_unique(data['image'].str[-4:]))

    data = data[data['image'].isin(image_list)]
    data = data.drop(columns=['image'])
    data['caption'] = data['caption'].str[::-1]

    print(data)

    

if __name__ == "__main__":
    train_imgs = get_image_list("Flickr_8k.trainImages.txt") # 6000
    val_imgs = get_image_list("Flickr_8k.devImages.txt") # 1000
    test_imgs = get_image_list("Flickr_8k.testImages.txt") # 1000


    # 8000 images
    all_imgs = train_imgs + val_imgs + test_imgs
    assert(len(all_imgs) == len(set(all_imgs))) # no image in more than one split category

    # 8091 images in the Flicker_8k dataset
    dir = os.path.join(ROOT, "data/Flicker8k_Dataset")
    file_list = os.listdir(dir)
    assert(len(file_list) == len(set(file_list)))

    # 91 images from Flicker_8k not used in train or val or test
    diff = set(file_list).difference(set(all_imgs))
    len(diff)


    # TODO: construct vocab & lemmatize captions for only those images who are in train, val or test (exclude the 91 not used)
    data_file = os.path.join(ROOT, "data/Flickr8k_text", "Flickr8k.arabic.full.txt")
    prep_data(data_file, all_imgs)
    exit()
    vocab, tokenized_captions = get_data(data_file)
    # exit()

    print(f'Vocab size: {len(vocab)}') # TODO: vocab size is only 10,700 - isn't that small considering there are 8091 images?
    print(f'Number of images: {len(tokenized_captions)}')

    # initialize model
    model = EndtoEnd(arabic_vocab_size=len(vocab))