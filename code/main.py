import os
from preprocess_caption import get_data
from model import EndtoEnd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_image_list(filename):
    path = os.path.join(os.getcwd(), "data/Flickr8k_text", filename)
    with open(path) as f:
        imgs = f.read().splitlines()
    assert(len(imgs) == len(set(imgs))) # no image listed more than once
    return imgs
    

if __name__ == "__main__":
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

    # load data
    data_file = os.path.join(ROOT, "data/Flickr8k_text", "Flickr8k.arabic.full.txt")
    vocab, tokenized_captions = get_data(data_file)

    # initialize model
    model = EndtoEnd(arabic_vocab_size=len(vocab))