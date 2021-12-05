import os
import pandas as pd
import numpy as np
import tensorflow as tf
from preprocess import get_data, preprocess_image
from model import Encoder, Decoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = "all_data.txt"

def get_image_list(filename):
    path = os.path.join(ROOT, "data/Flickr8k_text", filename)
    with open(path) as f:
        imgs = f.read().splitlines()
    assert(len(imgs) == len(set(imgs))) # no image listed more than once
    return imgs


def prep_data(file, image_list, txt_filename):
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
    # data['caption'] = data['caption'].str[::-1] // note the difference between whats printed out vs whats in the file

    prepped_data_path = os.path.join(ROOT, "data/Flickr8k_text", "all_data.txt")
    data.to_csv(prepped_data_path, sep="\t", header=False, index=False)
    return prepped_data_path

def split_data(img_to_captions, img_to_feats, train_list, val_list, test_list, vocab_size, max_len=20):
    '''
    Prepares the data to be passed into the model.
    :param img_to_captions: dictionary mapping each image to a list of captions
    :param img_to_feats: dictionary mapping each image to a 4096-dim feature vector
    :param train_list: list of images in training set 
    :param val_list: list of images in validaton set
    :param test_list: list of images in test set
    :param vocab_size: size of the vocabulary
    :param max_len: the maximum length of a caption
    :returns X_train, Y_train, X_val, Y_val, X_test, Y_test
    '''
    image_train, X_train, Y_train = [], [], []
    image_val, X_val, Y_val = [], [], []
    image_test, X_test, Y_test = [], [], []

    def split_helper(I, X, Y):
        captions = img_to_captions[image]
        for cap in captions:
            # loop through it -> pad its input & one-hot its output
            # print(f'CAPTION: {cap}')
            for ind in range(len(cap)):
                I.append(img_to_feats[image])
                input, output = cap[:ind], cap[ind]
                input = tf.keras.preprocessing.sequence.pad_sequences([input], maxlen=max_len, padding='post')[0] # taken from source code
                output = tf.one_hot(output, depth=vocab_size)
                X.append(input)
                Y.append(output)

    for image in img_to_captions:
        if image in train_list:
            split_helper(image_train, X_train, Y_train)
        elif image in val_list:
            split_helper(image_val, X_val, Y_val)
        elif image in test_list:
            split_helper(image_test, X_test, Y_test)
        else:
            ValueError("Image must be in one of the lists")

    assert(len(X_train) == len(Y_train))
    assert(len(X_val) == len(Y_val))
    assert(len(X_test) == len(Y_test))

    return np.array(image_train), np.array(X_train), np.array(Y_train), np.array(image_val), np.array(X_val), np.array(Y_val), \
            np.array(image_test), np.array(X_test), np.array(Y_test)


def get_features(model, image_list):
    '''
    Creates a map from each image to its features extracted from the Encoder model.
    :param model: the encoder model used to get features
    :param image_list: list of all images
    '''
    num_images = len(image_list)
    processed_images = []
    feat_map = {}
    for image in image_list:
        processed_image = preprocess_image(os.path.join(IMAGE_DIR, image))
        processed_images.append(processed_image)
    processed_images = np.array(processed_images)
    print(f'PROCESSED IMAGES SHAPE: {processed_images.shape}')
    
    features = model(processed_images)
    print(f'FEAT EXTRACTED IMAGES SHAPE: {features.shape}')
    exit()
    return feat_map

def train(model, input_image, input_text, label_text):
    '''
    Trains the decoder model.
    :param model: decoder model
    :param input_image
    :param input_text
    :param label_text
    '''
    model.compile(optimizer=model.optimizer, loss=model.loss)
    model.fit(x=[input_image, input_text], y=label_text, batch_size=model.batch_size, epochs=5)



if __name__ == "__main__":
    # TODO: STOP and PAD should be in vocabulary
    train_imgs = get_image_list("Flickr_8k.trainImages.txt") # 6000
    val_imgs = get_image_list("Flickr_8k.devImages.txt") # 1000
    test_imgs = get_image_list("Flickr_8k.testImages.txt") # 1000

    IMAGE_DIR = os.path.join(ROOT, "data/Flicker8k_Dataset")


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


    data_file = os.path.join(ROOT, "data/Flickr8k_text", "Flickr8k.arabic.full.txt")
    # prep_data(data_file, all_imgs, "test.txt")
    vocab, tokenized_captions = get_data(data_file)

    encoder, decoder = Encoder(), Decoder(len(vocab))
    
    print('before getting features')
    image_features = get_features(model=encoder, image_list=all_imgs)
    print('after getting features, before splitting data')
    I_train, X_train, Y_train, I_val, X_val, Y_val, I_test, X_test, Y_test = split_data(tokenized_captions, image_features, train_imgs, val_imgs, test_imgs, len(vocab))
    print('after splitting data')

    print(f'I_TRAIN SHAPE: {I_train.shape}')
    print(f'I_TRAIN SHAPE: {X_train.shape}')
    print(f'I_TRAIN SHAPE: {Y_train.shape}')
    train(decoder, I_train, X_train, Y_train)
    
    