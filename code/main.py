import os
import numpy as np
import tensorflow as tf
import pickle
from preprocess import get_data, preprocess_image
from model import Encoder, Decoder
from bleu import BleuCallback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(ROOT, "data/Flicker8k_Dataset")
VGG_16FEATURES_FILE = os.path.join(ROOT, 'data/vgg16_features.pickle')
DATA_FILE = os.path.join(ROOT, "data/Flickr8k_text", "Flickr8k.arabic.full.txt")
TEST_IMGS_FILE = os.path.join(ROOT, "data/Flickr8k_text", "Flickr_8k.testImages.txt")

def get_image_list(file):
    with open(file) as f:
        imgs = f.read().splitlines()
    assert(len(imgs) == len(set(imgs))) # no image listed more than once
    return imgs

def split_data(img_to_caps, img_to_feats, train_list, test_list):
    '''
    Prepares the data to be passed into the model.
    :param img_to_caps: dictionary mapping each image to a list of captions
    :param img_to_feats: dictionary mapping each image to a 4096-dim feature vector
    :param train_list: list of images in training set 
    :param test_list: list of images in test set
    :returns I_train, X_train, Y_train, I_test, X_test, Y_test
    '''
    def split_helper(img, I, X, Y):
        features, captions = img_to_feats[img], img_to_caps[img]

        for cap in captions:
            I.append(features)
            input, label = np.delete(cap, -1), np.delete(cap, 0)
            X.append(input)
            Y.append(label)

    I_train, X_train, Y_train = [], [], []
    I_test, X_test, Y_test = [], [], []

    for image in img_to_caps:
        if image in train_list:
            split_helper(image, I_train, X_train, Y_train)
        elif image in test_list:
            split_helper(image, I_test, X_test, Y_test)
        else:
            raise Exception("Image must either be in the training data or testing data.")

    assert(len(I_train) == len(X_train) == len(Y_train))
    assert(len(I_test) == len(X_test) == len(Y_test))

    return tf.squeeze(tf.convert_to_tensor(I_train)), tf.convert_to_tensor(X_train), tf.convert_to_tensor(Y_train), \
        tf.squeeze(tf.convert_to_tensor(I_test)), tf.convert_to_tensor(X_test), tf.convert_to_tensor(Y_test)


def get_features(image_list):
    '''
    Creates a map from each image to its features extracted from the Encoder model.
    :param image_list: list of all images
    '''
    image_features = None
    try:
        with open(VGG_16FEATURES_FILE, 'rb') as file:
            image_features = pickle.load(file)
    except:
        encoder = Encoder()
        image_features = {}
        for image in image_list:
            processed_image = preprocess_image(os.path.join(IMAGE_DIR, image))
            image_features[image] = encoder(processed_image)
        with open(VGG_16FEATURES_FILE, 'wb') as file:
            pickle.dump(image_features, file)
    
    assert(image_features is not None)
    return image_features

    

def train(model_class, image_inputs, text_inputs, text_labels):
    '''
    Trains the decoder model.
    :param model_class: class for the decoder model
    :param image_inputs
    :param text_inputs
    :param text_labels
    :param pad_index
    '''
    
    decoder = model_class.get_model()
    decoder.compile(optimizer=model_class.optimizer, loss=model_class.loss)
    callbacks = []
    decoder.fit(x=[image_inputs, text_inputs], y=text_labels, batch_size=model_class.batch_size, epochs=5, validation_split=0.2, callbacks=callbacks)

    # TODO: add a callback in decoder.fit to print out BlEU score



def test():
    # TODO: predict captions using Itest, Xtest, Ytest
    # TODO: report BLEU score and loss

    # TODO: refer to the evaluate function in  this tutorial: https://www.tensorflow.org/tutorials/text/image_captioning 
    # (i pulled this tutorial into Google Collab, which made it easy to understand and follow)
        # also can look at the original src code - what we want to do is pretty similar
        # both the link above and the src code do something called 'greedy search', which is to pick the next most likely word at every time step
            # feed in the first word -> predict the second word
                # feed in first & second -> predict third
                    # feed in first, second, third -> predict fourth
                        # keep going until predict <END>

    # TODO: implement beam search, which is more optimal than greedy search

    # TODO: 
    pass



if __name__ == "__main__":
    # 8091 images in the Flicker_8k dataset
    all_imgs = os.listdir(IMAGE_DIR)
    test_imgs = get_image_list(TEST_IMGS_FILE) # 1000
    train_imgs = list(set(all_imgs) - set(test_imgs))

    vocab, img2caps = get_data(DATA_FILE) # word2index, padded and tokenized caps
    img2features = get_features(all_imgs)
    
    Itrain, Xtrain, Ytrain, Itest, Xtest, Ytest = split_data(img2caps, img2features, train_imgs, test_imgs)

    print(f'Training on {len(Xtrain)} examples.')
    print(f'Testing on {len(Xtest)} examples.')

    print(f'ITrain {Itrain.shape} ')
    print(f'XTrain {Xtrain.shape}')
    print(f'YTrain {Ytrain.shape}')

    print(f'Itest {Itest.shape} ')
    print(f'Xtest {Xtest.shape}')
    print(f'Ytest {Ytest.shape}')
    

    train(Decoder(vocab_size=len(vocab)), Itrain, Xtrain, Ytrain)
    # test() # TODO: implement test, but don't call it

    # TODO: make a function to plot loss and BLEU for both training and testing
    
    