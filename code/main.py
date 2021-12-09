import os
import numpy as np
import tensorflow as tf
import pickle
from preprocess import get_data, preprocess_image
from model import Encoder, Decoder
from bleu import BleuCallback
from bleu import *

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(ROOT, "data/Flicker8k_Dataset")
VGG_16FEATURES_FILE = os.path.join(ROOT, 'data/vgg16_features.pickle')
DATA_FILE = os.path.join(ROOT, "data/Flickr8k_text", "Flickr8k.arabic.full.txt")
TEST_IMGS_FILE = os.path.join(ROOT, "data/Flickr8k_text", "Flickr_8k.testImages.txt")


START_TOKEN = "<START>"
END_TOKEN = "<END>"
SPACE = " "
MAXLEN = 20

def get_image_list(file):
    with open(file) as f:
        imgs = f.read().splitlines()
    assert(len(imgs) == len(set(imgs))) # no image listed more than once
    return imgs

def prep_data(img_to_caps, img_to_feats, train_list):
    '''
    Prepares the training data to be passed into the model.
    :param img_to_caps: dictionary mapping each image to a list of captions
    :param img_to_feats: dictionary mapping each image to a feature vector
    :param train_list: list of images in training set 
    :returns I_train, X_train, Y_train
    '''
    I_train, X_train, Y_train = [], [], []

    for image in train_list:
        features, captions = img_to_feats[image], img_to_caps[image]
        for cap in captions:
            I_train.append(features)
            input, label = np.delete(cap, -1), np.delete(cap, 0)
            X_train.append(input)
            Y_train.append(label)

    assert(len(I_train) == len(X_train) == len(Y_train))

    return tf.squeeze(tf.convert_to_tensor(I_train)), tf.convert_to_tensor(X_train), tf.convert_to_tensor(Y_train)


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

    

def train(model_class, model, image_inputs, text_inputs, text_labels):
    '''
    Trains the decoder model.
    :param model_class: class for the decoder model
    :param model
    :param image_inputs
    :param text_inputs
    :param text_labels
    :param pad_index
    '''
    model.compile(optimizer=model_class.optimizer, loss=model_class.loss)
    callbacks = [BleuCallback()]
    model.fit(x=[image_inputs, text_inputs], y=text_labels, batch_size=model_class.batch_size, epochs=5, validation_split=0.2, callbacks=callbacks)



def test(model, img_to_feats, testing_images, img2caps):
    '''
    Tests the decoder model using greedy search.
    :param model
    :param img_to_feats: map from image to feature vector
    :param testing_images: list of test images
    '''
    img2prediction = {}
    for img in testing_images:
        feat_vector = img_to_feats[img]
        predicted_caption = predict_caption(model, feat_vector)
        img2prediction[img] = predicted_caption
    
    # TODO: BLEU stuff
    one_gram_mean, two_gram_mean, three_gram_mean, four_gram_mean = bleu_score(testing_images, img2caps, img2prediction)

    print('one_gram: ' + str(one_gram_mean), 'two_gram: '+ str(two_gram_mean), 'three_gram: ' + str(three_gram_mean), 'four_gram: ' + str(four_gram_mean))
    visualize_accuracy([one_gram_mean, two_gram_mean, three_gram_mean, four_gram_mean])
    





def predict_caption(model, image_feats):
    '''
    Given a trained model and an encoded image features, returns a predicted caption.
    :param model
    :param image_feats: an feature vector that encodes an image
    return predicted caption
    '''
    cap = START_TOKEN
    for _ in MAXLEN:
        padded_cap = tf.keras.preprocessing.sequence.pad_sequences([cap], maxlen=MAXLEN, padding='post')
        next_word_dist = model.predict(x=[image_feats, padded_cap])
        next_word = tf.argmax(next_word_dist)
        cap = cap + SPACE + next_word

        if next_word == END_TOKEN:
            break
    
    return cap



if __name__ == "__main__":
    # 8091 images in the Flicker_8k dataset
    all_imgs = os.listdir(IMAGE_DIR)
    test_imgs = get_image_list(TEST_IMGS_FILE) # 1000
    train_imgs = list(set(all_imgs) - set(test_imgs))

    vocab, img2tokenizedcaps, img2caps = get_data(DATA_FILE) # word2index, padded and tokenized caps
    img2features = get_features(all_imgs)
    
    Itrain, Xtrain, Ytrain = prep_data(img2tokenizedcaps, img2features, train_imgs)

    print(f'Training on {len(Xtrain)} examples...')
    print(f'Training on {len(train_imgs)} examples...')
    print(f'Testing on {len(test_imgs)} images.')

    # print(f'ITrain {Itrain.shape} ')
    # print(f'XTrain {Xtrain.shape}')
    # print(f'YTrain {Ytrain.shape}')
    
    decoder_instance = Decoder(vocab_size=len(vocab))
    decoder = decoder_instance.get_model()
    train(decoder_instance, decoder, Itrain, Xtrain, Ytrain)
    test(decoder, img2features, test_imgs, img2caps)

    # TODO: make a function to plot loss and BLEU for both training and testing
    
    