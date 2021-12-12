import os
from re import A
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from preprocess import get_data, preprocess_image, show_image
from model import Encoder, Decoder
from bleu import *

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(ROOT, "data/Flicker8k_Dataset")
VGG_16FEATURES_FILE = os.path.join(ROOT, 'data/vgg16_features.pickle')
DATA_FILE = os.path.join(ROOT, "data/Flickr8k_text", "Flickr8k.arabic.full.txt")
TEST_IMGS_FILE = os.path.join(ROOT, "data/Flickr8k_text", "Flickr_8k.testImages.txt")
MODEL_PATH = os.path.join(ROOT, "trained_model")
PREDICTION_FILE = os.path.join(ROOT, 'results/prediction.pickle')


START_TOKEN = "<START>"
END_TOKEN = "<END>"
PAD_TOKEN = "<PAD>"
SPACE = " "
MAXLEN = 20


def get_image_list(file):
    '''
    Returns a list of images in the raw file.
    :param file: raw file to read from
    '''
    with open(file) as f:
        imgs = f.read().splitlines()
    assert(len(imgs) == len(set(imgs))) # no image listed more than once
    return imgs

def prep_data(img_to_token_caps, img_to_feats, train_list):
    '''
    Prepares the training data to be passed into the model.
    :param img_to_token_caps: dictionary mapping each image to a list of tokenized captions
    :param img_to_feats: dictionary mapping each image to a feature vector
    :param train_list: list of images in training set 
    :returns 3 tensors: 1) image features, 2) input text, 3) output text
    '''
    I_train, X_train, Y_train = [], [], []

    for image in train_list:
        features, captions = img_to_feats[image], img_to_token_caps[image]
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
    :returns dict of img -> feature vec
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


def plot_loss(hist):
    '''
    Plots training and validation loss.
    '''
    # loss plot
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
    plt.savefig('loss.png')


def train(model_class, model, image_inputs, text_inputs, text_labels, training_images, img_to_caps, word_to_id, id_to_word):
    '''
    Trains the decoder model.
    :param model_class: class for the decoder model
    :param model: decoder model
    :param image_inputs: feature vectors
    :param text_inputs: input text
    :param text_labels: output text
    :param training_images: list of training images
    :param word_to_id: vocab dict
    :param id_to_word: reverse vocab dict
    :returns training BLEU scores
    '''
    model.compile(optimizer=model_class.optimizer, loss=model_class.loss)
    bleu_callback = BleuCallback(model, image_inputs, text_inputs, training_images, img_to_caps, word_to_id, id_to_word)
    callbacks=[bleu_callback]
    history = model.fit(x=[image_inputs, text_inputs], y=text_labels, batch_size=model_class.batch_size, epochs=50, validation_split=0.2, callbacks=callbacks)
    plot_loss(history)
    # model.save(MODEL_PATH)
    return bleu_callback.get_data()



def test(model, img_to_feats, testing_images, word2id, id2word, img2caps):
    '''
    Tests the decoder model using greedy search.
    :param model: trained decoder model
    :param img_to_feats: map from image to feature vector
    :param testing_images: list of test images
    :param word2id: vocab dict
    :param id2word: reverse vocab dict
    :param img2caps: map from image to list of true captions
    :returns testing BLEU scores
    '''
    img2prediction = {}
    for i, img in enumerate(testing_images):
        print(i)
        feat_vector = img_to_feats[img]
        predicted_caption = predict_caption(model, feat_vector, word2id, id2word)
        img2prediction[img] = predicted_caption
    
    b1, b2, b3, b4 = bleu_score(testing_images, img2caps, img2prediction)

    print('b1: ' + str(b1), 'b2: '+ str(b2), 'b3: ' + str(b3), 'b4: ' + str(b4))

    # with open(PREDICTION_FILE, 'wb') as file:
    #     pickle.dump(img2prediction, file)
    
    return [b1, b2, b3, b4]
    

def predict_caption(model, image_feats, word_to_id, id_to_word):
    '''
    Given a trained model and an encoded image features, returns a predicted caption.
    :param model: trained decoder model
    :param image_feats: img -> feature vector that encodes an image
    :param word_to_id: vocab dict
    :param id_to_word: reverse vocab dict
    :returns predicted, detokenized caption
    '''
    cap = [word_to_id[START_TOKEN]]
    for _ in range(MAXLEN):
        padded_cap = tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([cap], maxlen=MAXLEN-1, padding='post'))
        next_word_dist = tf.squeeze(model.predict(x=[image_feats, padded_cap]))[-1] # get last hidden state
        next_word = int(tf.argmax(next_word_dist))
        cap.append(next_word)

        if next_word == word_to_id[END_TOKEN]:
            break

    return detokenize(cap, word_to_id, id_to_word)
    


def detokenize(caption, word2id, id2word):
    '''
    Detokenizes a predicted caption into Arabic text.
    :param: caption: tokenized caption
    :param word2id: vocab dict
    :param id2word: reverse vocab dict
    '''
    # remove start, end, and pad tokens from caption
    # print(f'CAPTION: {caption}')
    filtered_cap = list(filter(lambda w: w != word2id[PAD_TOKEN] and w != word2id[START_TOKEN] and w != word2id[END_TOKEN] , caption))
    # print(f'FILTERED CAPTION: {filtered_cap}')

    detoken_caption = ""
    for id in filtered_cap:
        detoken_caption = detoken_caption + SPACE + id2word[id]
    return detoken_caption.lstrip()


class BleuCallback(tf.keras.callbacks.Callback):
    '''
    Callback that prints BLEU scores at the end of each training epoch.
    '''
    def __init__(self, model, image_feats, input_text, train_imgs, img2caps, word2id, id2word):
        self.model = model
        self.image_feats = image_feats
        self.input_text = input_text
        self.train_imgs = train_imgs
        self.img2caps = img2caps
        self.word2id = word2id
        self.id2word = id2word
    
    def on_train_begin(self, logs={}):
        self.bleu_list = []

    def on_epoch_end(self, epoch, logs=None):
        '''
        Randomly samples 100 images in the training set for which to generate BLEU scores.
        '''
        # generating random sample of images
        rng = np.random.default_rng()
        batch_image_inds = rng.choice(len(self.train_imgs), size=100, replace=False) # 100 random image indicies we will test on 
        batch_image_feats = tf.gather(self.image_feats, indices=batch_image_inds*3)
        batch_input_text = tf.gather(self.input_text, indices=batch_image_inds*3)

        model_prediction = self.model.predict(x=[batch_image_feats, batch_input_text])
        seq_pred = np.argmax(model_prediction, axis=2) # 100 x 19

        batch_images = [self.train_imgs[i] for i in batch_image_inds]

        img2prediction = {}
        
        for img, seq in zip(batch_images, seq_pred):
            pred_cap = detokenize(list(seq), self.word2id, self.id2word)
            img2prediction[img] = pred_cap
        
        b1, b2, b3, b4 = bleu_score(batch_images, self.img2caps, img2prediction)
        print('b1: ' + str(b1), 'b2: '+ str(b2), 'b3: ' + str(b3), 'b4: ' + str(b4))
        self.bleu_list = [b1, b2, b3, b4]
    
    def get_data(self):
        return self.bleu_list


# def verify_caption_ordering(example_image, img_to_token_caps, img_to_caps, id_to_word):
#     '''
#     Verifies that the words in the captions are being processed in the correct order, since
#     Arabic is printed in a different direction than English.
#     '''
#     example_token_cap = img_to_token_caps[example_image][0]
#     example_cap = img_to_caps[example_image][0]
        
#     print(f'IMAGE: {example_image}')
#     show_image(file=example_image)
#     print(f'TOKEN CAPTION: {example_token_cap}')
#     print(f'FIRST WORD: {id_to_word[example_token_cap[1]]}')
#     print(f'CAPTION: {example_cap}')

#     with open('readme.txt', 'w') as f:
#         f.write(example_cap)



if __name__ == "__main__":
    all_imgs = os.listdir(IMAGE_DIR)
    test_imgs = get_image_list(TEST_IMGS_FILE)
    train_imgs = [img for img in all_imgs if img not in test_imgs]

    vocab, img2tokenizedcaps, img2caps = get_data(DATA_FILE, all_imgs)

    reverse_vocab = {} # maps id to word
    for word, id in vocab.items():
        reverse_vocab[id] = word
    
    # verify_caption_ordering(train_imgs[1], img2tokenizedcaps, img2caps, reverse_vocab)

    img2features = get_features(all_imgs)
    
    # Itrain, Xtrain, and Ytrain are in the order of train_imgs
    Itrain, Xtrain, Ytrain = prep_data(img2tokenizedcaps, img2features, train_imgs)

    print(f'Training on {len(Xtrain)} examples...')
    print(f'Training on {len(train_imgs)} examples...')
    print(f'Testing on {len(test_imgs)} images.')
    
    # try:
    #     decoder = tf.keras.models.load_model(MODEL_PATH)
    # except:

    decoder_instance = Decoder(vocab_size=len(vocab))
    decoder = decoder_instance.get_model()
    training_bleu = train(decoder_instance, decoder, Itrain, Xtrain, Ytrain, train_imgs, img2caps, vocab, reverse_vocab)
    
    testing_bleu = test(decoder, img2features, test_imgs, vocab, reverse_vocab, img2caps)
    visualize_accuracy(training_bleu, testing_bleu)
