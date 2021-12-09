import tensorflow as tf
from tensorflow.keras.preprocessing import image
import re

VGG16_IMG_SIZE = (224,224)

def show_image(file):
    import os
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(ROOT, "data/Flicker8k_Dataset", file)
    img = mpimg.imread(path)
    imgplot = plt.imshow(img)
    plt.show()

def preprocess_image(img):
    '''
    Preprocesses an image according to VGG16 preprocessing.
    :param image: image to be processed
    :return image: VGG16 processed image
    '''
    img = image.img_to_array(image.load_img(img, target_size=VGG16_IMG_SIZE))
    return tf.keras.applications.vgg16.preprocess_input(img)



START_TOKEN = "<START>"
END_TOKEN = "<END>"
PAD_TOKEN = "<PAD>"
SPACE = " "
MAXLEN = 20


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
    # print(caption)
    # caption = ''.join([char for char in caption if char not in punctuations][::-1])
    caption = ''.join([char for char in caption if char not in punctuations])
    # print(caption)

    # remove non-Arabic letters
    caption = re.sub(r'[a-zA-Z]+', '', caption)

    # remove numeric words
    caption = ' '.join([word for word in caption.split() if word.isalpha()])

    # add start and end tokens
    caption = START_TOKEN + SPACE + caption + SPACE + END_TOKEN
    
    return caption


def make_caption_dict(raw_data):
    '''
    Creates a dictionary mapping each image to a list of its captions.
    :param raw_data: file with image names and captions
    '''
    output = {}

    for line in raw_data:        
        # get the image name and caption
        if len(line.split('\t')) < 2:
            # TODO: the one line that errors is a leftover caption from the last line
            # we need to append this leftover caption to the caption from the last line
            continue
        image_name, caption = line.split('\t')

        # clean the caption
        caption = clean_caption(caption)

        # remove the last 2 characters (which denote caption number) from the end of the image name
        image_name = image_name[:-2]

        # map the image to its captions
        if image_name in output:
            output[image_name].append(caption)
        else:
            output[image_name] = [caption]

    return output


def make_vocab_dict(capt_dict, all_imgs):
    '''
    Takes in the output from make_caption_dict() and constructs a dictionary 
    that maps each unique word in all of these captions to a word id.

    :param output: dictionary that maps image number/name to a list of its captions
    :returns vocab_dict: vocab dictionary which maps each unique word in the captions to an id
    '''

    vocab_dict = {}
    all_words = []

    for image in all_imgs:
        capt_list = capt_dict[image]
        for caption in capt_list:
            words_array = caption.strip().split()
            for word in words_array:
                if word not in all_words:
                    all_words.append(word)

    
    vocab_dict = {w : i + 1 for i, w in enumerate(all_words)} # add 1 to all inidices to reserve index 0 for PAD_TOKEN
    vocab_dict[PAD_TOKEN] = 0
    return vocab_dict


def tokenize_pad_captions(image_to_caps, vocab):
    '''
    Tokenizes all captions.
    :param output: dictionary mapping each image to its captions (the output from make_caption_dict())
    :param vocab: vocabulary dictionary
    :returns dictionary mapping each image to its tokenized captions
    '''
    img_to_processed_caps = {}
    for image in image_to_caps:
        tokenized_captions = []
        for caption in image_to_caps[image]:
            tokenized_caption = []
            for word in caption.split():
                tokenized_caption.append(vocab[word])
            tokenized_captions.append(tokenized_caption)
        padded_captions = tf.keras.preprocessing.sequence.pad_sequences(tokenized_captions, maxlen=MAXLEN, padding='post')
        img_to_processed_caps[image] = padded_captions
    return img_to_processed_caps


def get_data(file, all_imgs):
    '''
    Returns 1) arabic vocabulary, 2) dict mapping image to list of its tokenized captions, 3) index of the pad token
    '''
    with open(file) as f:
        data = f.read().splitlines()
    
    img2caps = make_caption_dict(data)
    vocabulary = make_vocab_dict(img2caps, all_imgs)
    img2tokenizedcaps = tokenize_pad_captions(img2caps, vocabulary)

    return vocabulary, img2tokenizedcaps, img2caps
