# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import preprocess_input
import os
import re



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



# path = "/Users/kanzaazhar/Desktop/cs147/Arabic-Image-Captioning/data/Flickr8k_text/Flickr8k.arabic.full.txt"
path = "/Users/khosrow/fall2021/dl/Arabic-Image-Captioning/data/Flickr8k_text/Flickr8k.arabic.full.txt"
with open(path) as f:
    imgs = f.read().splitlines()


def clean_caption(caption):
    # print(caption)
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
    caption = ''.join([char for char in caption if char not in punctuations][::-1])
    # print(caption)

    # remove non-Arabic letters
    caption = re.sub(r'[a-zA-Z]+', '', caption)

    # remove numeric words
    caption = ' '.join([word for word in caption.split() if word.isalpha()])

    # add start and end tokens
    caption = "<start>" + ' ' + caption + ' ' + "<end>"

    # pad short captions with special token to ensure similar length captions
    # TODO

    return caption


def make_caption_dict(raw_data):
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

        # map the image to its captions
        if image_name in output:
            output[image_name].append(caption)
        else:
            output[image_name] = [caption]

    return output

"""
Takes in the output from make_caption_dict (which is a dictionary that maps image number/name to a list of its captions),
and constructs a dictionary that maps each unique word in all of these captions to a word id

INPUT: dictionary that maps image number/name to a list of its captions
RETURNS: vocab dictionary which maps each unique word in the captions to an id
"""
def make_vocab_dict(output):
    vocab_dict = {}
    word_set = set()

    for caption_list in output.values():
        for caption in caption_list:
            caption = caption[7:-6]
            words_array = caption.strip().split()
            word_set.update(words_array)

    vocab_dict = {w : i for i, w in enumerate(list(word_set))}
    return vocab_dict


output = make_caption_dict(imgs)
make_vocab_dict(output)

def tokenize_captions(output):
    vocab_dict = make_vocab_dict(output)
    for image in output:
        tokenized_captions = []
        for caption in output[image]:
            tokenized_caption = []
            caption = caption[7:-6]

            for word in caption.split():
                tokenized_caption.append(vocab_dict[word])
            tokenized_captions.append(tokenized_caption)
        output[image] = tokenized_captions
    return output

print(tokenize_captions(output))
                

