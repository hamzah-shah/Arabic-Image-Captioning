import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import tensorflow as tf
import matplotlib.pyplot as plt
# from nltk import corpus_bleu
START_TOKEN = "<START>"
END_TOKEN = "<END>"
SPACE = " "
MAXLEN = 20


# def bleu_function(actual_captions, predicted_captions):
#   references = [[caption.strip().split() for caption in captions] for captions in actual_captions]
#   hypotheses = [caption.strip().split() for caption in predicted_captions]
#   one_gram = nltk.corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
#   two_gram = nltk.corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
#   three_gram = nltk.corpus_bleu(references, hypotheses, weights=(float(1/3), float(1/3), float(1/3), 0))
#   four_gram = nltk.corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

#   return one_gram, two_gram, three_gram, four_gram


# def make_features_dict(x_train, y_train):
#   output = {}
#   for i in range(len(x_train)):
#     if x_train[i] in output:
#       output[x_train[i]].append(y_train[i])
#     else:
#       output[x_train[i]] = [y_train[i]]

#   return output


def bleu_score(image_list, img2caps, img2prediction):
  for image in image_list:
    references = [word for value in img2caps[image] for word in value.strip().split() if word != START_TOKEN or word != END_TOKEN] # list of the words that exist in 3 original captions
    prediction = img2prediction[image] # we need to use prediction funciton from model.py to predict a caption

    one_gram, two_gram, three_gram, four_gram = [], [], [], []
 
    one_gram.append(sentence_bleu(references, prediction, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1))
    two_gram.append(sentence_bleu(references, prediction, weights=(0.5, 0.5, 0, 0), smoothing_function=SmoothingFunction().method1))
    three_gram.append(sentence_bleu(references, prediction, weights=(float(1/3), float(1/3), float(1/3), 0), smoothing_function=SmoothingFunction().method1))
    four_gram.append(sentence_bleu(references, prediction, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1))

  return float(tf.math.reduce_mean(one_gram)), float(tf.math.reduce_mean(two_gram)), float(tf.math.reduce_mean(three_gram)), float(tf.math.reduce_mean(four_gram))

def visualize_accuracy(training_scores, testing_scores):
    # creating dataframe
    df = pd.DataFrame({
        'Gram': ['b1', 'b2', 'b3', 'b4'],
        'Train': training_scores,
        'Test': testing_scores
    })
  
    # plotting graph
    df.plot(x="Gram", y=["Train", "Test"], kind="bar", xlabel="bleu metric", ylabel="score")
    plt.show()

