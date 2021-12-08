import nltk
import tensorflow as tf
import matplotlib as plt
# from nltk import corpus_bleu
# Here's what we need for implementing accuracy function (BLEU):
# a list of predicted captions from testing data
# a list of actual captions from testing data

def bleu_function(actual_captions, predicted_captions):
  references = [[caption.strip().split() for caption in captions] for captions in actual_captions]
  hypotheses = [caption.strip().split() for caption in predicted_captions]
  one_gram = nltk.corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
  two_gram = nltk.corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
  three_gram = nltk.corpus_bleu(references, hypotheses, weights=(float(1/3), float(1/3), float(1/3), 0))
  four_gram = nltk.corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

  return one_gram, two_gram, three_gram, four_gram


def make_features_dict(x_train, y_train):
  output = {}
  for i in range(len(x_train)):
    if x_train[i] in output:
      output[x_train[i]].append(y_train[i])
    else:
      output[x_train[i]] = [y_train[i]]

  return output


def bleu_score(model, x_train, y_train, caption_dict):
  features_dict = make_features_dict(x_train, y_train)

  for key in features_dict:
    references = [word for value in caption_dict[key] for word in value.strip().split()] # list of the words that exist in 3 original captions

    prediction = None # we need to use prediction funciton from model.py to predict a caption

    one_gram, two_gram, three_gram, four_gram = [], [], [], []
    one_gram.append(nltk.sentence_bleu(references, prediction, weights=(1, 0, 0, 0)))
    two_gram.append(nltk.sentence_bleu(references, prediction, weights=(0.5, 0.5, 0, 0)))
    three_gram.append(nltk.sentence_bleu(references, prediction, weights=(float(1/3), float(1/3), float(1/3), 0)))
    four_gram.append(nltk.sentence_bleu(references, prediction, weights=(0.25, 0.25, 0.25, 0.25)))

  return tf.mean(one_gram), tf.mean(two_gram), tf.mean(three_gram), tf.mean(four_gram)

def visualize_accuracy(train_data, test_data):
  left = [1, 2, 3, 4]
  tick_label = ['one_gram', 'two_gram', 'three_gram', 'four_gram']
  plt.bar(left, train_data, tick_label = tick_label, width = 0.8, color = ['green', 'red'])

  plt.bar(left, test_data, tick_label = tick_label, width = 0.8, color = ['green', 'red'])
  plt.show()

