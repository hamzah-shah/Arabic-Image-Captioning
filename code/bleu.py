import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import tensorflow as tf
import matplotlib.pyplot as plt

START_TOKEN = "<START>"
END_TOKEN = "<END>"
SPACE = " "
MAXLEN = 20


def bleu_score(image_list, img2caps, img2prediction):
  '''
  Calculates the mean BLEU score (i.e. accuracy) for set of predicted captions.
  :param image_list: the list of images that were captioned
  :param img2caps: dict mapping an image name to a list of its ground-truth captions
  :param img2prediction: dict mapping an image to the caption predicted for it by the model
  returns avg bleu 1-4 scores
  '''
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
  '''
  Visualizes the BLEU scores for the training and testing data.
  :param training_scores: list of BLEU scores for training
  :param testing_scores: list of BLEU scores for testing
  '''
  df = pd.DataFrame({
      'Gram': ['b1', 'b2', 'b3', 'b4'],
      'Train': training_scores,
      'Test': testing_scores
  })
  
  df.plot(x="Gram", y=["Train", "Test"], kind="bar", xlabel="bleu metric", ylabel="score")
  plt.savefig('bleu_score.png')
  # plt.show()
