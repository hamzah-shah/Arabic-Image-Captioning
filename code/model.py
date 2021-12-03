import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16


class EndtoEnd(tf.keras.Model):
    def __init__(self, arabic_vocab_size):
        self.batch_size = 1024
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.encoder = Encoder().encoder
        self.decoder = Decoder(vocab_size=arabic_vocab_size).decoder

    def call(self, img):
        '''
        :param img: image to be encoded
        :return prbs: the probabilities of each word in the vocab
        '''
        return self.decoder(self.encoder(img))

    def loss_function(self):
        pass

    def accuracy(self):
        pass


class Encoder(tf.keras.Model):
    '''
    Encoder portion of the End-to-End captioning model.
    We employ transfer learning for the encoder, utilizing
    the pre-trained VGG16 model.
    '''
    def __init__(self):
        self.encoder = tf.keras.Sequential()

        # load pre-trained VGG16 model, excluding last layer
        vgg16 = VGG16(weights='imagenet')
        for layer in vgg16.layers[:-1]:
            layer.trainable = False
            self.encoder.add(layer)
        
        # manually add last layer
        self.fc3 = tf.keras.layers.Dense(units=256, activation="tanh", name="fc3") # TODO: how to train the weights of this last layer?
        self.encoder.add(self.fc3)

    def call(self, img):
        '''
        :param img: image to be encoded
        :return embedding: a 256 dimensional embedding of the image
        '''
        return self.encoder(img)


class Decoder(tf.keras.Model):
    # TODO: figure out window size
    '''
    Decoder portion of the End-to-End captioning model.
    The decoder is a recurrent network, specifically an LSTM.
    '''
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.embedding_size = 300
        
        self.embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_size)
        self.decoder = tf.keras.layers.LSTM(units=256)
        self.fc = tf.keras.layers.Dense(units=self.vocab_size, activation="softmax")

    def call(self, embedding):
        '''
        :param embedding: image embedding outputted by the encoder
        :return prbs: the probabilities of each word in the vocab
        '''
        decoded = self.decoder(embedding)
        return self.fc(decoded)
