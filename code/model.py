import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense


class Encoder(tf.keras.Model):
    '''
    Encoder portion of the End-to-End captioning model.
    We employ transfer learning for the encoder, utilizing
    the pre-trained VGG16 model.
    '''
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.encoder = tf.keras.Sequential()
        # load pre-trained VGG16 model, excluding last layer
        vgg16 = VGG16(weights='imagenet')
        for layer in vgg16.layers[:-1]:
            layer.trainable = False
            self.encoder.add(layer)

    def call(self, img):
        '''
        :param img: image to be encoded
        :return embedding: a 256 dimensional embedding of the image
        '''
        return self.encoder(img)


class Decoder():
    '''
    Decoder portion of the End-to-End captioning model.
    The decoder is a recurrent network, specifically an LSTM.
    '''
    def __init__(self, vocab_size):
        self.batch_size = 1024
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        
        self.vocab_size = vocab_size
        self.embedding_size = 300

        self.input_image = Input((4096,)) # 4096 = output of VGG16 last layer
        self.input_text = Input((20,)) # 20 = maxlen


        self.image_embeddings = Dense(units=256, activation="tanh") (self.input_image)


        self.text_embeddings = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_size) (self.input_text)
        self.lstm = LSTM(units=256, stateful=True)
        self.lstm_outputs = self.lstm(self.text_embeddings, initial_state=self.image_embeddings)
        self.output = Dense(units=vocab_size, activation="softmax") (self.lstm_outputs)

        self.decoder = tf.keras.Model(inputs=[self.input_image, self.input_text], outputs=self.output)
