import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GRU


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

    def call(self, imgs):
        '''
        :param imgs: batch of images to be encoded
        :return embedding: a 4096 dimensional embedding of each image
        '''
        return self.encoder(imgs)


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

        self.input_image = Input(shape=(4096,), batch_size=self.batch_size) # 4096 = output of VGG16 last layer
        self.input_text = Input(shape=(20,), batch_size=self.batch_size) # 20 = maxlen


        self.image_embeddings = Dense(units=256, activation="tanh") (self.input_image)
        print(f'Image embedding shape: {tf.shape(self.image_embeddings)}')


        self.text_embeddings = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_size) (self.input_text)
        print(f'Text embedding shape: {tf.shape(self.text_embeddings)}')
        self.gru = GRU(units=256, stateful=True)
        self.gru_outputs = self.gru(self.text_embeddings, initial_state=self.image_embeddings)
        print(f'LSTM outputs: {tf.shape(self.gru_outputs)}')
        self.output = Dense(units=vocab_size, activation="softmax") (self.gru_outputs)
        print(f'Dense outputs: {tf.shape(self.output)}')

        self.decoder = tf.keras.Model(inputs=[self.input_image, self.input_text], outputs=self.output)
