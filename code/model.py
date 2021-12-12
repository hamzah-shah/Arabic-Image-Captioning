import tensorflow as tf
import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Embedding, Dropout, Dense, GRU

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

VGG16_EMBEDDING_SIZE = 4096
MAXLEN = 20

class Encoder(tf.keras.Model):
    '''
    Encoder portion of the End-to-End captioning model.
    We employ transfer learning for the encoder, utilizing
    the pre-trained VGG16 model.
    '''
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.encoder = tf.keras.Sequential()
        # load pre-trained VGG16 model, excluding classifier layer
        vgg16 = VGG16(weights='imagenet')
        for layer in vgg16.layers[:-1]:
            layer.trainable = False
            self.encoder.add(layer)

    def call(self, img):
        '''
        :param img: an images to be encoded
        :return embedding: a 4096 dimensional embedding of the image
        '''
        assert(img.shape == (224,224,3))
        return self.encoder(tf.expand_dims(img, axis=0))


class Decoder():
    '''
    Decoder portion of the End-to-End captioning model.
    The decoder is a recurrent network, specifically a GRU.
    '''
    def __init__(self, vocab_size):
        self.batch_size = 512
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        
        self.vocab_size = vocab_size
        self.image_embedding_size = 256
        self.text_embedding_size = 256

        self.input_image = Input(shape=(VGG16_EMBEDDING_SIZE,))
        self.input_text = Input(shape=(MAXLEN-1,))

        self.image_dense = Dense(units=self.image_embedding_size, activation="tanh")
        # mask zero to mask the padding
        self.text_embedding = Embedding(input_dim=self.vocab_size, output_dim=self.text_embedding_size, input_length=MAXLEN-1, mask_zero=True) 
        self.gru = GRU(units=256,return_sequences=True, return_state=True)
        self.text_dense1 = Dense(units=256, activation="relu")
        self.dropout = Dropout(0.3)
        self.text_dense2 = Dense(units=vocab_size, activation="softmax")
    
    def get_model(self):
        image_embeddings = self.image_dense (self.input_image) 

        text_embeddings = self.text_embedding (self.input_text) 
        gru_outputs, _ = self.gru (text_embeddings, initial_state=image_embeddings) 
        fc1_outputs = self.text_dense1(gru_outputs)
        drop = self.dropout(fc1_outputs)
        fc2_outputs = self.text_dense2 (drop)

        decoder = tf.keras.Model(inputs=[self.input_image, self.input_text], outputs=fc2_outputs)
        decoder.summary()

        return decoder
