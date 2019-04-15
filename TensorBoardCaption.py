'''
File to define the customized TensorBoardCaption for training
'''

import io

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
from keras.preprocessing.text import Tokenizer
from PIL import Image, ImageDraw, ImageFont

from preprocessing.image import extract_features, load_images_as_arrays
from NIC import text_emb_lstm, image_dense_lstm, unit_size
from evaluate import beam_search

class TensorBoardCaption(Callback):

    def __init__(self, tokenizer, 
                       vocab_size,
                       max_len, 
                       beam_width = 5, 
                       alpha = 0.7,
                       log_dir = './logs/captions', 
                       feed_pics_dir = './eval', 
                       model_params_dir = './model-params'):
        super(TensorBoardCaption, self).__init__()

        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.beam_width = beam_width
        self.alpha = alpha

        self.log_dir = log_dir
        self.current_model_weigths_dir = model_params_dir + '/tensor_board_caption_weigths.h5'
        self.images = load_images_as_arrays(feed_pics_dir)

        self.image_features = extract_features(feed_pics_dir)

        self.writer = tf.summary.FileWriter(log_dir)
        self.font_tyle = ImageFont.truetype('c:/windows/fonts/Arial.ttf', size = 20)
        self.font_color = (116, 0, 0) # or Red (255, 0, 0)

        print('Tensor board caption is ready ...')


    def on_epoch_end(self, epoch, logs={}):

        self.model.save_weights(self.current_model_weigths_dir)

        # prepare inference model
        NIC_text_emb_lstm = text_emb_lstm(self.vocab_size)
        NIC_text_emb_lstm.load_weights(self.current_model_weigths_dir, by_name = True, skip_mismatch=True)
        NIC_image_dense_lstm = image_dense_lstm()
        NIC_image_dense_lstm.load_weights(self.current_model_weigths_dir, by_name = True, skip_mismatch=True)

        summary_str = []
        for id, image_array in self.images.items():
            fidx = self.image_features['ids'].index(id)
            a0, c0 = NIC_image_dense_lstm.predict([self.image_features['features'][fidx, :].reshape(1, -1), np.zeros([1, unit_size]), np.zeros([1, unit_size])])
            res = beam_search(NIC_text_emb_lstm, a0.reshape(1,-1), c0.reshape(1,-1), self.tokenizer, self.beam_width, self.max_len, self.alpha)
            best_idx = np.argmax(res['scores'])
            caption = self.tokenizer.sequences_to_texts([res['routes'][best_idx]])[0]

            summary_str.append(tf.Summary.Value(tag= id, image= self.make_image(image_array, caption)))

        self.writer.add_summary(tf.Summary(value = summary_str), epoch)
        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()


    def make_image(self, tensor, caption):
        """
        Convert an numpy representation image to Image protobuf and add caption.
        modified from https://github.com/lanpa/tensorboard-pytorch/
        """
        height, width, channel = tensor.shape
        image = Image.fromarray(tensor)
        
        ImageDraw.Draw(image).multiline_text(
            xy = (0, 0),  # Coordinates
            text = self.__caption_format(caption),  # Text
            fill = self.font_color,
            font = self.font_tyle
        )
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height,
                                width=width,
                                colorspace=channel,
                                encoded_image_string=image_string)


    def __caption_format(self, caption, max_length = 7):

        words = caption.split(' ')
        multiline_words = []
        for i in range(len(words)):
            multiline_words.append(words[i])
            if i!= 0 and i % max_length == 0:
                multiline_words[-1] = '\n' + multiline_words[-1]
        
        return ' '.join(multiline_words)
