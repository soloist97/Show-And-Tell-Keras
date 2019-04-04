'''
File to train the NIC model, based on the paper:

https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf
'''

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import plot_model

from NIC import model
from preprocessing.text import create_tokenizer
from utils import batch_generator

from TensorBoardCaption import TensorBoardCaption


def training(dirs_dict, lr, decay, reg, batch_size, epochs, max_len, initial_epoch, previous_model = None):

    dict_dir = dirs_dict['dict_dir']
    token_dir = dirs_dict['token_dir']
    train_dir = dirs_dict['train_dir']
    dev_dir = dirs_dict['dev_dir']
    params_dir = dirs_dict['params_dir']

    # Use Tokenizer to create vocabulary
    tokenizer = create_tokenizer(train_dir, token_dir, start_end = True)
    
    # Progressive loading
    # if batch size of training set is 30 and total 30000 sentences, then 1000 steps.
    # if batch size of dev set is 50 and total 5000 sentences, then 100 steps.
    generator_train = batch_generator(batch_size, max_len, tokenizer, dict_dir, train_dir, token_dir)
    generator_dev = batch_generator(50, max_len, tokenizer, dict_dir, dev_dir, token_dir)

    vocab_size = tokenizer.num_words or (len(tokenizer.word_index)+1)

    # Define NIC model structure
    NIC_model = model(vocab_size, max_len, reg)

    if not previous_model:
        NIC_model.summary()
        plot_model(NIC_model, to_file='./model.png',show_shapes=True)
    else:
        NIC_model.load_weights(model_dir, by_name = True, skip_mismatch=True)

    # Define checkpoint callback
    file_path = params_dir + '/model-ep{epoch:03d}-loss{loss:.4f}-val_loss{val_loss:.4f}.h5'
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_weights_only = True, period=1)
    tbc = TensorBoardCaption(tokenizer, vocab_size, max_len, log_dir = './logs', 
                            feed_pics_dir = './put-your-image-here',
                            model_params_dir = params_dir)


    # Compile the model
    NIC_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr = lr, decay=decay), metrics=['accuracy'])

    # train
    NIC_model.fit_generator(generator_train, steps_per_epoch=30000//batch_size, epochs=epochs, verbose=2, 
                            callbacks=[checkpoint, tbc],
                            validation_data = generator_dev, validation_steps = 100, initial_epoch = initial_epoch)


if __name__ == "__main__":

    dict_dir = './datasets/features_dict.pkl'
    train_dir = './datasets/Flickr8k_text/Flickr_8k.trainImages.txt'
    dev_dir = './datasets/Flickr8k_text/Flickr_8k.devImages.txt'
    token_dir = './datasets/Flickr8k_text/Flickr8k.token.txt'
    # where to put the model weigths
    params_dir = './model-params'

    dirs_dict={'dict_dir':dict_dir, 'train_dir':train_dir, 'dev_dir':dev_dir, 
                'token_dir':token_dir, 'params_dir':params_dir}
    
    training(dirs_dict, lr=0.001, decay=0., reg = 1e-4, batch_size = 120, epochs = 2, 
             max_len = 24, initial_epoch = 0, previous_model = None)
