'''
Module to preprocess filckr8k image data
'''
import numpy as np 
import os
from pickle import dump, load

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model

from PIL import Image

def load_images_as_arrays(directory):

    img_array_dict = {}
    for img_file in os.listdir(directory):
        
        img_path = directory + '/' + img_file
        
        img = Image.open(img_path)
        x = np.array(img)

        img_array_dict[os.path.splitext(img_file)[0]] = x
    
    return img_array_dict


def extract_features(directory):

    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    img_id = []
    img_matrices = []
    for img_file in os.listdir(directory):
        
        img_path = directory + '/' + img_file
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        img_id.append(os.path.splitext(img_file)[0])
        img_matrices.append(x)
    
    img_matrices = np.array(img_matrices)
    assert(len(img_matrices.shape)==4)

    img_features = model.predict(img_matrices, verbose=1)

    return {'ids':img_id, 'features':img_features}


def extract_feature_from_image(file_dir):

    img = image.load_img(file_dir, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    return model.predict(x)


def load_features(dict_dir, dataset_dir, repeat_times = 1):

    assert(repeat_times >= 1)
    
    img_ids = []
    with open(dataset_dir, 'r') as f:
        for line in f.readlines():
            img_ids.append(os.path.splitext(line)[0])
    
    features_dict = load(open(dict_dir, 'rb'))
    dataset_features = []
    for img_id in img_ids:
        fidx = features_dict['ids'].index(img_id)
        dataset_features.append(np.vstack([features_dict['features'][fidx, :]]*repeat_times))

    dataset_features = np.vstack(dataset_features)

    return dataset_features

if __name__ == "__main__":

    # pre-extract image features from Inception Net
    image_directory = './datasets/Flickr8k_Dataset'
    features_dict = extract_features(image_directory)

    dump(features_dict, open('./datasets/features_dict.pkl', 'wb'))
