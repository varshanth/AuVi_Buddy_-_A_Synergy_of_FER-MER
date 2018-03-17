from FER_LSTM_Train import _model_save_file_path
import os
from keras.models import load_model
from FER_CNN_Inference import fer_cnn_get_latent_rep_from_sequences
from PIL import Image
import numpy as np

def _fer_lstm_inference(test_sequences):
    '''
    Input: Test Sequences of Latent Representations of Image Sequences
    Purpose: 1) Load the trained model
             2) Run the model and predict the output
    Output: Categorical Predictions
    '''
    ############################ LOAD MODEL ###################################
    if not os.path.exists(_model_save_file_path):
        raise Exception('Model file {0} does not exist'.format(
                _model_save_file_path))
        
    model = load_model(_model_save_file_path)
    predictions = model.predict(test_sequences)
    return predictions


def fer_inference_pipeline(test_image_sequences):
    '''
    Input: Test Sequence of Images
    Purpose: 1) Run FER CNN Inference and get latent representation sequence
             2) Run FER LSTM Inference and get the categorical predictions
    Output: Categorical predictions
    '''
    latent_X = fer_cnn_get_latent_rep_from_sequences(test_image_sequences)
    predictions = _fer_lstm_inference(latent_X)
    return predictions


def load_an_image_sequence_from_dir(sequence_dir,
                                    resolution = None):
    '''
    Input 1: Image Sequence Directory Path
    Input 2: Resolution of Image Desired (Default: Original)
    Purpose: Load image sequence from directory to a numpy array
    Output: Ordered numpy array of images representing sequences
    '''
    if not os.path.exists(sequence_dir):
        raise Exception('Invalid Directory Provided')
    img_sequence = []
    for img_name in sorted(os.listdir(sequence_dir)):
        img = Image.open(sequence_dir + '/' +img_name)
        if resolution:
            img = img.reshape(resolution)
        img = np.array(img)
        img_sequence.append(img)
    img_sequence = np.array(img_sequence)
    return img_sequence

'''
# Sample Code:

img_seq = load_an_image_sequence_from_dir('./Happiness', (128, 128))
# Pack Image Sequence inside a numpy array to denote a single sequence
img_seq = np.array([img_seq])

categ_predictions = fer_inference_pipeline(img_seq)
    
'''
    