from keras.models import load_model
from FER_CNN_Train import _model_save_file_path
from FER_Dense_Net import latent_rep_layer_name
from FER_MER_data_set_utils import (oulu_casia_ds,
                                    get_image_seq_apply_optical_flow_norm)
from keras.models import Model
import numpy as np
import os


'''
TESTING CODE:
    
oulu_casia_dataset = oulu_casia_ds(dataset_mode = 'modified_expanded',
                                   normalize_mode = 'optical_flow')
_oulu_casia_dataset_config = oulu_casia_dataset.get_data_set_config()
_emotion_label_to_idx = _oulu_casia_dataset_config['_emotion_label_to_idx']
_idx_to_emotion_label = {idx:label for label, idx in
                         _emotion_label_to_idx.items()}
_evaluate_directory = './Evaluate/Anger'
X_test = get_image_seq_apply_optical_flow_norm(_evaluate_directory,
                                                       (128,128))

y_pred_one_hot = model.predict(X_test)
y_pred_as_idx = np.argmax(y_pred_one_hot, axis=1)
y_pred_as_label = np.array(
        [_idx_to_emotion_label[idx] for idx in y_pred_as_idx])
print(y_pred_as_label)
'''


def fer_cnn_get_latent_rep_from_sequences(image_sequences):
    '''
    Input 1: Image Sequences
    Purpose: 1) Load the Saved FER CNN Model
             2) Get output from GAP Layer for each of the image
                for each image sequence
             3) Return the Latent Representation from the images
    Output: Latent representation of the image sequence
    '''
        ############################ LOAD MODEL ###################################
    if not os.path.exists(_model_save_file_path):
        raise Exception('Model file {0} does not exist'.format(
                _model_save_file_path))
    
    model = load_model(_model_save_file_path)
    GAP_Out_Model = Model(inputs = model.input, outputs = model.get_layer(
            latent_rep_layer_name).output)
    latent_X = []
    for img_sequence in image_sequences:
        latent_img_representation = GAP_Out_Model.predict(img_sequence)
        latent_X.append(latent_img_representation)
    latent_X = np.array(latent_X)
    return latent_X


def fer_cnn_get_latent_rep_data_set():
    '''
    Input: None
    Purpose: 1) Load the trained model
             2) Load the sequence OULU CASIA dataset
             3) Derive the GAP Out from FER DENSE NET as latent representation
             4) For each image sequence, derive the latent representation and
                store as the latent representation for that image sequence
             5) Return the dataset
    Output: [Latent Representation of Image Sequences, Labels]
    '''
    ################### LOAD SEQUENTIAL OULU CASIA DATASET ####################
    oulu_casia_sequential_data = oulu_casia_ds(dataset_mode = 'sequence',
                                               shuffle_data = False,
                                               test_set_fraction = 0)
    oulu_casia_sequential_data.labels_to_categorical()
    X, y, not_used_0, not_used_1 = oulu_casia_sequential_data.get_data_set()
    latent_X = fer_cnn_get_latent_rep_from_sequences(X)
    return [latent_X, y]
        


    