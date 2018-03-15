from keras.models import load_model
from FER_CNN_Train import _model_save_file_path
from FER_MER_data_set_utils import (oulu_casia_ds,
                                    get_image_seq_apply_optical_flow_norm)
import numpy as np
import os

############################## LOAD MODEL #####################################
if not os.path.exists(_model_save_file_path):
    raise Exception('Model file {0} does not exist'.format(
            _model_save_file_path))

model = load_model(_model_save_file_path)
'''
Insert Code to Get Model Last Layer Here
'''
########################## LOAD INFERENCE DATA ################################
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

################### LOAD SEQUENTIAL OULU CASIA DATASET ########################
oulu_casia_sequential_data = oulu_casia_ds(dataset_mode = 'sequence',
                                           shuffle_data = False,
                                           test_set_fraction = 0)

X, y, not_used_0, not_used_1 = oulu_casia_sequential_data.get_data_set()

latent_X = []
for img_sequence in X:
    latent_sequence = []
    for img in img_sequence:
        '''
        latent_img_representation = GAP_Out_Model(np.array([img]))
        '''
        latent_sequence.append(latent_img_representation)
    latent_sequence = np.array(latent_sequence)
    latent_X.append(latent_sequence)
latent_X = np.array(latent_X)
        
    