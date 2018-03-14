from keras.models import load_model
from keras.utils import to_categorical
from FER_CNN_Train import _model_save_file_path
from FER_MER_data_set_utils import oulu_casia_ds, inference_from_evaluate
import numpy as np
import os

############################## LOAD MODEL #####################################
if not os.path.exists(_model_save_file_path):
    raise Exception('Model file {0} does not exist'.format(
            _model_save_file_path))

model = load_model(_model_save_file_path)

########################## LOAD INFERENCE DATA ################################
oulu_casia_dataset = oulu_casia_ds(dataset_mode = 'modified_expanded')
_oulu_casia_dataset_config = oulu_casia_dataset.get_data_set_config()
_emotion_label_to_idx = _oulu_casia_dataset_config['_emotion_label_to_idx']

_evaluate_directory = './Evaluate'
X_test, y_test = inference_from_evaluate(_evaluate_directory,
                                         _emotion_label_to_idx,
                                         (128,128))
y_test_as_categorical = to_categorical(y_test, len(_emotion_label_to_idx))
y_pred_one_hot = model.predict(X_test)
y_pred_as_label = np.argmax(y_pred_one_hot, axis=1)

print('Mapping: {0}'.format(_emotion_label_to_idx))
print('Predicted: {0}'.format(y_pred_as_label))
print('Actual: {0}'.format(y_test))
print('Score: {0}'.format(model.evaluate(X_test, y_test_as_categorical)))