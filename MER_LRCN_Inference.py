from MER_LRCN_Train import _model_save_file_path
import os
from keras.models import load_model
from FER_MER_data_set_utils import convert_a_v_vector_to_emotion_possibilities
import numpy as np


def _mer_lrcn_inference(test_spectrograms):
    '''
    Input: Test Spectrograms as numpy arrays
    Purpose: 1) Load the trained model
             2) Run the model and predict the output
    Output: [Valence Vectors, Arousal Vectors]
    '''
    if not os.path.exists(_model_save_file_path):
        raise Exception('Model file {0} does not exist'.format(
                _model_save_file_path))
        
    model = load_model(_model_save_file_path)
    predictions = model.predict(test_spectrograms)
    valence_preds = []
    arousal_preds = []
    for prediction in predictions:
        valence = prediction[:60]
        arousal = prediction[60:]
        valence_preds.append(valence)
        arousal_preds.append(arousal)
    valence_preds = np.array(valence_preds)
    arousal_preds = np.array(arousal_preds)
    return [valence_preds, arousal_preds]


def spectrograms_to_emotions(spectrograms):
    '''
    Input: Spectrograms as numpy arrays
    Purpose: Run MER LRCN Inference on the Spectrograms and return the emotion
             label associated with each spectrogram
    Output: Return a numpy array of possible emotion labels associated with the
            respective spectrograms
    '''
    valence_preds, arousal_preds = _mer_lrcn_inference(spectrograms)
    emotion_sets = []
    for arousal_pred, valence_pred in zip(arousal_preds, valence_preds):
        emotion_set = convert_a_v_vector_to_emotion_possibilities(arousal_pred,
                                                                  valence_pred)
        emotion_sets.append(emotion_set)
    emotion_set = np.array(emotion_set)
    return emotion_set
        
    
