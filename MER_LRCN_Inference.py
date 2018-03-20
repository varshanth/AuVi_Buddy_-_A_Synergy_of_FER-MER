from MER_LRCN_Train import _model_save_file_path
import os
from keras.models import load_model
from FER_MER_data_set_utils import convert_a_v_vector_to_emotion_possibilities
from FER_MER_application_utils import _fer_emotion_to_mer_emotion_mapping
import numpy as np
import re
from PIL import Image
import pickle


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
    emotion_sets = np.array(emotion_sets)
    return emotion_sets


def load_spectrograms_from_dir(spectro_dir, resolution = None):
    '''
    Input 1: Spectrograms Directory Path
    Input 2: Resolution of Image Desired (Default: Original)
    Purpose: Load spectrograms from directory to numpy arrays
    Output: Dictionary of song names along with the respective numpy array
            representation of the spectrograms
    '''
    if not os.path.exists(spectro_dir):
        raise Exception('Invalid Directory Provided')
    song_spectro_dic = {}
    file_name_re = re.compile('(.*)\..*')
    for spec_file_name in os.listdir(spectro_dir):
        spec = Image.open(spectro_dir + '/' +spec_file_name)
        if resolution:
            spec = spec.resize(resolution)
        spec = np.array(spec)
        song_name = file_name_re.findall(spec_file_name)[0]
        song_spectro_dic[song_name] = spec
    return song_spectro_dic
    

def spectrograms_to_em_songpool_cache(spectro_dir,
                                      em_songpool_cache_pkl_file,
                                      resolution = None):
    '''
    Input 1: Spectrograms Directory Path
    Input 2: Path to the emotions songpool cache pickle file
    Input 3: Resolution of Image Desired (Default: Original)
    Purpose: Generate the emotion songpool cache file containing a mapping of
             possible emotions to the list of possible songs matching each of
             the emotions
    Output: None
    '''
    song_emotions = set(_fer_emotion_to_mer_emotion_mapping.values())
    # Initialize Songpool Cache
    em_songpool_cache = {em : [] for em in song_emotions}
    # Load Song-Spectrogram Dictionary
    song_spectro_dic = load_spectrograms_from_dir(spectro_dir, resolution)
    # Get Songs
    songs = song_spectro_dic.keys()
    # Get spectrograms
    song_spectro_dic_vals = list(song_spectro_dic.values())
	spectros = np.array(song_spectro_dic_vals)
    # For each spectrogram get the possible emotion set associated with it
    emotion_sets_from_spectros = spectrograms_to_emotions(spectros)
    # Construct the Songpool Cache
    for song, emotion_set in zip(songs, emotion_sets_from_spectros):
        for emotion in emotion_set:
            em_songpool_cache[emotion].append(song)
    # Write the Song Pool Cache to the pickle file
    with open(em_songpool_cache_pkl_file, "wb") as em_cache:
        pickle.dump(em_songpool_cache, em_cache)
