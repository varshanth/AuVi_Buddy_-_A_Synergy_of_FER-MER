from FER_LSTM_Inference import (load_an_image_sequence_from_dir,
                                im_sequences_to_emotions)
from FER_MER_data_set_utils import _deam_ds_config
from MER_LRCN_Inference import spectrograms_to_em_songpool_cache
import vlc
import random
import os
import pickle


_emotion_reaction_switch_config = {
        # Reaction to Song : # Action
        'Anger' : 'Switch',
        'Surprise' : 'Switch',
        'Sadness' : 'Switch',
        'Disgust' : 'Switch',
        'Happiness' : 'Retain',
        'Fear' : 'Retain'
        }

_fer_mer_integration_config = {
        '_emotion_sequence_dir' : './emotion_sequence',
        '_reaction_sequence_dir' : './reaction_sequence'
        }


_fer_emotion_to_mer_emotion_mapping = {
        'Happiness' : 'Happiness',
        'Sadness' : 'Sadness',
        'Anger' : 'Anger',
        'Surprise' : 'Surprise',
        'Fear' : 'Fear',
        'Disgust' : 'Disgust'
        }


def get_random_song_from_choices(choices):
    '''
    Input 1: Numpy array or list of song choices
    Purpose: Return a random song from the list of songs
    Output: A random element from the choices list
    '''
    return random.choices(choices)


def load_em_songpool_cache(em_songpool_cache_pkl_file):
    '''
    Input 1: Emotion to Song Pool Cache Pickle File Path
    Purpose: Return the Emotion to Song Pool Cache Dictionary
    Output: Emotion to Song Pool Cache Dictionary
    '''
    if not os.path.exists(em_songpool_cache_pkl_file):
        raise Exception('Song Pool Cache File Path Invalid')
    em_songpool_cache = pickle.load(open(em_songpool_cache_pkl_file, 'rb'))
    return em_songpool_cache