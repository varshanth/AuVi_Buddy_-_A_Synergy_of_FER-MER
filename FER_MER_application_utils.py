import random
import re
import os
import pickle


_emotion_reaction_switch_config = {
        # Reaction to Song : # Action
        'Anger' : 'Switch',
        'Surprise' : 'Retain',
        'Sadness' : 'Switch',
        'Disgust' : 'Switch',
        'Happiness' : 'Retain',
        'Fear' : 'Retain'
        }

_fer_mer_integration_config = {
        '_emotion_sequence_dir' : './emotion_sequence',
        '_reaction_sequence_dir' : './reaction_sequence',
        '_playlist_spectro_path' : './playlist/spectrograms',
        '_playlist_song_path' : './playlist/songs',
        '_em_songpool_cache_pkl' : './playlist/em_songpool_cache.pkl',
        '_reaction_analysis_delay_in_sec' : 3
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


def process_playlist_and_return_song_to_songpath(
        playlist_dir = _fer_mer_integration_config['_playlist_song_path']):
    '''
    Input 1: User Playlist directory
             (Default taken from FER MER Integration Config)
    Purpose: Sanity check playlist directory and return the songs and path of
             each song as a dictionary keyed by the song file name
    Output: {SongFileName : PathOfSong}
    '''
    if not os.path.exists(playlist_dir):
        raise Exception('Playlist Path Invalid')
    file_name_re = re.compile('(.*)\..*)')
    song_path_dic = {}
    for song_file in os.listdir(playlist_dir):
        song_name = file_name_re.findall(song_file)[0]
        song_path_dic[song_name] = playlist_dir + '/' + song_file
    return song_path_dic