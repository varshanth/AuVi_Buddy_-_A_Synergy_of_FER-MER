from FER_LSTM_Inference import (load_an_image_sequence_from_dir,
                                im_sequences_to_emotions)
from MER_LRCN_Inference import spectrograms_to_em_songpool_cache
from FER_MER_data_set_utils import (_deam_ds_config,
                                    _oulu_casia_get_data_set_args)
import vlc
from FER_MER_application_utils import (
        _fer_mer_integration_config,
        load_em_songpool_cache,
        get_random_song_from_choices,
        process_playlist_and_return_song_to_songpath,
        _fer_emotion_to_mer_emotion_mapping,
        _emotion_reaction_switch_config)
import os
import numpy as np
from time import sleep

########################### ONE ITERATION ONLY ################################

########################### LOAD SONG MAPPINGS ################################

# Create Emotion Song Pool Cache if not present
if not os.path.exists(_fer_mer_integration_config['_em_songpool_cache_pkl']):
    spectrograms_to_em_songpool_cache(
            _fer_mer_integration_config['_playlist_spectro_path'],
            _fer_mer_integration_config['_em_songpool_cache_pkl'],
            _deam_ds_config['_image_resolution']
            )
# Load Emotion Song Pool Cache
em_songpool_cache = load_em_songpool_cache(
        _fer_mer_integration_config['_em_songpool_cache_pkl'])

# Load Song to Songpath Mapping
song_to_songpath = process_playlist_and_return_song_to_songpath(
        _fer_mer_integration_config['_playlist_song_path'])


####################### INITIAL EMOTION DETECTION #############################

img_seq = load_an_image_sequence_from_dir(
        _fer_mer_integration_config['_emotion_sequence_dir'],
        _oulu_casia_get_data_set_args['_image_resolution'])
# Pack Image Sequence inside a numpy array to denote a single sequence
img_seq = np.array([img_seq])
initial_emotion_found = im_sequences_to_emotions(img_seq)[0]
print('Initial Emotion Detected: {0}'.format(initial_emotion_found))

########### MAP RECOGNIZED EMOTION TO RECIPROCATION EMOTION ###################

music_emotion_reqd = _fer_emotion_to_mer_emotion_mapping[initial_emotion_found]
print('Mapping to Music Emotion: {0}'.format(music_emotion_reqd))
song_choices_from_playlist = em_songpool_cache[music_emotion_reqd]
random_song_from_choices = get_random_song_from_choices(
        song_choices_from_playlist)
print('Playing Song: {0}'.format(random_song_from_choices))

############## PLAY SONG AND ANALYZE REACTION AFTER DELAY #####################

print('Analyzing Reaction')
song_loaded = vlc.MediaPlayer(song_to_songpath[random_song_from_choices])
song_loaded.play()
sleep(_fer_mer_integration_config['_reaction_analysis_delay_in_sec'])

img_seq = load_an_image_sequence_from_dir(
        _fer_mer_integration_config['_reaction_sequence_dir'],
        _oulu_casia_get_data_set_args['_image_resolution'])
# Pack Image Sequence inside a numpy array to denote a single sequence
img_seq = np.array([img_seq])
reaction_emotion_found = im_sequences_to_emotions(img_seq)[0]
print('Reaction Emotion Found {0}'.format(reaction_emotion_found))

######### SWITCH SONG IF REACTION NEGATIVE ELSE KEEP PLAYING SONG #############
get_emotions_resulting_switch = {
        em for em, action in _emotion_reaction_switch_config.items()
        if action == 'Switch'}

action_required = _emotion_reaction_switch_config[reaction_emotion_found]
if initial_emotion_found in get_emotions_resulting_switch:
    action_required = 'Retain'
    
print('Action Required: {0}'.format(action_required))
if action_required == 'Switch':
    print('Stopping Song')
    song_loaded.stop()
    random_song_from_choices = get_random_song_from_choices(
        song_choices_from_playlist)
    print('Playing Song: {0}'.format(random_song_from_choices))
    song_loaded = vlc.MediaPlayer(song_to_songpath[random_song_from_choices])
    song_loaded.play()
    
###############################################################################
    
    