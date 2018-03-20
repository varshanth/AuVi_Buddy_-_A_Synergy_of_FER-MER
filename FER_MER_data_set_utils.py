from oulu_casia_utils import (oulu_casia_get_data_set,
                              oulu_casia_expand_sequences,
                              _emotion_label_to_idx)
import numpy as np
import os
import re
import pandas as pd
from PIL import Image
from keras.utils import to_categorical
from copy import copy
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


################### FER DATA SET: OULU CASIA NIR & VIS ########################

_oulu_casia_ds_mode = {
        'sequence',
        'expanded',
        'modified_expanded'
        }


_oulu_casia_get_data_set_args = {
        '_images_root_path' : './OriginalImg',
        '_max_im_per_seq' : 9,
        '_image_resolution' : (128, 128)
        }


_oulu_casia_config = {
                '_oulu_casia_get_data_set_args' :_oulu_casia_get_data_set_args,
                '_im_per_seq' : _oulu_casia_get_data_set_args[
                        '_max_im_per_seq'],
                '_emotion_label_to_idx' : copy(_emotion_label_to_idx),
                '_emotion_idx_to_label' : {idx:label for label, idx in
                                           _emotion_label_to_idx.items()}
                }
                

_oulu_casia_train_set_data_gen_config = {
                'rotation_range' : 15,
                'width_shift_range' : 0.1,
                'height_shift_range' : 0.1,
                'horizontal_flip' : True
                }


_oulu_casia_test_set_data_gen_config = {
                'rotation_range' : 20,
                'width_shift_range' : 0.2,
                'height_shift_range' : 0.2,
                'horizontal_flip' : True
                }

_oulu_casia_normalize_mode = {
        'none', 
        'optical_flow',
        'normalize_and_center'
        }


class oulu_casia_ds(object):
    
    def __init__(self,
                 dataset_mode = 'sequence',
                 data_set_config = _oulu_casia_config,
                 train_data_gen_config = _oulu_casia_train_set_data_gen_config,
                 test_data_gen_config = _oulu_casia_test_set_data_gen_config,
                 test_set_fraction = 0.1,
                 normalize_mode = 'none',
                 shuffle_data = True):
        '''
        Input 1: OULU CASIA Data Set Mode (Default: sequence)
        Input 2: Data Set Configuration Dictionary (Default in lib)
        Input 3: Training Set Image Data Generator Dictionary (Default in lib)
        Input 4: Testing Set Image Data Generator Dictionary (Default in lib)
        Input 5: Fraction of dataset dedicated for test set (Default: 0.1)
        Input 6: Normalize and Center Images Flag
                 0: None
                 1: Normalize and center
        Input 7: Flag to indicate whether to shuffle data randomly or not
        Purpose: Initialize OULU CASIA Data Set
        Output: None
        '''
        # Sanity Check
        if dataset_mode not in _oulu_casia_ds_mode:
            raise Exception('Invalid Data Set Mode')
        if test_set_fraction > 1 or test_set_fraction < 0:
            raise Exception('Invalid Test Set Fraction')
        if normalize_mode not in _oulu_casia_normalize_mode:
            raise Exception('Invalid Normalization Mode')
        
        # Configure instance
        self._oulu_casia_config = copy(data_set_config)
        self._oulu_casia_train_set_data_gen_config = copy(
                train_data_gen_config)
        self._oulu_casia_test_set_data_gen_config = copy(test_data_gen_config)
        self.dataset_mode = dataset_mode
        self.normalize_mode = normalize_mode
        
        # Get raw dataset
        if dataset_mode == 'sequence':
            X, y = self._get_ds_as_sequence()
        elif dataset_mode == 'expanded':
            X, y = self._get_ds_as_expanded()
        else:
            X, y, new_im_per_seq = self._get_ds_as_modified_expanded()
            self._oulu_casia_config['_im_per_seq'] = new_im_per_seq
        
        # Perform training and test set split
        self.X_train, self.X_test, self.y_train, self.y_test = (
                train_test_split(X, y, test_size = test_set_fraction,
                                 shuffle = shuffle_data))
                
        # Normalize and center images if flag is set
        if normalize_mode == 'normalize_and_center':
            self.normalize_and_center_images()
        
        if dataset_mode != 'sequence':
            # Initialize the ImageDataGenerators
            self.train_datagen = ImageDataGenerator(
                    **self._oulu_casia_train_set_data_gen_config)
            self.train_datagen.fit(self.X_train)
            self.test_datagen = ImageDataGenerator(
                    **self._oulu_casia_test_set_data_gen_config)
            self.test_datagen.fit(self.X_test)

    
    def _get_ds_as_sequence(self):
        '''
        Input 1: Optical Flow Normalization Flag
        Purpose: Wrapper for oulu_casia_get_data_set
        Output: Output of oulu_casia_get_data_set
        '''
        X, y = oulu_casia_get_data_set(
                **self._oulu_casia_config['_oulu_casia_get_data_set_args'])
        if self.normalize_mode == 'optical_flow':
            X = self.optical_flow_norm(X)
        return [X, y]
    
    
    def _get_ds_as_expanded(self):
        '''
        Input: None
        Purpose: Wrapper for oulu_casia_expand_sequences
        Output: Output of oulu_casia_get_data_set
        '''
        X, y = self._get_ds_as_sequence()
        X, y = oulu_casia_expand_sequences(X, y)
        return [X, y]
    
    
    def _get_ds_as_modified_expanded(self):
        '''
        Input: None
        Purpose: Modify the expanded OULU CASIA dataset to extract only some of
                 the images of the sequence and custom label the extracted
                 images
        Output: [Extracted Images, Modified Labels]
        '''
        _max_im_per_seq = self._oulu_casia_config[
                '_oulu_casia_get_data_set_args']['_max_im_per_seq']
        mask = np.array([False for i in range(_max_im_per_seq)])
        # 1st image as "Neutral" emotion
        mask[0] = True
        # Last 5 images representing labelled emotion
        mask[-5:] = True
        new_im_per_seq = len(mask[mask == True])
        X, y = self._get_ds_as_expanded()
        new_images = []
        new_labels = []
        # Add "Neutral" to Emotion Labels
        self.add_custom_emotion_labels(['Neutral'])
        # Modify 1st label to be "Neutral Expression"
        self.modified_labels = {
                # Index : Emotion
                0 : 'Neutral'
                }
        # Go through each sequence, mask the sequence and modify the labels
        for i in range(len(X) // _max_im_per_seq):
            start = i * _max_im_per_seq
            end = start + _max_im_per_seq
            sequence = X[start:end]
            sequence = sequence[mask]
            sequence_labels = y[start:end]
            sequence_labels = sequence_labels[mask]
            for index, modified_label in self.modified_labels.items():
                sequence_labels[index] = modified_label
            new_images.extend(sequence)
            new_labels.extend(sequence_labels)
        new_images = np.array(new_images)
        new_labels = np.array(new_labels)
        return [new_images, new_labels, new_im_per_seq]
        
    
    def add_custom_emotion_labels(self, custom_label_list):
        '''
        Input 1: List of Custom Emotion Labels
        Purpose: Add custom emotion labels onto the label mapping
        Output: None
        '''
        for custom_label in custom_label_list:
            self._oulu_casia_config['_emotion_label_to_idx'][custom_label] = (
                    len(self._oulu_casia_config['_emotion_label_to_idx']))
    
    
    def labels_to_categorical(self):
        '''
        Input: None
        Purpose: Convert the emotion labels to one hot encoded values
        Output: None
        '''
        _num_classes = len(self._oulu_casia_config['_emotion_label_to_idx'])
        # Modify Training Labels
        self.y_train = [
                self._oulu_casia_config['_emotion_label_to_idx'][label]
                for label in self.y_train]
        self.y_train = to_categorical(self.y_train, _num_classes)
        
        # Modify Testing Labels
        self.y_test = [
                self._oulu_casia_config['_emotion_label_to_idx'][label]
                for label in self.y_test]
        self.y_test = to_categorical(self.y_test, _num_classes)
        
    
    def train_set_data_generator_flow(self, flow_args):
        '''
        Input 1: Arguments to the ImageDataGenerator flow fn (Expected Dict)
                 Pass empty dictionary if default args to be used. Values for
                 x and y will be overwritten by the dataset
        Purpose: Generates augmented images using the training set images as
                 per the ImageDataGenerator flow
        Output: ImageDataGenerator flow output
        '''
        flow_args['x'] = self.X_train
        flow_args['y'] = self.y_train
        return self.train_datagen.flow(**flow_args)
    
    
    def test_set_data_generator_flow(self, flow_args):
        '''
        Input 1: Arguments to the ImageDataGenerator flow fn (Expected Dict)
                 Pass empty dictionary if default args to be used. Values for
                 x and y will be overwritten by the dataset
        Purpose: Returns a ImageDataGenerator fit to the testing set
                 configured according to the OULU CASIA Data Gen configuration
        Output: ImageDataGenerator object
        '''
        flow_args['x'] = self.X_test
        flow_args['y'] = self.y_test
        return self.test_datagen.flow(**flow_args)
    

    def normalize_and_center_images(self):
        '''
        Input: None
        Purpose: Subract images values by the mean and divide the difference by
                 max value
        Output: None
        '''
        self.X_train = (self.X_train - self.X_train.mean(axis=0)) / 255
        self.X_test = (self.X_test - self.X_test.mean(axis=0)) / 255


    def optical_flow_norm(self, X):
        '''
        Input: Sequence of images as a 5D Numpy array
        Purpose: Center images with respect to images in the sequence to get
                 optical flow of images
        Output: Sequence of images centered with respect to the images in the
                sequence
        '''
        for seq_idx, sequence in enumerate(X):
            sequence = sequence - sequence.mean(axis = 0)
            sequence += sequence.min()
            X[seq_idx] = sequence
        return X


    def get_data_set_config(self):
        '''
        Input: None
        Purpose: A getter function to read the dataset configuration
        Output: OULU CASIA Data Set Configuration as a dictionary
        '''
        return self._oulu_casia_config
    
    
    def get_data_set(self):
        '''
        Input: None
        Purpose: A getter function to read the OULU CASIA dataset
        Output: [Train Data, Train Labels, Test Data, Test Labels]
        '''
        return [self.X_train, self.y_train, self.X_test, self.y_test]


def inference_from_evaluate(evaluate_dir,
                            names_label_mapping,
                            resolution = None):
    '''
    Input 1: Path to Evaluate Directory
    Input 2: Dictionary where key is name of file and value is the label
    Input 3: Resize resolution (Default is None to retain original)
    Purpose: Export as numpy array the images and the names as the labels
             
             TO BE USED ONLY WHEN TRAINING IS
             DONE WITHOUT NORMALIZATION FOR
             MODIFIED EXPANDED DATASET MODE
             
    Output: [[Images], [Labels]]
    '''
    if not os.path.exists(evaluate_dir):
        raise Exception('Directory does not exist')
    
    file_name_without_extension = re.compile('(.*)\..*')
    images = []
    labels = []
    for file_name in os.listdir(evaluate_dir):
        img = Image.open(evaluate_dir + '/' + file_name)
        if resolution:
            img = img.resize(resolution)
        images.append(np.array(img))
        name = file_name_without_extension.findall(file_name)[0]
        labels.append(names_label_mapping[name])
    images = np.array(images)
    labels = np.array(labels)
    return [images, labels]


def get_image_seq_apply_optical_flow_norm(img_seq_dir,
                                          resolution = None):
    '''
    Input 1: Path to the image sequence directory
    Input 2: Resize resolution (Default is None to retain original)
    Purpose: Get image sequence from a directory and return as numpy array of
             optical flow normalized images
    Output: Numpy array of optical flow normalized images in order
    '''
    if not os.path.exists(img_seq_dir):
        raise Exception('Directory does not exist')
    images = []
    for file_name in sorted(os.listdir(img_seq_dir)):
        img = Image.open(img_seq_dir + '/' + file_name)
        if resolution:
            img = img.resize(resolution)
        images.append(np.array(img))
    images = np.array(images)
    images = images - images.mean(axis = 0)
    images += images.min()
    return images
        

############################ MER DATA SET: DEAM ###############################

_deam_ds_config = {
        '_spectro_path' : './deam/spectrograms',
        '_arousal_file_path' : './deam/arousal.csv',
        '_valence_file_path' : './deam/valence.csv',
        '_image_resolution' : (120, 240)
        }


class deam_ds(object):
    
    def __init__(self,
                 _spectro_path, 
                 _arousal_file_path,
                 _valence_file_path,
                 _image_resolution = None,
                 nEntries = 1744,
                 test_set_fraction = 0.0,
                 shuffle_data = False,
                 center_and_normalize = True):
        '''
        Input 1: Spectrogram Directory Path
        Input 2: Arousal CSV File Path
        Input 3: Valence CSV File Path
        Input 4: Image Resolution (Width, Height) (Default: Original)
        Input 5: Number of Entries to be read from CSV (Default: 1744)
        Input 6: Test Set Fraction
        Input 7: Shuffle Data (Default: True)
        Input 8: Center and Normalize Flag (Default: True)
        Purpose: Initialize the DEAM Dataset Class
        Output: None
        '''
        # Validate Arguments
        if not os.path.exists(_spectro_path):
            raise Exception('Spectrogram Folder Path Invalid')
        if not os.path.exists(_arousal_file_path):
            raise Exception('Arousal File Path Invalid')
        if not os.path.exists(_valence_file_path):
            raise Exception('Valence File Path Invalid')
        arousal = pd.read_csv(_arousal_file_path, nrows = nEntries)
        valence = pd.read_csv(_valence_file_path, nrows = nEntries)
        arousal = arousal.values
        valence = valence.values
        valence_arousal_map = []
        _start_col = 1
        _end_col = 60
        for row in range(arousal.shape[0]):
            valence_arousal = valence[row][_start_col: _end_col + 1]
            valence_arousal = np.r_[valence_arousal, arousal[row][
                    _start_col: _end_col + 1]]
            valence_arousal_map.append(valence_arousal)
        valence_arousal_map = np.array(valence_arousal_map)
        spectro_images  = []
        # Convert images/spectograms to numpy arrays
        for specto_file in sorted(os.listdir(_spectro_path)):
            img = Image.open(_spectro_path + "/" + specto_file)
            if _image_resolution:
                img = img.resize(_image_resolution)
            img_arr = np.array(img)
            spectro_images.append(img_arr)
        spectro_images = np.array(spectro_images)
        self.ds_mean = spectro_images.mean(axis = 0)
        if center_and_normalize:
            spectro_images = (spectro_images - self.ds_mean) / 255
        self.X_train, self.X_test, self.y_train, self.y_test = (
                train_test_split(spectro_images, valence_arousal_map,
                                 test_size = test_set_fraction,
                                 shuffle = shuffle_data))


    def rot_right_spectrograms(self):
        '''
        Input: None
        Purpose: Rotate the spectrograms in the dataset to the right so that
                 the time domain is aligned along row and frequency along cols
        Output: None
        '''
        self.X_train = np.flip(self.X_train, axis = 1).transpose((0, 2, 1, 3))
        self.X_test = np.flip(self.X_test, axis = 1).transpose((0, 2, 1, 3))


    def get_data_set(self):
        '''
        Input: None
        Purpose: Getter function for the dataset
        Output: [X_train, y_train, X_test, y_test]
        '''
        return [self.X_train, self.y_train, self.X_test, self.y_test]
    
    
    def get_data_set_mean(self):
        '''
        Input: None
        Purpose: Getter function to return the mean of the dataset
        Output: Data Set Mean
        '''
        return self.ds_mean
    
    
def convert_arousal_valence_to_emotion(arousal, valence):
    '''
    Input 1: Arousal Value between -1 and 1
    Input 2: Valence Value between -1 and 1
    Purpose: Convert (Valence, Arousal) coordinate to a known emotion label
             to emotion according to standard Valence Arousal Map generalizing
             over 5 emotions
    
    Output: Emotion Label
    '''
    _neg_limit = -1.0
    _pos_limit = 1.0
    emotion = 5
    _emotion_label = {
            0 : "Anger",
            1 : "Fear",
            2 : "Happiness",
            3 : "Sadness",
            4 : "Calm",
            5 : "Unmapped"
            }
    
    if(arousal < _neg_limit
       or valence < _neg_limit
       or arousal > _pos_limit
       or valence > _pos_limit):
        return _emotion_label[emotion]
    
    if(valence > 0):
        if(arousal <= -0.5 and valence <= 0.5): emotion = 4
        else: emotion = 2
    elif(arousal > 0.5):
        if(valence >= -0.25): emotion = 1
        else: emotion = 0
    elif(arousal > 0):
        emotion = 0
    elif(valence <= -0.5 or arousal >= -0.25):
        emotion = 3
    else:
        emotion = 4
    return _emotion_label[emotion]


def convert_a_v_vector_to_emotion_possibilities(arousal_vec, valence_vec):
    '''
    Input 1: Arousal Vector - Numpy array of arousal values between -1 and 1
    Input 2: Valence Vector - Numpy array of valence values between -1 and 1
    Purpose: Return the unique set of emotions associated with the arousal
             and valence vector based on the mean and standard deviation
    Output: Numpy array of unique emotion values
    '''
    a_mean = arousal_vec.mean()
    v_mean = valence_vec.mean()
    a_std = arousal_vec.std()
    v_std = arousal_vec.std()
    a_deltas = [0, a_std, -a_std]
    v_deltas = [0, v_std, -v_std]
    emotion_labels = []
    for a_delta in a_deltas:
        for v_delta in v_deltas:
            arousal = a_mean + a_delta
            valence = v_mean + v_delta
            emotion_labels.append(convert_arousal_valence_to_emotion(arousal,
                                                                     valence))
    emotion_labels = set(emotion_labels)
    if 'Unmapped' in emotion_labels:
        emotion_labels.remove('Unmapped')
    return emotion_labels


def rotate_right_spectrograms(spectrograms):
    '''
    Input 1: Numpy array of spectrograms
    Purpose: Rotate the spectrograms in the dataset to the right so that
             the time domain is aligned along row and frequency along cols
    Output: Right Rotated Spectrograms
    '''
    return np.flip(spectrograms, axis = 1).transpose((0, 2, 1, 3))