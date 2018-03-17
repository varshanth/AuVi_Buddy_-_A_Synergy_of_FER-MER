'''
File: FER_LSTM_Net.py
Title: Long Short Term Memory LSTM Network
Description: Proof-of-Concept implementation of stacked LSTM
'''
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from FER_MER_data_set_utils import _oulu_casia_config

_oulu_casia_n_classes = len(_oulu_casia_config['_emotion_label_to_idx'])

def fer_lstm_net(_input_shape,
                 _n_classes = _oulu_casia_n_classes):
    '''
    Input 1: Input Shape for the FER LSTM Net
    Input 2: Expected Number of Output Classes (Emotion Labels)
    Output: FER LSTM Network Model
    Architecture:
        
        Expected Input Shape: (9, 1536) : 9 Timesteps each of size 1536
                        |
                        v
                LSTM: 512 Units
                        |
                        v
                Dense: 128 Units
                        |
                        v
                Softmax Output: n_classes
                        |
                        v
    '''
    print('Architecture: FER LSTM Network')
    _n_hidden_LSTM = 512 # Number of LSTM cells per each LSTM layer
    
    model = Sequential(name = 'FER_LSTM_Net')
    model.add(Bidirectional(
            LSTM(units = _n_hidden_LSTM, return_sequences = False,
                 activation='relu'),
                 input_shape = _input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(units = _n_hidden_LSTM // 4, activation = 'relu'))
    model.add(Dense(units = _n_classes, activation = 'softmax'))
    return model


if __name__ == '__main__':
    print('\nFER LSTM')
    print('*********************\n')
    model = fer_lstm_net((9, 1536))
    model.summary()
    save = int(input('Save Model Visualization to file? 0|1\n'))
    if save > 0:
        plot_model(model, to_file='{0}.png'.format(model.name),
                   show_shapes=True)

		
		
		
		
		
		
		
		
		