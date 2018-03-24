from keras.layers import (Conv2D, Dense, MaxPooling2D, Flatten, Dropout,
                          TimeDistributed, Bidirectional, LSTM, concatenate,
                          BatchNormalization)
from ArchLayers import _create_input_layer, _preactivation_layers
from keras.models import Model
from keras.utils.vis_utils import plot_model


def mer_spectro_net(_input_shape,
                    _n_out = 120):
    '''
    Input 1: Spectrogram Input Shape
    Input 2: Number of Arousal + Valence Values (Default: 60 + 60 = 120)
    Output: MER Spectro Net  Model
    Architecture:
        
        Expected Image Size for Spectrogram: 120 x 240, where 240 represents
        the frequency and 120 represents the time i.e. Transposed Spectrogram
                    |
                    v
           Conv2D: 2x2, D_OUT
                    |
                    v
          Max Pooling Across Time Axis: 60 x 240 x D_OUT
                    |
                    v
          (Conv2D (Freq Dim) + Activation + Conv2D (Freq Dim) + Max Pooling)x4
                    |
                    v
             Time Distributed Flatten = 60 x D_OUT
                    |
                    v
             Time Distributed Dense = 60 x 1024
             |                                   |  
             v                                   v
          (V) Bidirectional LSTM (N_Hidden)   (A) Bidirectional LSTM (N_Hidden)
                    |                                   |
                    v                                   v
    (V) Bidirectional LSTM (N_Hidden/2)     (A) Bidirectional LSTM (N_Hidden/2)
                    |
                    v
                 Dense: 120x1
                    
    '''
    
    _d_init = 64
    _freq_d = [128, 256, 256, 384]
    _hidden_units_LSTM = 256
    
    input_layer = _create_input_layer(_input_shape)
    down_sample_time = Conv2D(_d_init,
                              kernel_size = (2, 2),
                              strides = (1, 1),
                              padding = 'same')(input_layer)
    conv1_out = _preactivation_layers(down_sample_time)
    mp_out = MaxPooling2D(pool_size = (2, 1), strides = (2, 1))(conv1_out)
    intermed_out = mp_out
    # Create frequency scaling time invariant layers
    for freq_conv_idx, num_filters in enumerate(_freq_d):
        intermed_out = _preactivation_layers(intermed_out)
        intermed_out = Conv2D(num_filters, kernel_size = (1, 3))(intermed_out)
        intermed_out = _preactivation_layers(intermed_out)
        intermed_out = Conv2D(num_filters, kernel_size = (1, 3))(intermed_out)
        intermed_out = MaxPooling2D(pool_size = (1, 2),
                                    strides = (1, 2))(intermed_out)
    intermed_out = BatchNormalization()(intermed_out)
    # Distribute Output with respect to Time
    f_time_dis_out = TimeDistributed(Flatten())(intermed_out)
    f_time_dis_out = TimeDistributed(Dense(1024, activation = 'tanh'))(
            f_time_dis_out)
    f_time_dis_out = TimeDistributed(Dropout(0.2))(f_time_dis_out)
        
    # Feed each time distributed input to 2 BLSTM: for Valence and Arousal 
    valence_lstm_out = Bidirectional(LSTM(_hidden_units_LSTM,
                                          return_sequences = True,
                                          activation = 'tanh'))(f_time_dis_out)
    v_dropout_out = Dropout(0.2)(valence_lstm_out)
    valence_lstm_out = Bidirectional(LSTM(_hidden_units_LSTM // 2,
                                          activation = 'tanh'))(v_dropout_out)
    
    arousal_lstm_out = Bidirectional(LSTM(_hidden_units_LSTM,
                                          return_sequences = True,
                                          activation = 'tanh'))(f_time_dis_out)
    a_dropout_out = Dropout(0.2)(arousal_lstm_out)
    arousal_lstm_out = Bidirectional(LSTM(_hidden_units_LSTM // 2,
                                  activation = 'tanh'))(a_dropout_out)
    
    # Arousal and Valence values are between -1 and 1, use tanh activation
    valence_dense = Dense(_n_out // 2, activation = 'tanh')(valence_lstm_out)
    arousal_dense = Dense(_n_out // 2, activation = 'tanh')(arousal_lstm_out)
    final_dense = concatenate([valence_dense, arousal_dense])
    model = Model(inputs = input_layer, outputs = final_dense,
                  name = 'MER_Spectro_Net')
    return model


if __name__ == '__main__':
    print('\nMER Spectro Net')
    print('*********************\n')
    model = mer_spectro_net((120, 240, 3))
    model.summary()
    save = int(input('Save Model Visualization to file? 0|1\n'))
    if save > 0:
        plot_model(model, to_file='{0}.png'.format(model.name),
                   show_shapes=True)
