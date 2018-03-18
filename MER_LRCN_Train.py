from FER_MER_data_set_utils import deam_ds, _deam_ds_config
from MER_Spectro_Net import mer_spectro_net

########################### GLOBAL CONFIGURATION ##############################
save_model = True
_model_save_file_path = 'MER_LRCN_model.h5'
_test_size = 0.2


if __name__ == '__main__':
    ####################### LOAD DEAM SPECTROGRAM DATASET #####################
    
    deam_data_set = deam_ds(**_deam_ds_config)
    deam_data_set.rot_right_spectrograms()
    X_train, y_train, X_test, y_test = deam_data_set.get_data_set()
    
    ############################## LOAD MODEL #################################
    
    input_shape = (*_deam_ds_config['_image_resolution'], 3)
    model = mer_spectro_net(input_shape)
    model.summary()
    ############################# TRAIN MODEL #################################
    # Training Configuration
    _batch_size = 16
    _n_epochs = 180
    
    model.compile(optimizer = 'adam', 
    				loss = 'mean_squared_error',
    				metrics=['mean_squared_error'])
    model.fit(X_train, y_train, validation_data = (X_test, y_test), 
              epochs = _n_epochs, batch_size = _batch_size)
    if save_model:
        model.save(_model_save_file_path)