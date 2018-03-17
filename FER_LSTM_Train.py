from FER_LSTM_Net import fer_lstm_net
from FER_CNN_Inference import fer_cnn_get_latent_rep_data_set
from sklearn.model_selection import train_test_split

########################### GLOBAL CONFIGURATION ##############################
save_model = True
_model_save_file_path = 'FER_LSTM_model.h5'
_test_size = 0.2


if __name__ == '__main__':
    ##################### LOAD LATENT REPRESENTATION DATASET ##################
    X, y = fer_cnn_get_latent_rep_data_set()
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size = _test_size)
    ############################## LOAD MODEL #################################
    model = fer_lstm_net(X_train.shape[1:])
    ############################# TRAIN MODEL #################################
    # Training Configuration
    _batch_size = 16
    _n_epochs = 15
    model.compile(optimizer = 'adam', 
    				loss = 'categorical_crossentropy',
    				metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, validation_data = (X_test, y_test), 
              epochs = _n_epochs, batch_size = _batch_size)
    if save_model:
        model.save(_model_save_file_path)







