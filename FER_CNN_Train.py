from keras import optimizers
from keras.callbacks import ModelCheckpoint
from FER_MER_data_set_utils import oulu_casia_ds
from FER_Dense_Net import fer_dense_net


############### LOAD MODIFIED EXPANDED OULU CASIA DATASET #####################
oulu_casia_dataset = oulu_casia_ds(dataset_mode = 'modified_expanded')
_oulu_casia_dataset_config = oulu_casia_dataset.get_data_set_config()
_num_classes = len(_oulu_casia_dataset_config['_emotion_label_to_idx'])
img_shape = _oulu_casia_dataset_config['_oulu_casia_get_data_set_args'][
        '_image_resolution']
img_shape = (img_shape[0], img_shape[1], 3)
# Convert emotion labels to categorical
oulu_casia_dataset.labels_to_categorical()


############################## LOAD MODEL #####################################
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model = fer_dense_net(img_shape, _num_classes)

model.compile(loss = 'categorical_crossentropy',
              optimizer = sgd,
              metrics=['accuracy'])
model.summary()

############################# TRAIN MODEL #####################################

# Training Configuration
checkpoint_file_name = 'FER_CNN_weights_best.h5'
batch_size = 32
flow_gen_args = {'batch_size' : batch_size}
req_training_set_size = 10000
req_testing_set_size = 0.2 * req_training_set_size
training_steps_per_epoch = req_training_set_size // batch_size
validation_steps_per_epoch = req_testing_set_size // batch_size
num_epochs = 100

# Configure Checkpoint
checkpoint = ModelCheckpoint(checkpoint_file_name, 
                             monitor = 'val_acc',
                             verbose = 1,
                             save_best_only = True,
                             mode = 'max')

history = model.fit_generator(
        oulu_casia_dataset.train_set_data_generator_flow(flow_gen_args),
        steps_per_epoch = training_steps_per_epoch,
        validation_data = oulu_casia_dataset.test_set_data_generator_flow(
                flow_gen_args),
        validation_steps = validation_steps_per_epoch,
        callbacks = [checkpoint],
        verbose = 1,
        epochs = num_epochs)
