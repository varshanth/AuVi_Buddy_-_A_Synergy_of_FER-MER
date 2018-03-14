'''
File: FER_Dense_Net.py
Title: Densely Connected CNN
Description: Proof-of-Concept implementation of DenseNet
'''

from keras.layers import Conv2D, Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.utils.vis_utils import plot_model
from ArchLayers import _create_input_layer, _dense_layer, _dense_transition


def fer_dense_net(_input_shape, _n_classes):
    '''
    Expected Image Size for OULU CASIA: 224x224xd (channel last)
			|
			v
    2D: 7x7 D_OUT= D
			|
			v
    Dense_Block0: (D_IN = D->([1x1xGR_LIMIT], 3x3) -> D_OUT= D_IN + G_RATE) * (LAYERS_PER_BLOCK=4)
    Dense_Transition:
         (NxN, D_IN = Dense_Block0_D_OUT)-> (COMPRESS, MAXPOOL) -> OUT: (N/2 x N/2, D_OUT=COMPRESS_D0)
			|
			v
    Dense_Block1: (D_IN = D->([1x1xGR_LIMIT], 3x3) -> D_OUT= D_IN + G_RATE) * (LAYERS_PER_BLOCK=4)
    Dense_Transition:
		IN: (N/2 x N/2, D_IN = Dense_Block1_D_OUT)-> (COMPRESS, MAXPOOL) -> OUT: (N/4 x N/4, D_OUT=COMPRESS_D1)
			|
			v
    Dense_Block2: (D_IN = D->([1x1xGR_LIMIT], 3x3) -> D_OUT= D_IN + G_RATE) * (LAYERS_PER_BLOCK=4)
    Dense_Transition:
		IN: (N/4 x N/4 , D_IN = Dense_Block2_D_OUT)-> (COMPRESS, MAXPOOL) -> OUT: (N/8 x N/8, D_OUT=COMPRESS_D2)
			|
			v
    Dense_Block3: (D_IN = D->([1x1xGR_LIMIT], 3x3) -> D_OUT= D_IN + G_RATE) * (LAYERS_PER_BLOCK=4)
			|
			v
    GlobalAveragePool: OUT: 1 x 1, D_OUT = Dense_Block3_D_OUT
			|
			v
    Dense+ReLU Activation: D_OUT = 128
			|
			v
    Dense+Softmax Activation: D_OUT = N_CLASSES
			|
			v
    '''
    print('Architecture: FER Dense Network')
    _d_init = 96 # Initial Depth
    _growth_limit = 384 #  Growth Limit
    _n_dense_blocks = 4 # Number of Dense Blocks
    # Number of Dense Layers per block
    _n_dense_layers_per_block = (6, 12, 24, 16)
    growth_rate = 16 # Growth Rate for dense layers

    input_layer = _create_input_layer(_input_shape)
    intermed = Conv2D(_d_init, kernel_size=(7,7), strides=(2, 2),
                      padding='same')(input_layer)
    new_depth = _d_init
    '''
    TO_CORRECT:
    
    Output Sizes:
    *************
    Input Size = 224 x 224
    Output Size After ith Transition:
		1: 16 x 16
		2: 8 x 8
		3: 4 x 4

    Depth Sizes:
    ************
	 Depth to 1st Block = 64
	 Depth to the ith (i > 1) block:
         2: 128
         3: 256
         4: 512
    Depth Out = 1024
    '''
    for dense_block_idx in range(_n_dense_blocks):
        print('Block IDX = {0}'.format(dense_block_idx))
        for dense_layer_idx in range(
                _n_dense_layers_per_block[dense_block_idx]):
            intermed = _dense_layer(intermed, new_depth, growth_rate,
                                    _growth_limit)
            new_depth += growth_rate
        if dense_block_idx != (_n_dense_blocks-1):
            # No Dense Transition for last layer
            intermed = _dense_transition(intermed, new_depth,
                                         int(new_depth))
            
    gap_out = GlobalAveragePooling2D()(intermed)
    dense_out = Dense(128, activation='relu')(gap_out) 
    final_out = Dense(_n_classes, activation='softmax')(dense_out)
    model = Model(inputs=input_layer, outputs=final_out, name='FER_Dense_Net')
    return model


if __name__ == '__main__':
    print('\nFER Densely Connected CNN')
    print('*********************\n')
    model = fer_dense_net((224,224,3),7)
    model.summary()
    save = int(input('Save Model Visualization to file? 0|1\n'))
    if save > 0:
        plot_model(model, to_file='{0}.png'.format(model.name),
                   show_shapes=True)