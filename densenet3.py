from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution3D
from keras.layers.pooling import AveragePooling3D, MaxPooling3D
from keras.layers.pooling import GlobalAveragePooling3D, GlobalMaxPooling3D
from keras.layers import Input, merge, Flatten
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K

def conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1E-5):
    ''' Apply BatchNorm, Relu 3x3, Conv3D, optional bottleneck block and dropout

    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor

    Returns: keras tensor with batch_norm, relu and convolution3d added (optional bottleneck)

    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(ip)
    x = Activation('relu')(x)

    if bottleneck:
        inter_channel = nb_filter * 4 # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

        x = Convolution3D(inter_channel, 1, 1, 1,init='he_uniform', border_mode='same', bias=False,
                          W_regularizer=l2(weight_decay))(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)

    x = Convolution3D(nb_filter, 3, 3, 3, init="he_uniform", border_mode="same", bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(ip, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-5):
    ''' Apply BatchNorm, Relu 1x1, Conv3D, optional compression, dropout and Maxpooling3D

    Args:
        ip: keras tensor
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay factor

    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool

    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(ip)
    x = Activation('relu')(x)
    x = Convolution3D(int(nb_filter * compression), 1, 1, 1,init="he_uniform", border_mode="same", bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(x)

    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1E-5):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones

    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor

    Returns: keras tensor with nb_layers of conv_block appended

    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    feature_list = [x]

    for i in range(nb_layers):
        x = conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
        feature_list.append(x)
        x = merge(feature_list, mode='concat', concat_axis=concat_axis)
        nb_filter += growth_rate

    return x, nb_filter


def create_dense_net(nb_classes, img_dim, nb_layers = 6, nb_dense_block=3, growth_rate=12, nb_filter=-1,
                     bottleneck=False, reduction=0.0, dropout_rate=None, weight_decay=1E-5):
    ''' Build the create_dense_net model

    Args:
        nb_classes: number of classes
        img_dim: tuple of shape(rows, columns, depth, channels)
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters. Default -1 indicates initial number of filters is 2 * growth_rate
        bottleneck: add bottleneck blocks
        reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay

    Returns: keras tensor with nb_layers of conv_block appended

    '''

    model_input = Input(shape=img_dim,name='inputs')

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, "reduction value must lie between 0.0 and 1.0"

    # layers in each dense block

    # compute initial nb_filter if -1, else accept users initial nb_filter
    if nb_filter <= 0:
        nb_filter = 2 * growth_rate

    # compute compression factor
    compression = 1.0 - reduction
    
    # Initial convolution
    #x = Convolution3D(nb_filter, 3, 3, 3, init="he_uniform", border_mode="same", name="initial_conv3D", bias=False,
    #                  W_regularizer=l2(weight_decay))(model_input)
    x = model_input
    x = transition_block(x, nb_filter, compression=compression, dropout_rate=dropout_rate,
                             weight_decay=weight_decay)
        
    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=bottleneck,
                                   dropout_rate=dropout_rate, weight_decay=weight_decay)
        # add transition_block
        x = transition_block(x, nb_filter, compression=compression, dropout_rate=dropout_rate,
                             weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    # The last dense_block does not have a transition_block
    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=bottleneck,
                               dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    #x = GlobalAveragePooling3D()(x)
    x = GlobalMaxPooling3D()(x)
    #x= Flatten()(x)
    #x= Dropout(0.5)(x)
    x = Dense(nb_classes, W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay),name='dense',init='normal')(x)
    if nb_classes>1:
        x = Activation('softmax',name='output')(x)
    else:
        x = Activation('sigmoid',name='output')(x)
    densenet = Model(input=model_input, output=x, name="create_dense_net")
    return densenet


if __name__ == '__main__':
    model = create_dense_net(nb_classes=10, img_dim=(32, 32, 32,1), nb_layers=4, growth_rate=12, bottleneck=True, reduction=0.5)

    model.summary()
