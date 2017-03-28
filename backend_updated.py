from keras.backend import *
from keras.backend.tensorflow_backend import _preprocess_conv3d_input, _preprocess_conv3d_kernel, _preprocess_border_mode, _postprocess_conv3d_output

def _preprocess_deconv3d_output_shape(x,shape, dim_ordering):
    if dim_ordering == 'th':
        shape = (shape[0], shape[2], shape[3], shape[4], shape[1])
    if shape[0] is None:
        shape = (tf.shape(x)[0], ) + tuple(shape[1:])
        shape = tf.stack(list(shape))
    return shape

def deconv3d(x, kernel, output_shape, strides=(1, 1, 1),
             border_mode='valid',
             dim_ordering='default',
             volume_shape=None, filter_shape=None):
    '''3D deconvolution (i.e. transposed convolution).

    # Arguments
        x: input tensor.
        kernel: kernel tensor.
        output_shape: 1D int tensor for the output shape.
        strides: strides tuple.
        border_mode: string, "same" or "valid".
        dim_ordering: "tf" or "th".
            Whether to use Theano or TensorFlow dimension ordering
            for inputs/kernels/ouputs.
    '''

    #print '*** Deconv3d: \n\tX:{0} \n\tkernel:{1} \n\tstride:{2} \n\tfilter_shape:{3} \n\toutput_shape:{4}'.format(int_shape(x), int_shape(kernel), strides, filter_shape, output_shape)

    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    x = _preprocess_conv3d_input(x, dim_ordering)
    output_shape = _preprocess_deconv3d_output_shape(x,output_shape, dim_ordering)
    kernel = _preprocess_conv3d_kernel(kernel, dim_ordering)
    kernel = tf.transpose(kernel, (0, 1, 2, 4, 3))
    padding = _preprocess_border_mode(border_mode)
    strides = (1,) + strides + (1,)

    #print '*** Deconv3d: \n\tkernel:{0} \n\tfilter_shape:{1} '.format(int_shape(kernel), filter_shape)
    #print output_shape,strides,padding

    x = tf.nn.conv3d_transpose(x, kernel, output_shape, strides,
                               padding=padding)
    return _postprocess_conv3d_output(x, dim_ordering)

 
