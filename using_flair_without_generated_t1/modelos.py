from keras.layers import Input,Conv1D, Conv2D,Activation, Reshape, Lambda, dot, add, Conv3D , MaxPool1D,Dropout, Concatenate, BatchNormalization,Conv3DTranspose,GlobalAveragePooling3D, Add,MaxPooling3D, UpSampling3D,Flatten,Dense,Activation,SpatialDropout3D,Reshape, Lambda,GlobalMaxPooling3D
from keras.models import Model , load_model
from keras.activations import softmax
from keras import backend as K
from keras import optimizers
from keras.regularizers import l2
from keras.layers.merge import concatenate
import tensorflow as tf
from keras.legacy import interfaces
import keras.backend as K
from keras.optimizers import Optimizer
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers
from keras.utils import conv_utils
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import numpy as np

def non_local_block(ip, intermediate_dim=None, compression=2,
                    mode='embedded', add_residual=True):
    """
    Adds a Non-Local block for self attention to the input tensor.
    Input tensor can be or rank 3 (temporal), 4 (spatial) or 5 (spatio-temporal).
    Arguments:
        ip: input tensor
        intermediate_dim: The dimension of the intermediate representation. Can be
            `None` or a positive integer greater than 0. If `None`, computes the
            intermediate dimension as half of the input channel dimension.
        compression: None or positive integer. Compresses the intermediate
            representation during the dot products to reduce memory consumption.
            Default is set to 2, which states halve the time/space/spatio-time
            dimension for the intermediate step. Set to 1 to prevent computation
            compression. None or 1 causes no reduction.
        mode: Mode of operation. Can be one of `embedded`, `gaussian`, `dot` or
            `concatenate`.
        add_residual: Boolean value to decide if the residual connection should be
            added or not. Default is True for ResNets, and False for Self Attention.
    Returns:
        a tensor of same shape as input
    """
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    ip_shape = K.int_shape(ip)

    if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
        raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

    if compression is None:
        compression = 1

    dim1, dim2, dim3 = None, None, None

    # check rank and calculate the input shape
    if len(ip_shape) == 3:  # temporal / time series data
        rank = 3
        batchsize, dim1, channels = ip_shape

    elif len(ip_shape) == 4:  # spatial / image data
        rank = 4

        if channel_dim == 1:
            batchsize, channels, dim1, dim2 = ip_shape
        else:
            batchsize, dim1, dim2, channels = ip_shape

    elif len(ip_shape) == 5:  # spatio-temporal / Video or Voxel data
        rank = 5

        if channel_dim == 1:
            batchsize, channels, dim1, dim2, dim3 = ip_shape
        else:
            batchsize, dim1, dim2, dim3, channels = ip_shape

    else:
        raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')

    # verify correct intermediate dimension specified
    if intermediate_dim is None:
        # intermediate_dim = channels // 2
        intermediate_dim = channels // 4

        if intermediate_dim < 1:
            intermediate_dim = 1

    else:
        intermediate_dim = int(intermediate_dim)

        if intermediate_dim < 1:
            raise ValueError('`intermediate_dim` must be either `None` or positive integer greater than 1.')

    if mode == 'gaussian':  # Gaussian instantiation
        x1 = Reshape((-1, channels))(ip)  # xi
        x2 = Reshape((-1, channels))(ip)  # xj
        f = dot([x1, x2], axes=2)
        f = Activation('softmax')(f)

    elif mode == 'dot':  # Dot instantiation
        # theta path
        theta = _convND(ip, rank, intermediate_dim)
        theta = Reshape((-1, intermediate_dim))(theta)

        # phi path
        phi = _convND(ip, rank, intermediate_dim)
        phi = Reshape((-1, intermediate_dim))(phi)

        f = dot([theta, phi], axes=2)

        size = K.int_shape(f)

        # scale the values to make it size invariant
        f = Lambda(lambda z: (1. / float(size[-1])) * z)(f)

    elif mode == 'concatenate':  # Concatenation instantiation
        raise NotImplementedError('Concatenate model has not been implemented yet')

    else:  # Embedded Gaussian instantiation
        # theta path
        theta = _convND(ip, rank, intermediate_dim)
        theta = Reshape((-1, intermediate_dim))(theta)

        # phi path
        phi = _convND(ip, rank, intermediate_dim)
        phi = Reshape((-1, intermediate_dim))(phi)

        if compression > 1:
            # shielded computation
            phi = MaxPool1D(compression)(phi)

        f = dot([theta, phi], axes=2)
        f = Activation('softmax')(f)

    # g path
    g = _convND(ip, rank, intermediate_dim)
    g = Reshape((-1, intermediate_dim))(g)

    if compression > 1 and mode == 'embedded':
        # shielded computation
        g = MaxPool1D(compression)(g)

    # compute output path
    y = dot([f, g], axes=[2, 1])

    # reshape to input tensor format
    if rank == 3:
        y = Reshape((dim1, intermediate_dim))(y)
    elif rank == 4:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, intermediate_dim))(y)
        else:
            y = Reshape((intermediate_dim, dim1, dim2))(y)
    else:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, dim3, intermediate_dim))(y)
        else:
            y = Reshape((intermediate_dim, dim1, dim2, dim3))(y)

    # project filters
    y = _convND(y, rank, channels)

    # residual connection
    if add_residual:
        y = add([ip, y])

    return y


def _convND(ip, rank, channels):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    if rank == 3:
        x = Conv1D(channels, 1, padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    elif rank == 4:
        x = Conv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    else:
        x = Conv3D(channels, (1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    return x


def infer_spatial_rank(input_tensor):
    """
    e.g. given an input tensor [Batch, X, Y, Z, Feature] the spatial rank is 3
    """
    input_shape = input_tensor.shape
    input_shape.with_rank_at_least(3)
    #dims = input_tensor.get_shape().ndims - 2
    #assert dims > 0, "input tensor should have at least one spatial dim, " \
    #                 "in addition to batch and channel dims"
    return int(input_shape.ndims - 2)

def expand_spatial_params(input_param, spatial_rank, param_type=int):
    """
    Expand input parameter
    e.g., ``kernel_size=3`` is converted to ``kernel_size=[3, 3, 3]``
    for 3D images (when ``spatial_rank == 3``).
    """
    spatial_rank = int(spatial_rank)
    try:
        if param_type == int:
            input_param = int(input_param)
        else:
            input_param = float(input_param)
        return (input_param,) * spatial_rank
    except (ValueError, TypeError):
        pass
    try:
        if param_type == int:
            input_param = \
                np.asarray(input_param).flatten().astype(np.int).tolist()
        else:
            input_param = \
                np.asarray(input_param).flatten().astype(np.float).tolist()
    except (ValueError, TypeError):
        # skip type casting if it's a TF tensor
        pass
    assert len(input_param) >= spatial_rank, \
        'param length should be at least have the length of spatial rank'
    return tuple(input_param[:spatial_rank])

class LinearResizeLayer(Layer):
	"""
	Resize 2D/3D images using ``tf.image.resize_bilinear``
	(without trainable parameters).
	"""

	def __init__(self, new_size, name='trilinear_resize'):
		"""

		:param new_size: integer or a list of integers set the output
			2D/3D spatial shape.  If the parameter is an integer ``d``,
			it'll be expanded to ``(d, d)`` and ``(d, d, d)`` for 2D and
			3D inputs respectively.
		:param name: layer name string
		"""
		super(LinearResizeLayer, self).__init__(name=name)
		self.new_size = new_size

	def compute_output_shape(self, input_shape):
		return (input_shape[0],self.new_size[0],self.new_size[1],self.new_size[2],input_shape[4])

	def call(self, input_tensor):
		"""
		Resize the image by linearly interpolating the input
		using TF ``resize_bilinear`` function.

		:param input_tensor: 2D/3D image tensor, with shape:
			``batch, X, Y, [Z,] Channels``
		:return: interpolated volume
		"""

		input_spatial_rank = infer_spatial_rank(input_tensor)
		assert input_spatial_rank in (2, 3), \
			"linearly interpolation layer can only be applied to " \
			"2D/3D images (4D or 5D tensor)."
		self.new_size = expand_spatial_params(self.new_size, input_spatial_rank)

		if input_spatial_rank == 2:
			return tf.image.resize_bilinear(input_tensor, self.new_size)

		b_size, x_size, y_size, z_size, c_size = input_tensor.shape.as_list()
		x_size_new, y_size_new, z_size_new = self.new_size

		if (x_size == x_size_new) and (y_size == y_size_new) and (z_size == z_size_new):
			# already in the target shape
			return input_tensor

		# resize y-z
		squeeze_b_x = tf.reshape(input_tensor, [-1, y_size, z_size, c_size])
		resize_b_x = tf.image.resize_bilinear(squeeze_b_x, [y_size_new, z_size_new])
		resume_b_x = tf.reshape(resize_b_x,  [-1, x_size, y_size_new, z_size_new, c_size])

		# resize x
		#   first reorient
		reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])

		#   squeeze and 2d resize
		squeeze_b_z = tf.reshape(reoriented, [-1, y_size_new, x_size, c_size])
		resize_b_z = tf.image.resize_bilinear(squeeze_b_z, [y_size_new, x_size_new])
		resume_b_z = tf.reshape(resize_b_z, [-1, z_size_new, y_size_new, x_size_new, c_size])

		output_tensor = tf.transpose(resume_b_z, [0, 3, 2, 1, 4])
		return output_tensor





def to_list(x):
    if type(x) not in [list, tuple]:
        return [x]
    else:
        return list(x)

class GroupNormalization(Layer):
    def __init__(self, axis=-1,
                 gamma_init='one', beta_init='zero',
                 gamma_regularizer=None, beta_regularizer=None,
                 epsilon=1e-6,
                 group=32,
                 data_format=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)

        self.axis = to_list(axis)
        self.gamma_init = initializers.get(gamma_init)
        self.beta_init = initializers.get(beta_init)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.epsilon = epsilon
        self.group = group
        self.data_format = K.normalize_data_format(data_format)

        self.supports_masking = True

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = [1 for _ in input_shape]
        if self.data_format == 'channels_last':
            channel_axis = -1
            shape[channel_axis] = input_shape[channel_axis]
        elif self.data_format == 'channels_first':
            channel_axis = 1
            shape[channel_axis] = input_shape[channel_axis]
        #for i in self.axis:
        #    shape[i] = input_shape[i]
        self.gamma = self.add_weight(shape=shape,
                                     initializer=self.gamma_init,
                                     regularizer=self.gamma_regularizer,
                                     name='gamma')
        self.beta = self.add_weight(shape=shape,
                                    initializer=self.beta_init,
                                    regularizer=self.beta_regularizer,
                                    name='beta')
        self.built = True

    def call(self, inputs, mask=None):
        input_shape = K.int_shape(inputs)

        if len(input_shape) == 5:
            if self.data_format == 'channels_last':
                batch_size, h, w, l, c = input_shape
                if batch_size is None:
                    batch_size = -1

                if c < self.group:
                    raise ValueError('Input channels should be larger than group size' +
                                     '; Received input channels: ' + str(c) +
                                     '; Group size: ' + str(self.group)
                                    )

                x = K.reshape(inputs, (batch_size, h, w, l , self.group, c // self.group))
                mean = K.mean(x, axis=[1, 2,3, 5], keepdims=True)
                std = K.sqrt(K.var(x, axis=[1, 2, 3, 5], keepdims=True) + self.epsilon)
                x = (x - mean) / std

                x = K.reshape(x, (batch_size, h, w, l , c))
                return self.gamma * x + self.beta


    def get_config(self):
        config = {'epsilon': self.epsilon,
                  'axis': self.axis,
                  'gamma_init': initializers.serialize(self.gamma_init),
                  'beta_init': initializers.serialize(self.beta_init),
                  'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
                  'beta_regularizer': regularizers.serialize(self.gamma_regularizer),
                  'group': self.group
                 }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def load_UNET3D_bottleneck_regularized(ps1,ps2,ps3,ch,nc=4,nf=24,drop=0.5,groups=8,final_act='softmax'):   #3 levels + linear upsampling
    # See https://arxiv.org/pdf/1806.00546.pdf
    #nc: number of output classes
    #nf: number of filter
    #ch: number of channels
    G=groups
    # model UNET 3D
    pool_size=[2,2,2]

    input_img = Input(shape=(ps1, ps2, ps3, ch))     # adapt this if using `channels_first` image data format

    conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(input_img)
    conv1 = GroupNormalization(group=G)(conv1)
    #conv1 = WeightNorm(conv1)
    conv1 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv1)
    #conv1 = BatchNormalization()(conv1)
    #conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)
    if(drop>0):
        pool1 = Dropout(drop)(pool1)

    conv2 = GroupNormalization(group=G)(pool1)
    #conv2 = WeightNorm(pool1)
    conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = GroupNormalization(group=G)(conv2)
    #conv2 = WeightNorm(conv2)

    conv2 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv2)
    #conv2 = BatchNormalization()(conv2)
    #conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)
    if(drop>0):
        pool2 = Dropout(drop)(pool2)

    conv3 = GroupNormalization(group=G)(pool2)
    #conv3 = WeightNorm(pool2)

    conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = GroupNormalization(group=G)(conv3)
    #conv3 = WeightNorm(conv3)

    conv3 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv3)
    #conv3 = BatchNormalization()(conv3)
    #conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)
    if(drop>0):
            pool3 = Dropout(drop)(pool3)

    conv4 = GroupNormalization(group=G)(pool3)
    #conv4 = WeightNorm(pool3)

    conv4 = Conv3D(nf*16, (3, 3, 3), activation='relu', name='bottleneck' ,padding='same')(conv4) #,
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)


    #up5 = UpSampling3D()(conv4)
    new_shape = conv3.shape.as_list()[1:-1]
    up5  = LinearResizeLayer(new_shape,name='up5')(conv4)

    up5 = concatenate([up5, conv3], axis=4) # up5 = 512 + conv3 = 256
    up5 = GroupNormalization(group=G)(up5)
    #up5 = WeightNorm(up5)

    conv5 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(up5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)


    #up6 = UpSampling3D()(conv5)
    new_shape = conv2.shape.as_list()[1:-1]
    up6  = LinearResizeLayer(new_shape,name='up6')(conv5)

    up6 = concatenate([up6, conv2], axis=4) # up6 = 256 + conv2 = 128
    up6 = GroupNormalization(group=G)(up6)
    #up6 = WeightNorm(up6)

    conv6 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)


    #up7 = UpSampling3D()(conv6)
    new_shape = conv1.shape.as_list()[1:-1]
    up7   = LinearResizeLayer(new_shape,name='up7')(conv6)

    up7 = concatenate([up7, conv1], axis=4) # up7 = 128 + conv1 = 64
    up7 = GroupNormalization(group=G)(up7)
    #up7 = WeightNorm(up7)

    conv7 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)


    output = Conv3D(nc, (3, 3, 3), activation=final_act, padding='same')(conv7)
    bottleneck_reduced=GlobalAveragePooling3D()(conv4)
    model = Model(input_img, [output,bottleneck_reduced])
    #model = Model(input_img, [output,conv4])

    return model

class Adam_lr_mult(Optimizer):
    """Adam optimizer.
    Adam optimizer, with learning rate multipliers built on Keras implementation
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)

    AUTHOR: Erik Brorson
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False,
                 multipliers=None, debug_verbose=False,**kwargs):
        super(Adam_lr_mult, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.multipliers = multipliers
        self.debug_verbose = debug_verbose

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):

            # Learning rate multipliers
            if self.multipliers:
                multiplier = [mult for mult in self.multipliers if mult in p.name]
            else:
                multiplier = None
            if multiplier:
                new_lr_t = lr_t * self.multipliers[multiplier[0]]
                if self.debug_verbose:
                    print('Setting {} to learning rate {}'.format(multiplier[0], new_lr_t))
                    print(K.get_value(new_lr_t))
            else:
                new_lr_t = lr_t
                if self.debug_verbose:
                    print('No change in learning rate {}'.format(p.name))
                    print(K.get_value(new_lr_t))
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - new_lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - new_lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad,
                  'multipliers':self.multipliers}
        base_config = super(Adam_lr_mult, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def filter_size(init,mult,exp):
    i=int(init*(mult)**exp)
    if((i%8 )== 0):

        return i
    else:
        if((i%8)>3):

            i= 8*(1+ i//8)
        else:

            i= 8*(i//8)

        return int(i)


    #conv2 = GroupNormalization(group=G)(pool1)

def double_conv_layer(x, size, dropout=0.0, batch_norm=True):
    axis = 4
    conv = Conv3D(size, (3,3,3), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv3D(size, (3,3,3), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = SpatialDropout3D(dropout)(conv)
    return conv


def load_UNET3D_MULTITASK(ps1,ps2,ps3,ch,nc=4,nf=24,drop=0.5,G=8):   #3 levels + linear upsampling

    pool_size=[2,2,2]

    input_img = Input(shape=(ps1, ps2, ps3, ch))     # adapt this if using `channels_first` image data format

    conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(input_img)
    #conv1 = BatchNormalization()(conv1)
    conv1 =GroupNormalization(group=G)(conv1)
    conv1 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv1)

    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)
    if(drop>0):
        pool1 = Dropout(drop)(pool1)

    conv2 = GroupNormalization(group=G)(pool1)

    conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = GroupNormalization(group=G)(conv2)
    conv2 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv2)

    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)
    if(drop>0):
        pool2 = Dropout(drop)(pool2)

    conv3 = GroupNormalization(group=G)(pool2)
    conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = GroupNormalization(group=G)(conv3)
    conv3 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv3)

    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)
    if(drop>0):
            pool3 = Dropout(drop)(pool3)

    conv4 = GroupNormalization(group=G)(pool3)
    conv4 = Conv3D(nf*16, (3, 3, 3), activation='relu', padding='same')(conv4)


    new_shape = conv3.shape.as_list()[1:-1]
    up5  = LinearResizeLayer(new_shape,name='up5')(conv4)

    up5 = concatenate([up5, conv3], axis=4) # up5 = 512 + conv3 = 256
    up5 = GroupNormalization(group=G)(up5)
    conv5 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(up5)
    up5_  = LinearResizeLayer(new_shape,name='up5_')(conv4)
    up5_ = GroupNormalization(group=G)(up5_)
    conv5_ = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(up5_)


    new_shape = conv2.shape.as_list()[1:-1]
    up6  = LinearResizeLayer(new_shape,name='up6')(conv5)

    up6 = concatenate([up6, conv2], axis=4) # up6 = 256 + conv2 = 128
    up6 = GroupNormalization(group=G)(up6)
    conv6 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up6)
    up6_  = LinearResizeLayer(new_shape,name='up6_')(conv5_)
    up6_ = GroupNormalization(group=G)(up6_)
    conv6_ = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up6_)

    new_shape = conv1.shape.as_list()[1:-1]
    up7   = LinearResizeLayer(new_shape,name='up7')(conv6)

    up7 = concatenate([up7, conv1], axis=4) # up7 = 128 + conv1 = 64
    up7 = GroupNormalization(group=G)(up7)
    conv7 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up7)

    output = Conv3D(nc, (3, 3, 3), activation='softmax', padding='same')(conv7)

    up7_   = LinearResizeLayer(new_shape,name='up7_')(conv6_)
    up7_ = GroupNormalization(group=G)(up7_)
    conv7_ = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same',name='conv7_')(up7_)
    output_ = Conv3D(nc, (3, 3, 3), padding='same')(conv7_)
    final_output= concatenate([output,output_], axis=4)
    model = Model(input_img, final_output)
    #model = Model(input_img,[output,output_])
    return model

def load_UNET3D_SLANT27_v2_groupNorm(ps1,ps2,ps3,ch,nc=4,nf=24,drop=0.5,groups=8,final_act='softmax'):   #3 levels + linear upsampling
    # See https://arxiv.org/pdf/1806.00546.pdf
    #nc: number of output classes
    #nf: number of filter
    #ch: number of channels
    G=groups
    # model UNET 3D
    pool_size=[2,2,2]

    input_img = Input(shape=(ps1, ps2, ps3, ch))     # adapt this if using `channels_first` image data format

    conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(input_img)
    conv1 = GroupNormalization(group=G)(conv1)
    #conv1 = WeightNorm(conv1)
    conv1 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv1)
    #conv1 = BatchNormalization()(conv1)
    #conv1 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)
    if(drop>0):
        pool1 = Dropout(drop)(pool1)

    conv2 = GroupNormalization(group=G)(pool1)
    #conv2 = WeightNorm(pool1)
    conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = GroupNormalization(group=G)(conv2)
    #conv2 = WeightNorm(conv2)

    conv2 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv2)
    #conv2 = BatchNormalization()(conv2)
    #conv2 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)
    if(drop>0):
        pool2 = Dropout(drop)(pool2)

    conv3 = GroupNormalization(group=G)(pool2)
    #conv3 = WeightNorm(pool2)

    conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = GroupNormalization(group=G)(conv3)
    #conv3 = WeightNorm(conv3)

    conv3 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv3)
    #conv3 = BatchNormalization()(conv3)
    #conv3 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)
    if(drop>0):
            pool3 = Dropout(drop)(pool3)

    conv4 = GroupNormalization(group=G)(pool3)
    #conv4 = WeightNorm(pool3)

    conv4 = Conv3D(nf*16, (3, 3, 3), activation='relu', name='bottleneck' ,padding='same')(conv4) #,
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(conv4)


    #up5 = UpSampling3D()(conv4)
    new_shape = conv3.shape.as_list()[1:-1]
    up5  = LinearResizeLayer(new_shape,name='up5')(conv4)

    up5 = concatenate([up5, conv3], axis=4) # up5 = 512 + conv3 = 256
    up5 = GroupNormalization(group=G)(up5)
    #up5 = WeightNorm(up5)

    conv5 = Conv3D(nf*8, (3, 3, 3), activation='relu', padding='same')(up5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(conv5)


    #up6 = UpSampling3D()(conv5)
    new_shape = conv2.shape.as_list()[1:-1]
    up6  = LinearResizeLayer(new_shape,name='up6')(conv5)

    up6 = concatenate([up6, conv2], axis=4) # up6 = 256 + conv2 = 128
    up6 = GroupNormalization(group=G)(up6)
    #up6 = WeightNorm(up6)

    conv6 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv3D(nf*2, (3, 3, 3), activation='relu', padding='same')(conv6)


    #up7 = UpSampling3D()(conv6)
    new_shape = conv1.shape.as_list()[1:-1]
    up7   = LinearResizeLayer(new_shape,name='up7')(conv6)

    up7 = concatenate([up7, conv1], axis=4) # up7 = 128 + conv1 = 64
    up7 = GroupNormalization(group=G)(up7)
    #up7 = WeightNorm(up7)

    conv7 = Conv3D(nf*4, (3, 3, 3), activation='relu', padding='same')(up7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(conv7)


    output = Conv3D(nc, (3, 3, 3), activation=final_act, padding='same')(conv7)

    model = Model(input_img, output)

    return model
