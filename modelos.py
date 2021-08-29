from keras.layers import Input, Conv3D , Dropout, concatenate,Concatenate, BatchNormalization,Conv3DTranspose, Add,MaxPooling3D,concatenate, UpSampling3D,Flatten,Dense,Activation,SpatialDropout3D, Lambda,GlobalMaxPooling3D
from keras.models import Model , load_model
from keras.activations import softmax
from keras import backend as K
from keras import optimizers
from keras.regularizers import l2
import tensorflow as tf
from keras.legacy import interfaces
from keras.optimizers import Optimizer
from keras.layers import Activation, Reshape, Lambda, dot, add, Conv1D, Conv2D, Conv3D, MaxPool1D
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers
from keras.utils import conv_utils
import numpy as np


def load_UNET3D_Sintesis_v2(ps1,ps2,ps3,ch=1,nf=4,ks=3):
	# model UNET

	input_img = Input(shape=(ps1, ps2, ps3, ch))     # adapt this if using `channels_first` image data format

	#input_img = Lambda(Subimage_decomposition,output_shape=(1,91,109,91,8))(input_img)

	# ENCODER
	conv1 = Conv3D(nf, (ks,ks,ks), activation='relu', padding='same')(input_img)
	conv1 = BatchNormalization()(conv1)
	pool1 = Conv3D(nf, (ks,ks,ks), activation='relu', padding='same',strides=2)(conv1)

	conv2 = BatchNormalization()(pool1)
	conv2 = Conv3D(nf*2, (ks,ks,ks), activation='relu', padding='same')(conv2)
	conv2 = BatchNormalization()(conv2)
	pool2 = Conv3D(nf*2, (ks,ks,ks), activation='relu', padding='same',strides=2)(conv2)

	conv3 = BatchNormalization()(pool2)
	conv3 = Conv3D(nf*4, (ks,ks,ks), activation='relu', padding='same')(conv3)
	conv3 = BatchNormalization()(conv3)
	pool3 = Conv3D(nf*4, (ks,ks,ks), activation='relu', padding='same',strides=2)(conv3)

	#LATENT
	conv4 = BatchNormalization()(pool3)
	conv4 = Conv3D(nf*8, (ks,ks,ks), activation='relu', padding='same')(conv4)
	conv4 = BatchNormalization()(conv4)
	conv4 = Conv3D(nf*8, (ks,ks,ks), activation='relu', padding='same')(conv4)
	#conv4 = Add([pool3,conv4])

	#DECODER
	up5 = LinearResizeLayer(conv3.shape.as_list()[1:-1],name='up5')(conv4)
	up5 = concatenate([up5, conv3], axis=4)
	up5 = BatchNormalization()(up5)
	conv5 = Conv3D(nf*4, (ks,ks,ks), activation='relu', padding='same')(up5)
	conv5 = BatchNormalization()(conv5)
	conv5 = Conv3D(nf*4, (ks,ks,ks), activation='relu', padding='same')(conv5)

	up6 = LinearResizeLayer(conv2.shape.as_list()[1:-1],name='up6')(conv5)
	up6 = concatenate([up6, conv2], axis=4)
	up6 = BatchNormalization()(up6)
	conv6 = Conv3D(nf*2, (ks,ks,ks), activation='relu', padding='same')(up6)
	conv6 = BatchNormalization()(conv6)
	conv6 = Conv3D(nf*2, (ks,ks,ks), activation='relu', padding='same')(conv6)

	up7 = LinearResizeLayer(conv1.shape.as_list()[1:-1],name='up7')(conv6)
	up7 = concatenate([up7, conv1], axis=4)
	up7 = BatchNormalization()(up7)
	conv7 = Conv3D(nf, (ks,ks,ks), activation='relu', padding='same')(up7)
	conv7 = BatchNormalization()(conv7)
	conv7 = Conv3D(nf, (ks,ks,ks), activation='relu', padding='same')(conv7)

	conv7 = BatchNormalization()(conv7)
	output0 = Conv3D(ch, (1, 1, 1), activation='relu',padding='same')(conv7)

	#SR block
	output = LinearResizeLayer([182,218,182],name='up8')(output0)
	#output = Lambda(Subimage_reconstruction,output_shape=(1,182,218,182,1))(output0)
	output = Conv3D(ch, (ks,ks,ks), activation='relu', padding='same')(output)
	output = BatchNormalization()(output)
	output = Conv3D(1, (1, 1, 1), activation='relu', padding='same')(output)

	model = Model(input_img, [output0,output])

	return model


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
