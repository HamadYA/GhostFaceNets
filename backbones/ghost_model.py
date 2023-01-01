"""
The main GhostNet architecture as specified in "GhostNet: More Features from Cheap Operations"
Paper:
https://arxiv.org/pdf/1911.11907.pdf
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    DepthwiseConv2D,
    GlobalAveragePooling2D,
    Input,
    PReLU,
    Reshape,
    Multiply,
)
import math

CONV_KERNEL_INITIALIZER = keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
# CONV_KERNEL_INITIALIZER = 'glorot_uniform'


def _make_divisible(v, divisor=4, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def activation(inputs):
    return Activation("relu")(inputs)
    # return PReLU(shared_axes=[1, 2], alpha_initializer=tf.initializers.Constant(0.25))(inputs)


def se_module(inputs, se_ratio=0.25):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = inputs.shape[channel_axis]
    reduction = _make_divisible(filters * se_ratio)
    se = GlobalAveragePooling2D()(inputs)
    se = Reshape((1, 1, filters))(se)
    se = Conv2D(reduction, kernel_size=1, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER)(se)
    # se = PReLU(shared_axes=[1, 2])(se)
    se = Activation("relu")(se)
    se = Conv2D(filters, kernel_size=1, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER)(se)
    se = Activation("hard_sigmoid")(se)
    return Multiply()([inputs, se])


def ghost_module(inputs, out, convkernel=1, dwkernel=3, add_activation=True):
    # conv_out_channel = math.ceil(out * 1.0 / 2)
    conv_out_channel = out // 2
    # tf.print("[ghost_module] out:", out, "conv_out_channel:", conv_out_channel)
    cc = Conv2D(conv_out_channel, convkernel, use_bias=False, strides=(1, 1), padding="same", kernel_initializer=CONV_KERNEL_INITIALIZER)(
        inputs
    )  # padding=kernel_size//2
    cc = BatchNormalization(axis=-1)(cc)
    if add_activation:
        cc = activation(cc)

    channel = int(out - conv_out_channel)
    nn = DepthwiseConv2D(dwkernel, 1, padding="same", use_bias=False, depthwise_initializer=CONV_KERNEL_INITIALIZER)(cc)  # padding=dw_size//2
    nn = BatchNormalization(axis=-1)(nn)
    if add_activation:
        nn = activation(nn)
    # tf.print("[ghost_module] nn.shape:", nn.shape, "channel:", channel)
    # nn = nn[:, :, :, :channel]
    # nn = tf.gather(nn, range(channel), axis=-1)
    return Concatenate()([cc, nn])


def ghost_bottleneck(inputs, dwkernel, strides, exp, out, se_ratio=0, shortcut=True):

    #The first ghost module block, it includes BN ReLU as in the paper
    nn = ghost_module(inputs, exp, add_activation=True)  # ghost1 = GhostModule(in_chs, exp, relu=True)
    # nn = BatchNormalization(axis=-1)(nn)
    # nn = Activation('relu')(nn)

    #If the strides are above 1, add depthwise convolution with the same number of strides, and BN
    if strides > 1:
        # Extra depth conv if strides higher than 1
        nn = DepthwiseConv2D(dwkernel, strides, padding="same", use_bias=False, depthwise_initializer=CONV_KERNEL_INITIALIZER)(nn)
        nn = BatchNormalization(axis=-1)(nn)
        # nn = Activation('relu')(nn)

    #Apply Squeeze and Excite (SE) modules when se_ratio is above 0 (as in the paper)
    #The SE module is added after the first ghost_module (and if strides is above 1, we add depthwise convolution and BN).
    if se_ratio > 0:
        # Squeeze and excite
        nn = se_module(nn, se_ratio)  # se = SqueezeExcite(exp, se_ratio=se_ratio)

    #The second ghost module block does not have activation function as in the paper
    #BN layer after ghost module block (the BN is added in the ghost module block function)
    
    # Point-wise linear projection
    nn = ghost_module(nn, out, add_activation=False)  # ghost2 = GhostModule(exp, out, relu=False)
    # nn = BatchNormalization(axis=-1)(nn) #No need, already added in ghost module function

    #
    if shortcut:
        xx = DepthwiseConv2D(dwkernel, strides, padding="same", use_bias=False, depthwise_initializer=CONV_KERNEL_INITIALIZER)(
            inputs
        )  # padding=(dw_kernel_size-1)//2
        xx = BatchNormalization(axis=-1)(xx)
        xx = Conv2D(out, (1, 1), strides=(1, 1), padding="valid", use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER)(xx)  # padding=0
        xx = BatchNormalization(axis=-1)(xx)
    else:
        xx = inputs
    return Add()([xx, nn])


def GhostNet(input_shape=(224, 224, 3), include_top=True, classes=0, width=1.3, strides=2, name="GhostNet"):
    inputs = Input(shape=input_shape)
    out_channel = _make_divisible(16 * width, 4) #20
    nn = Conv2D(out_channel, (3, 3), strides=strides, padding="same", use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER)(inputs)  # padding=1
    nn = BatchNormalization(axis=-1)(nn)
    nn = activation(nn)
    # nn = Conv2D(960, (1, 1), strides=(1, 1), padding='same', use_bias=False)(nn)

    dwkernels = [3, 3, 3, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5]
    exps = [16, 48, 72, 72, 120, 240, 200, 184, 184, 480, 672, 672, 960, 960, 960, 512]
    outs = [16, 24, 24, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 160, 160]
    use_ses = [0, 0, 0, 0.25, 0.25, 0, 0, 0, 0, 0.25, 0.25, 0.25, 0, 0.25, 0, 0.25]
    strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]

    pre_out = out_channel
    for dwk, stride, exp, out, se in zip(dwkernels, strides, exps, outs, use_ses):
        out = _make_divisible(out * width, 4)
        exp = _make_divisible(exp * width, 4)
        shortcut = False if out == pre_out and stride == 1 else True
        nn = ghost_bottleneck(nn, dwk, stride, exp, out, se, shortcut)
        pre_out = out

    out = _make_divisible(exps[-1] * width, 4)
    nn = Conv2D(out, (1, 1), strides=(1, 1), padding="valid", use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER)(nn)  # padding=0
    nn = BatchNormalization(axis=-1)(nn)
    nn = activation(nn)

    if include_top:
        nn = GlobalAveragePooling2D()(nn)
        nn = Reshape((1, 1, int(nn.shape[1])))(nn)
        nn = Conv2D(1280, (1, 1), strides=(1, 1), padding="same", use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER)(nn)
        nn = BatchNormalization(axis=-1)(nn)
        nn = activation(nn)

        nn = Conv2D(classes, (1, 1), strides=(1, 1), padding="same", use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER)(nn)
        nn = K.squeeze(nn, 1)
        nn = Activation("softmax")(nn)
    return Model(inputs=inputs, outputs=nn, name=name)
