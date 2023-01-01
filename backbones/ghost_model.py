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
    #get the channel axis
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    #filters = channel axis shape
    filters = inputs.shape[channel_axis]

    reduction = _make_divisible(filters * se_ratio)

    #from None x H x W x C to None x C
    se = GlobalAveragePooling2D()(inputs)

    #Reshape None x C to None 1 x 1 x C
    se = Reshape((1, 1, filters))(se)

    #Squeeze by using C*se_ratio. The size will be 1 x 1 x C*se_ratio 
    se = Conv2D(reduction, kernel_size=1, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER)(se)
    # se = PReLU(shared_axes=[1, 2])(se)
    se = Activation("relu")(se)

    #Excitation using C filters. The size will be 1 x 1 x C
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
    nn = ghost_module(inputs, exp, add_activation=True)  # ghost1 = GhostModule(in_chs, exp, relu=True)
    # nn = BatchNormalization(axis=-1)(nn)
    # nn = Activation('relu')(nn)
    if strides > 1:
        # Extra depth conv if strides higher than 1
        nn = DepthwiseConv2D(dwkernel, strides, padding="same", use_bias=False, depthwise_initializer=CONV_KERNEL_INITIALIZER)(nn)
        nn = BatchNormalization(axis=-1)(nn)
        # nn = Activation('relu')(nn)

    if se_ratio > 0:
        # Squeeze and excite
        nn = se_module(nn, se_ratio)  # se = SqueezeExcite(exp, se_ratio=se_ratio)

    # Point-wise linear projection
    nn = ghost_module(nn, out, add_activation=False)  # ghost2 = GhostModule(exp, out, relu=False)
    # nn = BatchNormalization(axis=-1)(nn)

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

#1.3 is the width of the GhostNet as in the paper (Table 7)
def GhostNet(input_shape=(224, 224, 3), include_top=True, classes=0, width=1.3, strides=2, name="GhostNet"):
    inputs = Input(shape=input_shape)

    #out_channel equals to 20 in this case
    out_channel = _make_divisible(16 * width, 4)

    #20 filters, kernel size (3, 3), strides = 1 or 2, in this case it is 2, padding = output size same as input size
    #use_bias=False Boolean, whether the layer uses a bias vector.
    # Padding "SAME": output size is the same as input size. This requires the filter window to slip outside input map, hence the need to pad.
    # Padding "VALID": Filter window stays at valid position inside input map, so output size shrinks by filter_size - 1. No padding occurs.
    # kernel_initializer = keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal") Initializer for the kernel weights matrix
    # Initializer capable of adapting its scale to the shape of weights tensors. Check https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling
    nn = Conv2D(out_channel, (3, 3), strides=strides, padding="same", use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER)(inputs)  # padding=1
    
    # Batch normalization applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.
    # axis: Integer, the axis that should be normalized (typically the features axis). For instance, after a Conv2D layer with data_format="channels_first", set axis=1 in BatchNormalization.
    nn = BatchNormalization(axis=-1)(nn)

    # Activation function, such as tf.nn.relu, or string name of built-in activation function, such as "relu".
    nn = activation(nn)

    #The above block is as in the paper except the out_channel, in the paper they used 16

    # nn = Conv2D(960, (1, 1), strides=(1, 1), padding='same', use_bias=False)(nn)

    #As in the paper Table 1
    #The last value in exps which is 512 is the embedding shape
    dwkernels = [3, 3, 3, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5]
    exps = [16, 48, 72, 72, 120, 240, 200, 184, 184, 480, 672, 672, 960, 960, 960, 512]
    outs = [16, 24, 24, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 160, 160]
    use_ses = [0, 0, 0, 0.25, 0.25, 0, 0, 0, 0, 0.25, 0.25, 0.25, 0, 0.25, 0, 0.25]
    strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]


    pre_out = out_channel
    for dwk, stride, exp, out, se in zip(dwkernels, strides, exps, outs, use_ses):
        
        out = _make_divisible(out * width, 4) # [ 20 32 32 52 52 104 104 104 104 144 144 208 208 208 208 208 ]
        exp = _make_divisible(exp * width, 4) # [ 20 64 92 92 156 312 260 240 240 624 872 872 1248 1248 1248 664 ]
        
        #basically, it creates ghost bottlenecks until the final similar "out" number appear then and only then a shortcut will be created
        #In other words, a shortcut is created when we shift from number to another different number in "out" 
        #shortcut = [ 1 False 2 True 3 False 4 True 5 False 6 True 7 False 8 False 9 False 10 True 11 False 12 True 13 False 14 False 15 False 16 False ]
        # if out == pre_out:
        #     shortcut = False
        # else:
        #     shortcut = True
        shortcut = False if out == pre_out and stride == 1 else True

        #building ghost_bottleneck
        nn = ghost_bottleneck(nn, dwk, stride, exp, out, se, shortcut)
        pre_out = out # [ 20 32 32 52 52 104 104 104 104 144 144 208 208 208 208 208 ]

    out = _make_divisible(exps[-1] * width, 4) #664
    nn = Conv2D(out, (1, 1), strides=(1, 1), padding="valid", use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER)(nn)  # padding=0
    nn = BatchNormalization(axis=-1)(nn)
    nn = activation(nn)

    #If we want to use it for classifcation or recognition for instance
    #we include the top layer 
    if include_top:
        #As in the paper
        #Global Pooling condenses all of the feature maps into a single one, pooling all of the relevant information into a single map that can be easily understood by a single dense classification layer instead of multiple layers.
        #https://stackabuse.com/dont-use-flatten-global-pooling-for-cnns-with-tensorflow-and-keras/
        nn = GlobalAveragePooling2D()(nn)

        #Layer that reshapes inputs into the given shape.
        #tf.keras.layers.Reshape(target_shape, **kwargs)
        nn = Reshape((1, 1, int(nn.shape[1])))(nn)
        nn = Conv2D(1280, (1, 1), strides=(1, 1), padding="same", use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER)(nn)
        nn = BatchNormalization(axis=-1)(nn)
        nn = activation(nn)

        #Head containing features with the same number of classes
        nn = Conv2D(classes, (1, 1), strides=(1, 1), padding="same", use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER)(nn)

        #Removes dimensions of size 1 from the shape of a tensor. (removes redundant axis (maybe None))
        #tf.squeeze(input, axis=None, name=None)
        nn = K.squeeze(nn, 1)
        #Last layer's activation is softmax
        nn = Activation("softmax")(nn)

    return Model(inputs=inputs, outputs=nn, name=name)
