import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


def __init_model_from_name__(name, input_shape=(112, 112, 3), weights="imagenet", **kwargs):
    name_lower = name.lower()
    """ Basic model """
    if name_lower == "ghostnetv1":
        from backbones import ghost_model

        xx = ghost_model.GhostNet(input_shape=input_shape, include_top=False, width=1, **kwargs)
    
    elif name_lower == "ghostnetv2":
        from backbones import ghostv2

        xx = ghostv2.GhostNetV2(stem_width=16,
                                stem_strides=1,
                                width_mul=1.3,
                                num_ghost_module_v1_stacks=2,  # num of `ghost_module` stcks on the head, others are `ghost_module_multiply`, set `-1` for all using `ghost_module`
                                input_shape=(112, 112, 3),
                                num_classes=0,
                                activation="prelu",
                                classifier_activation=None,
                                dropout=0,
                                pretrained=None,
                                model_name="ghostnetv2",
                                **kwargs)

    else:
        return None
    xx.trainable = True
    return xx


def buildin_models(
    stem_model,
    dropout=1,
    emb_shape=512,
    input_shape=(112, 112, 3),
    output_layer="GDC",
    bn_momentum=0.99,
    bn_epsilon=0.001,
    add_pointwise_conv=False,
    pointwise_conv_act="relu",
    use_bias=False,
    scale=True,
    weights="imagenet",
    **kwargs
):
    if isinstance(stem_model, str):
        xx = __init_model_from_name__(stem_model, input_shape, weights, **kwargs)
        name = stem_model
    else:
        name = stem_model.name
        xx = stem_model

    if bn_momentum != 0.99 or bn_epsilon != 0.001:
        print(">>>> Change BatchNormalization momentum and epsilon default value.")
        for ii in xx.layers:
            if isinstance(ii, keras.layers.BatchNormalization):
                ii.momentum, ii.epsilon = bn_momentum, bn_epsilon
        xx = keras.models.clone_model(xx)

    inputs = xx.inputs[0]
    nn = xx.outputs[0]

    if add_pointwise_conv:  # Model using `pointwise_conv + GDC` / `pointwise_conv + E` is smaller than `E`
        filters = nn.shape[-1] // 2 if add_pointwise_conv == -1 else 512  # Compitable with previous models...
        nn = keras.layers.Conv2D(filters, 1, use_bias=False, padding="valid", name="pw_conv")(nn)
        # nn = keras.layers.Conv2D(nn.shape[-1] // 2, 1, use_bias=False, padding="valid", name="pw_conv")(nn)
        nn = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon, name="pw_bn")(nn)
        if pointwise_conv_act.lower() == "prelu":
            nn = keras.layers.PReLU(shared_axes=[1, 2], name="pw_" + pointwise_conv_act)(nn)
        else:
            nn = keras.layers.Activation(pointwise_conv_act, name="pw_" + pointwise_conv_act)(nn)
    
    """ GDC """
    nn = keras.layers.DepthwiseConv2D(nn.shape[1], use_bias=False, name="GDC_dw")(nn)
    # nn = keras.layers.Conv2D(nn.shape[-1], nn.shape[1], use_bias=False, padding="valid", groups=nn.shape[-1])(nn)
    nn = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon, name="GDC_batchnorm")(nn)
    if dropout > 0 and dropout < 1:
        nn = keras.layers.Dropout(dropout)(nn)
    nn = keras.layers.Conv2D(emb_shape, 1, use_bias=use_bias, kernel_initializer="glorot_normal", name="GDC_conv")(nn)
    nn = keras.layers.Flatten(name="GDC_flatten")(nn)
    # nn = keras.layers.Dense(emb_shape, activation=None, use_bias=use_bias, kernel_initializer="glorot_normal", name="GDC_dense")(nn)

    # `fix_gamma=True` in MXNet means `scale=False` in Keras
    embedding = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon, scale=scale, name="pre_embedding")(nn)
    embedding_fp32 = keras.layers.Activation("linear", dtype="float32", name="embedding")(embedding)

    basic_model = keras.models.Model(inputs, embedding_fp32, name=xx.name)
    return basic_model


def add_l2_regularizer_2_model(model, weight_decay, custom_objects={}, apply_to_batch_normal=False, apply_to_bias=False):
    # https://github.com/keras-team/keras/issues/2717#issuecomment-456254176
    if 0:
        regularizers_type = {}
        for layer in model.layers:
            rrs = [kk for kk in layer.__dict__.keys() if "regularizer" in kk and not kk.startswith("_")]
            if len(rrs) != 0:
                # print(layer.name, layer.__class__.__name__, rrs)
                if layer.__class__.__name__ not in regularizers_type:
                    regularizers_type[layer.__class__.__name__] = rrs
        print(regularizers_type)

    for layer in model.layers:
        attrs = []
        if isinstance(layer, keras.layers.Dense) or isinstance(layer, keras.layers.Conv2D):
            # print(">>>> Dense or Conv2D", layer.name, "use_bias:", layer.use_bias)
            attrs = ["kernel_regularizer"]
            if apply_to_bias and layer.use_bias:
                attrs.append("bias_regularizer")
        elif isinstance(layer, keras.layers.DepthwiseConv2D):
            # print(">>>> DepthwiseConv2D", layer.name, "use_bias:", layer.use_bias)
            attrs = ["depthwise_regularizer"]
            if apply_to_bias and layer.use_bias:
                attrs.append("bias_regularizer")
        elif isinstance(layer, keras.layers.SeparableConv2D):
            # print(">>>> SeparableConv2D", layer.name, "use_bias:", layer.use_bias)
            attrs = ["pointwise_regularizer", "depthwise_regularizer"]
            if apply_to_bias and layer.use_bias:
                attrs.append("bias_regularizer")
        elif apply_to_batch_normal and isinstance(layer, keras.layers.BatchNormalization):
            # print(">>>> BatchNormalization", layer.name, "scale:", layer.scale, ", center:", layer.center)
            if layer.center:
                attrs.append("beta_regularizer")
            if layer.scale:
                attrs.append("gamma_regularizer")
        elif apply_to_batch_normal and isinstance(layer, keras.layers.PReLU):
            # print(">>>> PReLU", layer.name)
            attrs = ["alpha_regularizer"]

        for attr in attrs:
            if hasattr(layer, attr) and layer.trainable:
                setattr(layer, attr, keras.regularizers.L2(weight_decay / 2))

    # So far, the regularizers only exist in the model config. We need to
    # reload the model so that Keras adds them to each layer's losses.
    # temp_weight_file = "tmp_weights.h5"
    # model.save_weights(temp_weight_file)
    # out_model = keras.models.model_from_json(model.to_json(), custom_objects=custom_objects)
    # out_model.load_weights(temp_weight_file, by_name=True)
    # os.remove(temp_weight_file)
    # return out_model
    return keras.models.clone_model(model)


def replace_ReLU_with_PReLU(model, target_activation="PReLU", **kwargs):
    from tensorflow.keras.layers import ReLU, PReLU, Activation

    def convert_ReLU(layer):
        # print(layer.name)
        if isinstance(layer, ReLU) or (isinstance(layer, Activation) and layer.activation == keras.activations.relu):
            if target_activation == "PReLU":
                layer_name = layer.name.replace("_relu", "_prelu")
                print(">>>> Convert ReLU:", layer.name, "-->", layer_name)
                # Default initial value in mxnet and pytorch is 0.25
                return PReLU(shared_axes=[1, 2], alpha_initializer=tf.initializers.Constant(0.25), name=layer_name, **kwargs)
            elif isinstance(target_activation, str):
                layer_name = layer.name.replace("_relu", "_" + target_activation)
                print(">>>> Convert ReLU:", layer.name, "-->", layer_name)
                return Activation(activation=target_activation, name=layer_name, **kwargs)
            else:
                act_class_name = target_activation.__name__
                layer_name = layer.name.replace("_relu", "_" + act_class_name)
                print(">>>> Convert ReLU:", layer.name, "-->", layer_name)
                return target_activation(**kwargs)
        return layer

    input_tensors = keras.layers.Input(model.input_shape[1:])
    return keras.models.clone_model(model, input_tensors=input_tensors, clone_function=convert_ReLU)


def convert_to_mixed_float16(model, convert_batch_norm=False):
    policy = keras.mixed_precision.Policy("mixed_float16")
    policy_config = keras.utils.serialize_keras_object(policy)
    from tensorflow.keras.layers import InputLayer, Activation
    from tensorflow.keras.activations import linear, softmax

    def do_convert_to_mixed_float16(layer):
        if not convert_batch_norm and isinstance(layer, keras.layers.BatchNormalization):
            return layer
        if isinstance(layer, InputLayer):
            return layer
        if isinstance(layer, Activation) and layer.activation == softmax:
            return layer
        if isinstance(layer, Activation) and layer.activation == linear:
            return layer

        aa = layer.get_config()
        aa.update({"dtype": policy_config})
        bb = layer.__class__.from_config(aa)
        bb.build(layer.input_shape)
        bb.set_weights(layer.get_weights())
        return bb

    input_tensors = keras.layers.Input(model.input_shape[1:])
    mm = keras.models.clone_model(model, input_tensors=input_tensors, clone_function=do_convert_to_mixed_float16)
    if model.built:
        mm.compile(optimizer=model.optimizer, loss=model.compiled_loss, metrics=model.compiled_metrics)
        # mm.optimizer, mm.compiled_loss, mm.compiled_metrics = model.optimizer, model.compiled_loss, model.compiled_metrics
        # mm.built = True
    return mm


def convert_mixed_float16_to_float32(model):
    from tensorflow.keras.layers import InputLayer, Activation
    from tensorflow.keras.activations import linear

    def do_convert_to_mixed_float16(layer):
        if not isinstance(layer, InputLayer) and not (isinstance(layer, Activation) and layer.activation == linear):
            aa = layer.get_config()
            aa.update({"dtype": "float32"})
            bb = layer.__class__.from_config(aa)
            bb.build(layer.input_shape)
            bb.set_weights(layer.get_weights())
            return bb
        return layer

    input_tensors = keras.layers.Input(model.input_shape[1:])
    return keras.models.clone_model(model, input_tensors=input_tensors, clone_function=do_convert_to_mixed_float16)


def convert_to_batch_renorm(model):
    def do_convert_to_batch_renorm(layer):
        if isinstance(layer, keras.layers.BatchNormalization):
            aa = layer.get_config()
            aa.update({"renorm": True, "renorm_clipping": {}, "renorm_momentum": aa["momentum"]})
            bb = layer.__class__.from_config(aa)
            bb.build(layer.input_shape)
            bb.set_weights(layer.get_weights() + bb.get_weights()[-3:])
            return bb
        return layer

    input_tensors = keras.layers.Input(model.input_shape[1:])
    return keras.models.clone_model(model, input_tensors=input_tensors, clone_function=do_convert_to_batch_renorm)
