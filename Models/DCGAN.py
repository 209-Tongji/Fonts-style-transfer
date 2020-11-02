import tensorflow as tf

class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True
        )

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True
        )

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


def downsample(filters, size, norm_type='batchnorm', apply_norm=True):
    """Downsample an input.

    Conv2D => Norm => LeakyReLU

    Args:
        filters: number of filters
        size: filter size
        norm_type: Normalization type; either 'batchnorm' or 'instancenorm'
        apply_norm: if True, adds the norm layer

    Returns:
        Downsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))

    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
    """Upsamples an input

    Conv2DTranspose => Norm => Dropout => Relu

    Args:
        filters: number of filters
        size: filter size
        norm_type: Normalization type; either 'batchnorm' or 'instancenorm'
        apply_dropout: if True, adds the dropout layer

    Returns:
        Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                               kernel_initializer=initializer, use_bias=False))

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def build_unet_generator(output_channels, norm_type='batchnorm', target=True):
    """Modified u-net generator model

    Args:
        output_channels: Output channels
        norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.

    Returns:
        Generator model
    """

    down_stack = [
        downsample(64, 4, norm_type, apply_norm=False), # (bs, 112, 112, 64)
        downsample(128, 4, norm_type), # (bs, 56, 56, 128)
        downsample(256, 4, norm_type), # (bs, 28, 28, 256)
        downsample(512, 4, norm_type), # (bs, 14, 14, 512)
        downsample(512, 4, norm_type), # (bs, 7, 7, 512)
        #downsample(512, 4, norm_type), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, norm_type, apply_dropout=True), # (bs, 14, 14, 1024)
        #upsample(512, 4, norm_type), # (bs, 28, 28, 1024)
        upsample(256, 4, norm_type), # (bs, 28, 28, 512)
        upsample(128, 4, norm_type), # (bs, 56, 56, 256)
        upsample(64, 4, norm_type), # (bs, 112, 112, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)

    last = tf.keras.layers.Conv2DTranspose(output_channels, 4, strides=2, padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh') # (bs, 224, 224, 3)

    concat = tf.keras.layers.Concatenate()

    inp = tf.keras.Input(shape=[None, None, 3], name='input_image')
    x = inp

    if target:
        tar = tf.keras.Input(shape=[None, None, 3], name='target_image') # style target
        x = tf.keras.layers.concatenate([inp, tar]) # (bs, 224, 224, channels*2)

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling through the model
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=x)
    else:
        return tf.keras.Model(inputs=inp, outputs=x)


def build_discriminator(norm_type='batchnorm', target=True):
    """PatchGan discriminator model

    Args:
        norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'
        target: Bool, indicating whether target style image is an input or not.

    Returns:
        Discriminator model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')

    x = inp

    if target:
        tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image') # 目标风格
        x = tf.keras.layers.concatenate([inp, tar])  # (bs, 224, 224, channels*2)

    down1 = downsample(64, 4, norm_type, False)(x)  # (bs, 112, 112, 64)
    down2 = downsample(128, 4, norm_type)(down1)  # (bs, 56, 56, 128)
    down3 = downsample(256, 4, norm_type)(down2)  # (bs, 28, 28, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 30, 30, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1) # (bs, 27, 27, 512)

    if norm_type.lower() == 'batchnorm':
        norm1 = tf.keras.layers.BatchNormalization()(conv)
    elif norm_type.lower() == 'instancenorm':
        norm1 = InstanceNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 29, 29, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2) # (bs, 26, 26, 1)

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    else:
        return tf.keras.Model(inputs=inp, outputs=last)
