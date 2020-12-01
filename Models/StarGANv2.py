import tensorflow as tf
import tensorflow_addons as tfa
import math


# deprecated
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


class AdaptiveInstanceNormalization(tf.keras.layers.Layer):
    """Trainable Adaptive Instance Normalization Layer."""

    def __init__(self, style_dim, num_features):
        # style_dim is not needed
        super(AdaptiveInstanceNormalization, self).__init__()
        self.fc = tf.keras.layers.Dense(num_features * 2)

    def call(self, x, s):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv

        h = self.fc(s)
        for _ in range(2):
            h = tf.expand_dims(h, axis=1)
        gamma, beta = tf.split(h, num_or_size_splits=2, axis=-1)

        return (1. + gamma) * normalized + beta


class ResBlock(tf.keras.Model):
    def __init__(self, dim_in, dim_out, normalize=False, downsample=False):
        super(ResBlock, self).__init__()
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = (dim_in != dim_out)

        self.conv1 = tf.keras.layers.Conv2D(filters=dim_in, kernel_size=3, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=dim_out, kernel_size=3, padding='same')
        if self.normalize:
            self.norm1 = tfa.layers.InstanceNormalization()
            self.norm2 = tfa.layers.InstanceNormalization()
        if self.learned_sc:
            self.conv1x1 = tf.keras.layers.Conv2D(filters=dim_out, kernel_size=1, padding='same', use_bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = tf.keras.layers.AveragePooling2D()(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = self.conv1(x)
        if self.downsample:
            x = tf.keras.layers.AveragePooling2D()(x)
        if self.normalize:
            x = self.norm2(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        return x

    def call(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2) # unit variance


class AdaInResBlock(tf.keras.Model):
    def __init__(self, dim_in, dim_out, style_dim=64, upsample=False):
        super(AdaInResBlock, self).__init__()
        self.upsample = upsample
        self.learned_sc = (dim_in != dim_out)

        self.conv1 = tf.keras.layers.Conv2D(filters=dim_out, kernel_size=3, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=dim_out, kernel_size=3, padding='same')

        self.norm1 = AdaptiveInstanceNormalization(style_dim, dim_in)
        self.norm2 = AdaptiveInstanceNormalization(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = tf.keras.layers.Conv2D(filters=dim_out, kernel_size=1,
                                                  padding='same', use_bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = tf.keras.layers.UpSampling2D()(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        if self.upsample:
            x = tf.keras.layers.UpSampling2D()(x)
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        return x

    def call(self, x, s):
        x = self._residual(x, s) + self._shortcut(x)
        return x / math.sqrt(2)


class Generator(tf.keras.Model):
    def __init__(self, dim_in=64, style_dim=64, max_conv_dim=512, repeat_num=4):
        super(Generator, self).__init__()
        self.from_rgb = tf.keras.layers.Conv2D(filters=dim_in, kernel_size=3, padding='same')
        self.encode = []
        self.decode = []
        self.to_rgb = tf.keras.Sequential([
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2D(filters=3, kernel_size=1)
        ])

        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(ResBlock(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(0, AdaInResBlock(dim_out, dim_in, style_dim, upsample=True))
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(ResBlock(dim_out, dim_out, normalize=True))
            self.decode.insert(0, AdaInResBlock(dim_out, dim_out, style_dim))

    def call(self, x, s):
        x = self.from_rgb(x)
        for block in self.encode:
            x = block(x)
        for block in self.decode:
            x = block(x, s)
        return self.to_rgb(x)


def build_generator(dim_in=64, style_dim=64, max_conv_dim=512, repeat_num=4):
    image_input = tf.keras.Input((None, None, 3))
    style_code_input = tf.keras.Input((style_dim,))
    genenator = Generator(dim_in=dim_in, style_dim=style_dim, max_conv_dim=max_conv_dim, repeat_num=repeat_num)
    output = genenator(image_input, style_code_input)
    return tf.keras.Model(inputs=[image_input, style_code_input], outputs=output)


class MappingNetwork(tf.keras.Model):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=3):
        super(MappingNetwork, self).__init__()
        self.shared = tf.keras.Sequential()
        self.shared.add(tf.keras.Input(shape=(latent_dim,)))
        for _ in range(4):
            self.shared.add(tf.keras.layers.Dense(512))
            self.shared.add(tf.keras.layers.ReLU())
        self.unshared = []
        for _ in range(num_domains):
            self.unshared += [tf.keras.Sequential([
                tf.keras.layers.Dense(512),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(512),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(512),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(style_dim),
            ])]

    def call(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = tf.stack(out, axis=1)  # (batch, num_domains, style_dim)
        res = []
        for i in range(y.shape[0]):
            y_index = y[i]
            res.append(out[i, y_index])  # (batch, style_dim)
        return tf.stack(res, axis=0)

def build_mapping_network(latent_dim=16, style_dim=64, num_domains=3):
    mapping_network = MappingNetwork(latent_dim, style_dim, num_domains)
    return mapping_network


class StyleEncoder(tf.keras.Model):
    def __init__(self, dim_in=64, style_dim=64, num_domains=3, max_conv_dim=512, repeat_nums=5):
        super(StyleEncoder, self).__init__()
        self.shared = []
        self.shared.append(tf.keras.layers.Conv2D(filters=dim_in, kernel_size=3, padding='same'))

        for _ in range(repeat_nums):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.shared.append(ResBlock(dim_in, dim_out, downsample=True))
            dim_in = dim_out

        self.shared.append(tf.keras.layers.LeakyReLU(0.2))
        self.shared.append(tf.keras.layers.Conv2D(filters=dim_out, kernel_size=7, strides=1, padding='valid'))
        self.shared.append(tf.keras.layers.LeakyReLU(0.2))

        self.unshared = []
        for _ in range(num_domains):
            self.unshared.append(tf.keras.layers.Dense(style_dim))

    def call(self, x, y):
        for block in self.shared:
            x = block(x)
        x = tf.reshape(x, shape=[x.shape[0], -1])

        out = []
        for layer in self.unshared:
            out += [layer(x)]
        out = tf.stack(out, axis=1)  # (batch, num_domains, style_dim)
        res = []
        for i in range(y.shape[0]):
            y_index = y[i]
            res.append(out[i, y_index])  # (batch, style_dim)
        return tf.stack(res, axis=0)

def build_style_encoder(dim_in=64, style_dim=64, num_domains=3, max_conv_dim=512, repea_nums=5):
    style_encoder = StyleEncoder(dim_in, style_dim, num_domains, max_conv_dim, repea_nums)
    return style_encoder



class Discriminator(tf.keras.Model):
    def __init__(self, dim_in=64, num_domains=3, max_conv_dim=512, repeat_nums=5):
        super(Discriminator, self).__init__()
        self.blocks = []
        self.blocks.append(tf.keras.layers.Conv2D(filters=dim_in, kernel_size=3, padding='same'))

        for _ in range(repeat_nums):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.blocks.append(ResBlock(dim_in, dim_out, downsample=True))
            dim_in = dim_out

        self.blocks.append(tf.keras.layers.LeakyReLU(0.2))
        self.blocks.append(tf.keras.layers.Conv2D(filters=dim_out, kernel_size=7, strides=1, padding='valid'))
        self.blocks.append(tf.keras.layers.LeakyReLU(0.2))

        # domain's results are differentiated by channels
        self.blocks.append(tf.keras.layers.Conv2D(filters=num_domains, kernel_size=1, strides=1, padding='valid'))

    def call(self, x, y):
        for block in self.blocks:
            x = block(x)
        x = tf.reshape(x, shape=[x.shape[0], -1])  # (batch, num_domains)
        res = []
        for i in range(y.shape[0]):
            y_index = y[i]
            res.append(x[i, y_index])  # (batch, )
        return tf.stack(res, axis=0)

def build_discriminator(dim_in=64, num_domains=3, max_conv_dim=512, repea_nums=5):
    discriminator = Discriminator(dim_in, num_domains, max_conv_dim, repea_nums)
    return discriminator






