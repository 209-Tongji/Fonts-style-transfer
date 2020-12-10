import tensorflow as tf


class EncodeBlock(tf.keras.Model):
    def __init__(self, dim_out):
        super(EncodeBlock, self).__init__()

        self.act = tf.keras.layers.LeakyReLU(0.2)
        self.conv = tf.keras.layers.Conv2D(filters=dim_out,
                                           kernel_size=5,
                                           strides=(2, 2),
                                           padding='same',
                                           use_bias=True)
        self.norm = tf.keras.layers.BatchNormalization()

    def call(self, x):
        x = self.act(x)
        x = self.conv(x)
        x = self.norm(x)
        return x


class DecodeBlock(tf.keras.Model):
    def __init__(self, dim_out, use_norm=True, use_dropout=False):
        super(DecodeBlock, self).__init__()

        self.use_norm = use_norm
        self.use_dropout = use_dropout
        self.act = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2DTranspose(filters=dim_out,
                                                    kernel_size=5,
                                                    strides=(2, 2),
                                                    padding='same',
                                                    use_bias=True)
        if use_norm:
            self.norm = tf.keras.layers.BatchNormalization()
        if use_dropout:
            self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, x):
        x = self.act(x)
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        if self.use_dropout:
            x = self.dropout(x)

        return x


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.e1 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=(2, 2), padding='same', use_bias=True)
        self.e2 = EncodeBlock(128)
        self.e3 = EncodeBlock(256)
        self.e4 = EncodeBlock(512)
        self.e5 = EncodeBlock(512)
        self.e6 = EncodeBlock(512)
        self.e7 = EncodeBlock(512)
        self.e8 = EncodeBlock(512)
        self.encode = []

        self.d1 = DecodeBlock(dim_out=512, use_norm=True, use_dropout=True)
        self.d2 = DecodeBlock(dim_out=512, use_norm=True, use_dropout=True)
        self.d3 = DecodeBlock(dim_out=512, use_norm=True, use_dropout=True)
        self.d4 = DecodeBlock(dim_out=512)
        self.d5 = DecodeBlock(dim_out=256)
        self.d6 = DecodeBlock(dim_out=128)
        self.d7 = DecodeBlock(dim_out=64)
        self.d8 = DecodeBlock(dim_out=3, use_norm=False, use_dropout=False)

    def call(self, x, embeddings, embedding_ids):
        x1 = self.e1(x)  # (bs, 128, 128, 64)
        self.encode.append(x1)
        x2 = self.e2(x1)  # (bs, 64, 64, 128)
        self.encode.append(x2)
        x3 = self.e3(x2)  # (bs, 32, 32, 256)
        self.encode.append(x3)
        x4 = self.e4(x3)  # (bs, 16, 16, 512)
        self.encode.append(x4)
        x5 = self.e5(x4)  # (bs, 8, 8, 512)
        self.encode.append(x5)
        x6 = self.e6(x5)  # (bs, 4, 4, 512)
        self.encode.append(x6)
        x7 = self.e7(x6)  # (bs, 2, 2, 512)
        self.encode.append(x7)
        x8 = self.e8(x7)  # (bs, 1, 1, 512)

        local_embeddings = tf.nn.embedding_lookup(embeddings, ids=embedding_ids)
        embedded = tf.concat([x8, local_embeddings], 3)  # (bs, 1, 1, 512 + 128)

        out = self.d1(embedded)  # (bs, 2, 2, 512)
        out = tf.concat([out, self.encode[-1]], 3)  # (bs, 2, 2, 1024)
        self.encode.pop()
        out = self.d2(out)  # (bs, 4, 4, 512)
        out = tf.concat([out, self.encode[-1]], 3)  # (bs, 4, 4, 1024)
        self.encode.pop()
        out = self.d3(out)  # (bs, 8, 8, 512)
        out = tf.concat([out, self.encode[-1]], 3)  # (bs, 8, 8, 1024)
        self.encode.pop()
        out = self.d4(out)  # (bs, 16, 16, 512)
        out = tf.concat([out, self.encode[-1]], 3)  # (bs, 16, 16, 1024)
        self.encode.pop()
        out = self.d5(out)  # (bs, 32, 32, 256)
        out = tf.concat([out, self.encode[-1]], 3)  # (bs, 32, 32, 512)
        self.encode.pop()
        out = self.d6(out)  # (bs, 64, 64, 128)
        out = tf.concat([out, self.encode[-1]], 3)  # (bs, 64, 64, 256)
        self.encode.pop()
        out = self.d7(out)  # (bs, 128, 128, 64)
        out = tf.concat([out, self.encode[-1]], 3)  # (bs, 128, 128, 128)
        self.encode.pop()
        out = self.d8(out)  # (bs, 256, 256, 3)

        return x8, tf.nn.tanh(out)


def build_generator():
    generator = Generator()
    return generator


class Discriminator(tf.keras.Model):
    def __init__(self, num_domains=10, batch_size=4):
        super(Discriminator, self).__init__()
        self.batch_size = batch_size

        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=(2, 2), padding='same', use_bias=True)
        self.lrelu1 = tf.keras.layers.LeakyReLU(0.2)
        self.conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=(2, 2), padding='same', use_bias=True)
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.lrelu2 = tf.keras.layers.LeakyReLU(0.2)
        self.conv3 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=(2, 2), padding='same', use_bias=True)
        self.norm3 = tf.keras.layers.BatchNormalization()
        self.lrelu3 = tf.keras.layers.LeakyReLU(0.2)
        self.conv4 = tf.keras.layers.Conv2D(filters=512, kernel_size=5, strides=(2, 2), padding='same', use_bias=True)
        self.norm4 = tf.keras.layers.BatchNormalization()
        self.lrelu4 = tf.keras.layers.LeakyReLU(0.2)
        self.fc1 = tf.keras.layers.Dense(1)
        self.fc2 = tf.keras.layers.Dense(num_domains)

    def call(self, x):
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.lrelu3(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.lrelu4(x)

        d1 = self.fc1(tf.reshape(x, [self.batch_size, -1]))
        d2 = self.fc2(tf.reshape(x, [self.batch_size, -1]))

        return tf.nn.sigmoid(d1), d1, d2

def build_discriminator(num_domains=10, batch_size=4):
    discriminator = Discriminator(num_domains=num_domains, batch_size=batch_size)
    return discriminator