import tensorflow as tf


# L1 reconstruction loss
def l1_loss(real_image, gen_image, l1_lambda):
    loss = tf.reduce_mean(tf.abs(real_image - gen_image))
    return l1_lambda * loss



# Standard GAN
def standard_generator_loss(disc_gen):
    """standard gan generator loss

    Args:
        disc_gen: Discriminator judge images that generated from Generator

    """
    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return loss_obj(tf.ones_like(disc_gen), disc_gen)


def standard_discriminator_loss(disc_real, disc_gen):
    """standard gan discriminator loss

    Args:
        disc_real: Discriminator judge images that is real
        disc_gen: Discriminator judge images that generated from Generator

    """

    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_obj(tf.ones_like(disc_real), disc_real)
    gen_loss = loss_obj(tf.zeros_like(disc_gen), disc_gen)
    return (real_loss + gen_loss) * 0.5


# Relativistic Standard GAN
def relativistic_standard_generator_loss(disc_real, disc_gen):
    """relativistic standard generator loss

    Args:
        disc_real: Discriminator judge images that is real
        disc_gen: Discriminator judge images that generated from Generator

    """
    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return loss_obj(tf.ones_like(disc_real), (disc_gen - disc_real))


def relativistic_standard_discriminator_loss(disc_real, disc_gen):
    """relativistic standard discriminator loss

    Args:
        disc_real: Discriminator judge images that is real
        disc_gen: Discriminator judge images that generated from Generator

    """
    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return loss_obj(tf.ones_like(disc_real), (disc_real - disc_gen))



# Relativistic Average Standard GAN - RaSGAN
def relativistic_average_standard_generator_loss(disc_real, disc_gen):
    """relativistic average standard generator loss

    Args:
        disc_real: Discriminator judge images that is real
        disc_gen: Discriminator judge images that generated from Generator

    """

    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_obj(tf.zeros_like(disc_real), disc_real - tf.reduce_mean(disc_gen))
    gen_loss = loss_obj(tf.ones_like(disc_gen), disc_gen - tf.reduce_mean(disc_real))
    return (real_loss + gen_loss) / 2


def relativistic_average_standard_discriminator_loss(disc_real, disc_gen):
    """relativistic average standard discriminator loss

    Args:
        disc_real: Discriminator judge images that is real
        disc_gen: Discriminator judge images that generated from Generator

    """

    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_obj(tf.ones_like(disc_real), disc_real - tf.reduce_mean(disc_gen))
    gen_loss = loss_obj(tf.zeros_like(disc_gen), disc_gen - tf.reduce_mean(disc_real))
    return (real_loss + gen_loss) / 2




