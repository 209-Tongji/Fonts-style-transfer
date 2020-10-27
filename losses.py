import tensorflow as tf


# L1 reconstruction loss
def l1_loss(real_image, gen_image, l1_lambda):
    loss = tf.reduce_mean(tf.abs(real_image - gen_image))
    return l1_lambda * loss


def standard_generator_loss(disc_gen):
    """standard gan generator loss

    Args:
        disc_gen: Discriminator judge images that generated from Generator

    """
    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return loss_obj(tf.ones_like(disc_gen), disc_gen)


def standard_discriminator_loss(disc_real, disc_gen):
    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_obj(tf.ones_like(disc_real), disc_real)
    gen_loss = loss_obj(tf.zeros_like(disc_gen), disc_gen)
    return (real_loss + gen_loss) * 0.5


