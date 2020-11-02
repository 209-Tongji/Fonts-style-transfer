import os
import sys
import argparse
import time
import tensorflow as tf

from utils import *
from dataset import get_image_dataset
from Models.DCGAN import build_unet_generator, build_discriminator
from losses import l1_loss, standard_generator_loss, standard_discriminator_loss


def parse_args():
    parser = argparse.ArgumentParser(description='train dcgan besed style transfer')

    parser.add_argument('--train-dir', type=str, default='pics/train', help='Train Images Directory')
    parser.add_argument('--val-dir', type=str, default='pics/val', help='Val Images Directory')
    parser.add_argument('--test-dir', type=str, default='pics/test', help='Test Images Directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch Size')
    parser.add_argument('--shuffle-size', type=int, default=0, help='Shuffle Size')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Initial Learning Rate')
    parser.add_argument('--l1-lambda', type=int, default=100, help='L1 Loss Lambda')
    parser.add_argument('--norm-type', type=str, default='batchnorm', help='Norm Type')
    parser.add_argument('--epochs', type=int, default=30, help='Epoch Nums')
    parser.add_argument('--output-dir', type=str, default='results', help='Output Results Directory')
    parser.add_argument('--font-classes', type=int, default=14, help='Number of Target Font Style')
    parser.add_argument('--save-tag', type=str, default='notag', help='Saving Experiment Tag')
    # parser.add_argument('--font-nums', type=int, required=True, help='Number of One Font Style')
    parser.add_argument('--output-feature', default='False', action='store_true', help='Output Hidden Layer Feature')
    args = parser.parse_args()
    return args


@tf.function
def train_step(origin_images, style_target, target_images,
               generator, discriminator,
               generator_optimizer, discriminator_optimizer, l1_lambda):

    with tf.GradientTape(persistent=True) as tape:
        gen_images = generator([origin_images, style_target], training=True)

        disc_real = discriminator([target_images, style_target], training=True)
        disc_gen = discriminator([gen_images, style_target], training=True)

        # cal loss
        gen_adversarial_loss = standard_generator_loss(disc_gen)
        gen_l1_loss = l1_loss(target_images, origin_images, l1_lambda)
        gen_loss = gen_adversarial_loss + gen_l1_loss
        disc_loss = standard_discriminator_loss(disc_real, disc_gen)

    # calculating gradients
    generator_grad = tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_grad = tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_grad, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_grad, discriminator.trainable_variables))

    return gen_adversarial_loss, gen_l1_loss


val_l1_loss = tf.keras.metrics.Mean(name='val_l1_loss')
val_ssim = tf.keras.metrics.Mean(name='val_ssim')
def val_step(origin_images, style_target, target_images, generator):
    output_images = generator([origin_images, style_target], training=False)

    target_images_renormalized = (target_images + 1.0) / 2
    output_images_renormalized = (output_images + 1.0) / 2

    ssim = tf.image.ssim(target_images_renormalized[0], output_images_renormalized[0])

    mae_loss_object = tf.keras.losses.MeanAbsoluteError()
    mae_loss = mae_loss_object(target_images_renormalized, output_images_renormalized)

    val_l1_loss(mae_loss)
    val_ssim(ssim)


def test_step(origin_images, style_target, target_images, target_path, generator, output_dir):
    output_images = generator([origin_images, style_target], training=False)

    output_images_renormalized = (output_images + 1.0) / 2
    output_images_renormalized = output_images_renormalized[0]
    output_images_renormalized = tf.cast(output_images_renormalized * 255.0, tf.uint8)
    output_images_path = target_path[0].numpy().decode('utf-8')
    output_images_path = output_images_path.split('/')
    font_class = output_images_path[-2]
    font_index = output_images_path[-1]

    plot_image(output_dir, output_images_renormalized, font_class, font_index)





if __name__ == '__main__':
    args = parse_args()
    train_dir = args.train_dir
    val_dir = args.val_dir
    test_dir = args.test_dir
    batch_size = args.batch_size
    shuffle_size = args.shuffle_size
    learning_rate = args.learning_rate
    l1_lambda = args.l1_lambda
    norm_type = args.norm_type
    epochs = args.epochs
    output_dir = args.output_dir
    font_classes = args.font_classes
    font_nums = 1000
    save_tag = args.save_tag
    output_feature = args.output_feature

    output_dir = init_out_dir(output_dir, save_tag)
    initial_logger(os.path.join(output_dir, 'log.txt'))

    logger = get_logger()

    steps_per_epoch = int(font_classes * font_nums / batch_size)

    if shuffle_size == 0:
        shuffle_size = font_classes * font_nums

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    generator = build_unet_generator(3, norm_type, True)
    discriminator = build_discriminator(norm_type, True)

    learning_rate = [learning_rate, learning_rate * 0.1, learning_rate * 0.01][:]
    boundaries = [int(0.4 * epochs * steps_per_epoch), int(0.6 * epochs * steps_per_epoch)]
    learning_rate_schedules = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, learning_rate)

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedules, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedules, beta_1=0.5)

    train_dataset = get_image_dataset(train_dir,
                                      font_classes,
                                      'train').shuffle(shuffle_size).batch(batch_size).prefetch(AUTOTUNE)

    val_dataset = get_image_dataset(val_dir, font_classes, 'val').batch(1).prefetch(AUTOTUNE)
    test_dataset = get_image_dataset(test_dir, font_classes, 'test').batch(1).prefetch(AUTOTUNE)

    epoch_timer = Timer()

    for epoch in range(epochs):
        # Reset train metrics
        train_ad_loss_sum = 0.0
        train_l1_loss_sum = 0.0
        num_batches = 0

        # Train Loop
        epoch_timer.timeit()
        for batch in train_dataset:
            origin = batch['origin']
            style_inputs = batch['style_target']
            target = batch['target']

            gen_ad_loss, gen_l1_loss = train_step(origin, style_inputs, target,
                                                  generator, discriminator,
                                                  generator_optimizer, discriminator_optimizer,
                                                  l1_lambda)
            train_ad_loss_sum += gen_ad_loss
            train_l1_loss_sum += gen_l1_loss
            num_batches += 1
        used_time_train = epoch_timer.timeit()

        train_log_template = (
            'Epoch {}, Train Adversarial Loss: {:.4f}, Train L1 Loss: {:.4f}, '
            'Train Time: {:.4f} min'
        )
        logger.info(
            train_log_template.format(
                epoch + 1,
                train_ad_loss_sum / num_batches,
                train_l1_loss_sum / num_batches,
                used_time_train / 60
            )
        )


        # Reset val metrics
        val_l1_loss.reset_states()
        val_ssim.reset_states()

        # Val Loop
        epoch_timer.timeit()
        for batch in val_dataset:
            origin = batch['origin']
            style_inputs = batch['style_target']
            target = batch['target']
            val_step(origin, style_inputs, target, generator)
        used_time_val = epoch_timer.timeit()

        val_log_template = (
            'Epoch {}, Val L1 Loss: {:.4f}, Val SSIM: {:.4f}, Val Time: {:.4f} min'
        )
        logger.info(
            val_log_template.format(
                epoch + 1,
                val_l1_loss.result(),
                val_ssim.result(),
                used_time_val / 60
            )
        )


        # Test Loop
        epoch_timer.timeit()
        for batch in test_dataset:
            origin = batch['origin']
            style_inputs = batch['style_target']
            target = batch['target']
            target_path = batch['target_path']
            test_step(origin, style_inputs, target, target_path, generator, output_dir)
        used_time_test = epoch_timer.timeit()

        test_log_template = (
            'Epoch {}, Val Time: {:.4f} min'
        )
        logger.info(
            test_log_template.format(
                epoch + 1,
                used_time_test / 60
            )
        )


        # Save Model
        generator.save(os.path.join(output_dir,
                                    'checkpoints',
                                    'epoch_{}'.format(epoch+1)))





















