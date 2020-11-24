import os
import sys
import argparse
import time
import tensorflow as tf

from utils import *
from dataset import get_image_dataset
from Models.CycleGAN import build_unet_generator, build_discriminator
from losses import l1_loss, standard_generator_loss, standard_discriminator_loss
from build_json import build_train_json, build_val_json, build_test_json


def parse_args():
    parser = argparse.ArgumentParser(description='train dcgan besed style transfer')

    parser.add_argument('--train-dir', type=str, default='pics/train', help='Train Images Directory')
    parser.add_argument('--val-dir', type=str, default='pics/val', help='Val Images Directory')
    parser.add_argument('--test-dir', type=str, default='pics/test', help='Test Images Directory')
    parser.add_argument('--train-jsonfile', type=str, default=None, help='Train Images Json File')
    parser.add_argument('--val-jsonfile', type=str, default=None, help='Val Images Json File')
    parser.add_argument('--test-jsonfile', type=str, default=None, help='Test Images Json File')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch Size')
    #parser.add_argument('--shuffle-size', type=int, default=0, help='Shuffle Size')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Initial Learning Rate')
    parser.add_argument('--l1-lambda', type=int, default=10, help='L1 Loss Lambda')
    parser.add_argument('--norm-type', type=str, default='instancenorm', help='Norm Type')
    parser.add_argument('--epochs', type=int, default=30, help='Epoch Nums')
    parser.add_argument('--output-dir', type=str, default='results', help='Output Results Directory')
    #parser.add_argument('--font-classes', type=int, default=14, help='Number of Target Font Style')
    parser.add_argument('--font-classes', type=int, nargs='+', default=1, help='Index List of Target Font Style')
    parser.add_argument('--save-tag', type=str, default='notag', help='Saving Experiment Tag')
    # parser.add_argument('--font-nums', type=int, required=True, help='Number of One Font Style')
    parser.add_argument('--output-feature', default='False', action='store_true', help='Output Hidden Layer Feature')
    args = parser.parse_args()
    return args


@tf.function
def train_step(origin_images, style_target, target_images,
               generator_g, generator_f, discriminator_x, discriminator_y,
               generator_g_optimizer, generator_f_optimizer,
               discriminator_x_optimizer, discriminator_y_optimizer, l1_lambda):

    with tf.GradientTape(persistent=True) as tape:
        # origin_images -> real_x
        # target_images -> real_y

        # Generator G transform: X -> Y
        # Generator F transform: Y -> X
        fake_y = generator_g(origin_images, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(target_images, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y is for identity loss
        same_x = generator_f(origin_images, training=True)
        same_y = generator_g(target_images, training=True)

        disc_real_x = discriminator_x(origin_images, training=True)
        disc_real_y = discriminator_y(target_images, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # cal loss
        gen_g_loss = standard_generator_loss(disc_fake_y) # Generator G tries to cheat D
        gen_f_loss = standard_generator_loss(disc_fake_x) # Generator F tries to cheat D

        total_cycle_loss = l1_loss(origin_images, cycled_x, l1_lambda) + l1_loss(target_images, cycled_y, l1_lambda)

        total_gen_g_loss = gen_g_loss + total_cycle_loss + l1_loss(target_images, same_y, l1_lambda * 0.5)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + l1_loss(origin_images, same_x, l1_lambda * 0.5)

        disc_x_loss = standard_discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = standard_discriminator_loss(disc_real_y, disc_fake_y)

    # calculating gradients
    generator_g_grad = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
    generator_f_grad = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)

    discriminator_x_grad = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
    discriminator_y_grad = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)

    # apply gradients
    generator_g_optimizer.apply_gradients(zip(generator_g_grad, generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(generator_f_grad, generator_f.trainable_variables))
    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_grad, discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_grad, discriminator_y.trainable_variables))

    return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss


val_l1_loss = tf.keras.metrics.Mean(name='val_l1_loss')
val_ssim = tf.keras.metrics.Mean(name='val_ssim')
def val_step(origin_images, style_target, target_images, generator, l1_lambda):
    output_images = generator(origin_images, training=False)

    target_images_renormalized = (target_images + 1.0) / 2
    output_images_renormalized = (output_images + 1.0) / 2

    ssim = tf.image.ssim(target_images_renormalized[0], output_images_renormalized[0], max_val=1.0)

    #mae_loss_object = tf.keras.losses.MeanAbsoluteError()
    #mae_loss = mae_loss_object(target_images_renormalized, output_images_renormalized)
    mae_loss = l1_loss(target_images_renormalized[0], output_images_renormalized[0], l1_lambda)

    val_l1_loss(mae_loss)
    val_ssim(ssim)


def test_step(origin_images, style_target, target_images, target_path, generator, output_dir):
    output_images = generator(origin_images, training=False)

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
    train_jsonfile = args.train_jsonfile
    val_jsonfile = args.val_jsonfile
    test_jsonfile = args.test_jsonfile
    batch_size = args.batch_size
    #shuffle_size = args.shuffle_size
    learning_rate = args.learning_rate
    l1_lambda = args.l1_lambda
    norm_type = args.norm_type
    epochs = args.epochs
    output_dir = args.output_dir
    font_classes = args.font_classes # list
    font_nums = 1000
    save_tag = args.save_tag
    output_feature = args.output_feature

    output_dir = init_out_dir(output_dir, save_tag) # output_dir is current logging saving directory
    initial_logger(os.path.join(output_dir, 'log.txt'))

    logger = get_logger()

    steps_per_epoch = int(len(font_classes) * font_nums / batch_size)

    #if shuffle_size == 0:
    #    shuffle_size = font_classes * font_nums

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    generator_g = build_unet_generator(3, norm_type, False)
    generator_f = build_unet_generator(3, norm_type, False)
    discriminator_x = build_discriminator(norm_type, False)
    discriminator_y = build_discriminator(norm_type, False)

    learning_rate = [learning_rate, learning_rate * 0.1, learning_rate * 0.01][:]
    boundaries = [int(0.4 * epochs * steps_per_epoch), int(0.6 * epochs * steps_per_epoch)]
    learning_rate_schedules = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, learning_rate)

    generator_g_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedules, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedules, beta_1=0.5)
    discriminator_x_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedules, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedules, beta_1=0.5)

    if not train_jsonfile:
        train_jsonfile = build_train_json(train_dir, font_classes, output_dir, save_tag)
    if not val_jsonfile:
        val_jsonfile = build_val_json(val_dir, font_classes, output_dir, save_tag)
    if not test_jsonfile:
        test_jsonfile = build_test_json(test_dir, font_classes, output_dir, save_tag)
    train_dataset = get_image_dataset(train_jsonfile, 'train').batch(batch_size).prefetch(AUTOTUNE)

    val_dataset = get_image_dataset(val_jsonfile, 'val').batch(1).prefetch(AUTOTUNE)
    test_dataset = get_image_dataset(test_jsonfile, 'test').batch(1).prefetch(AUTOTUNE)

    epoch_timer = Timer()

    for epoch in range(epochs):
        # Reset train metrics
        train_gen_g_loss_sum = 0.0
        train_gen_f_loss_sum = 0.0
        train_disc_x_loss_sum = 0.0
        train_disc_y_loss_sum = 0.0
        num_batches = 0

        # Train Loop
        epoch_timer.timeit()
        for batch in train_dataset:
            origin = batch['origin']
            style_inputs = batch['style_target']
            target = batch['target']

            generator_g_loss, generator_f_loss, discriminator_x_loss, discriminator_y_loss = train_step(origin,
                                                                                                        style_inputs,
                                                                                                        target,
                                                                                                        generator_g,
                                                                                                        generator_f,
                                                                                                        discriminator_x,
                                                                                                        discriminator_y,
                                                                                                        generator_g_optimizer,
                                                                                                        generator_f_optimizer,
                                                                                                        discriminator_x_optimizer,
                                                                                                        discriminator_y_optimizer,
                                                                                                        l1_lambda)
            train_gen_g_loss_sum += generator_g_loss
            train_gen_f_loss_sum += generator_f_loss
            train_disc_x_loss_sum += discriminator_x_loss
            train_disc_y_loss_sum += discriminator_y_loss
            num_batches += 1
        used_time_train = epoch_timer.timeit()

        train_log_template = (
            'Epoch {}, Train Generator G Loss: {:.4f}, Train Generator F Loss: {:.4f}, '
            'Train Discriminator X Loss: {:.4f}, Train Discriminator Y Loss: {:.4f}, '
            'Train Time: {:.4f} min'
        )
        logger.info(
            train_log_template.format(
                epoch + 1,
                train_gen_g_loss_sum / num_batches,
                train_gen_f_loss_sum / num_batches,
                train_disc_x_loss_sum / num_batches,
                train_disc_y_loss_sum / num_batches,
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
            val_step(origin, style_inputs, target, generator_g, l1_lambda)
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
        cur_output_dir = os.path.join(output_dir, 'epoch{}'.format(epoch + 1))
        tf.io.gfile.mkdir(cur_output_dir)
        epoch_timer.timeit()
        for batch in test_dataset:
            origin = batch['origin']
            style_inputs = batch['style_target']
            target = batch['target']
            target_path = batch['target_path']
            test_step(origin, style_inputs, target, target_path, generator_g, cur_output_dir)
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

        # Save Model (weights)
        generator_g.save_weights(os.path.join(output_dir,
                                              'checkpoints_weights',
                                              'epoch_{}'.format(epoch + 1),
                                              'variables'))
        # Save Model
        #generator.save(os.path.join(output_dir,
        #                            'checkpoints',
        #                            'epoch_{}'.format(epoch+1)))