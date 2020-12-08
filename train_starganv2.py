import os
import sys
import argparse
import time
import tensorflow as tf
from tqdm import tqdm

from utils import *
from dataset import get_image_dataset
from Models.StarGANv2 import build_generator, build_mapping_network, build_style_encoder, build_discriminator
from losses import l1_loss, standard_generator_loss, multi_discriminator_loss
from build_json import build_train_json, build_val_json, build_test_json


def parse_args():
    parser = argparse.ArgumentParser(description='train dcgan besed style transfer')

    parser.add_argument('--train-dir', type=str, default='pics/train', help='Train Images Directory')
    parser.add_argument('--val-dir', type=str, default='pics/val', help='Val Images Directory')
    parser.add_argument('--test-dir', type=str, default='pics/test', help='Test Images Directory')
    parser.add_argument('--train-jsonfile', type=str, default=None, help='Train Images Json File')
    parser.add_argument('--val-jsonfile', type=str, default=None, help='Val Images Json File')
    parser.add_argument('--test-jsonfile', type=str, default=None, help='Test Images Json File')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch Size')
    #parser.add_argument('--shuffle-size', type=int, default=0, help='Shuffle Size')
    parser.add_argument('--g-learning-rate', type=float, default=1e-4, help='Initial G Learning Rate')
    parser.add_argument('--d-learning-rate', type=float, default=1e-4, help='Initial D Learning Rate')
    parser.add_argument('--e-learning-rate', type=float, default=1e-4, help='Initial E Learning Rate')
    parser.add_argument('--f-learning-rate', type=float, default=1e-6, help='Initial F Learning Rate')
    parser.add_argument('--l1-lambda', type=int, default=1, help='L1 Loss Lambda')
    parser.add_argument('--cycle-lambda', type=int, default=1, help='Cycle Loss Lambda')
    parser.add_argument('--style-recons-lambda', type=int, default=1, help='Style Reconstruction Loss Lambda')
    parser.add_argument('--style-div-lambda', type=int, default=1, help='Style Diversity Loss Lambda')
    #parser.add_argument('--norm-type', type=str, default='instancenorm', help='Norm Type')
    parser.add_argument('--epochs', type=int, default=30, help='Epoch Nums')
    parser.add_argument('--output-dir', type=str, default='results', help='Output Results Directory')
    #parser.add_argument('--font-classes', type=int, default=14, help='Number of Target Font Style')
    parser.add_argument('--font-classes', type=int, nargs='+', default=1, help='Index List of Target Font Style')
    parser.add_argument('--save-tag', type=str, default='notag', help='Saving Experiment Tag')
    # parser.add_argument('--font-nums', type=int, required=True, help='Number of One Font Style')
    # parser.add_argument('--output-feature', default='False', action='store_true', help='Output Hidden Layer Feature')
    args = parser.parse_args()
    return args


@tf.function
def first_call(origin_images, style_target, target_images, target_class, origin_class, z_target, z2,
               generator, discriminator, mapping_network, style_encoder,
               generator_optimizer, discriminator_optimizer, mapping_network_optimizer, style_encoder_optimizer,
               cycle_lambda, style_recons_lambda, style_div_lambda):
    with tf.GradientTape(persistent=True) as tape:
        disc_real_source = discriminator(origin_images, origin_class, training=False)

        style_code_target = mapping_network(z_target, target_class, training=False)

        style_code_target_gen = style_encoder(style_target, target_class, training=False)

        gen_images = generator(origin_images, style_code_target, training=False)

    return None


@tf.function
def train_discriminator_step(origin_images, style_target, target_images, target_class, origin_class, z_target,
               generator, discriminator, mapping_network, style_encoder, discriminator_optimizer):

    with tf.GradientTape(persistent=True) as tape:

        disc_real_source = discriminator(origin_images, origin_class, training=True)
        disc_real_target = discriminator(target_images, target_class, training=True)

        if z_target is not None:
            style_code_target = mapping_network(z_target, target_class, training=False)
        else:
            # style target is used
            style_code_target = style_encoder(style_target, target_class, training=False)
        gen_images = generator(origin_images, style_code_target, training=False)

        disc_fake = discriminator(gen_images, target_class, training=True)

        # cal disc loss
        disc_loss = multi_discriminator_loss(disc_real_source, disc_real_target, disc_fake)


    # calculating gradients
    discriminator_grad = tape.gradient(disc_loss, discriminator.trainable_variables)

    # apply gradients
    discriminator_optimizer.apply_gradients(zip(discriminator_grad, discriminator.trainable_variables))

    return disc_loss


@tf.function
def train_generator_step(origin_images, style_target, target_images, target_class, origin_class, z_target, z2,
                         generator, discriminator, mapping_network, style_encoder,
                         generator_optimizer, mapping_network_optimizer, style_encoder_optimizer,
                         cycle_lambda, style_recons_lambda, style_div_lambda):

    with tf.GradientTape(persistent=True) as tape:

        # adversarial loss
        if z_target is not None:
            style_code_target = mapping_network(z_target, target_class, training=True)
        else:
            style_code_target = style_encoder(style_target, target_class, training=True)

        gen_images = generator(origin_images, style_code_target, training=True)

        disc_gen = discriminator(gen_images, target_class, training=False)

        loss_adv = standard_generator_loss(disc_gen)

        # style reconstruction loss
        style_code_pred = style_encoder(gen_images, target_class, training=True)

        loss_style = l1_loss(style_code_pred, style_code_target)

        # diversity sensitive loss
        if z2 is not None:
            style_code_target_2 = mapping_network(z2, target_class, training=True)
        else:
            style_code_target_2 = style_encoder(target_images, target_class, training=True)
        gen_images_2 = generator(origin_images, style_code_target_2, training=True)
        loss_diversity = l1_loss(gen_images, gen_images_2)

        # cycle-consistency loss
        style_code_origin = style_encoder(origin_images, origin_class, training=True)
        recons_images = generator(gen_images, style_code_origin, training=True)
        loss_cycle = l1_loss(origin_images, recons_images)

        # cal total loss
        loss = loss_adv + style_recons_lambda * loss_style - style_div_lambda * loss_diversity + cycle_lambda * loss_cycle

    # calculating gradients
    generator_grad = tape.gradient(loss, generator.trainable_variables)
    if z_target is not None:
        mapping_network_grad = tape.gradient(loss, mapping_network.trainable_variables)
        style_encoder_grad = tape.gradient(loss, style_encoder.trainable_variables)

    # apply gradients
    generator_optimizer.apply_gradients(zip(generator_grad, generator.trainable_variables))
    if z_target is not None:
        mapping_network_optimizer.apply_gradients(zip(mapping_network_grad, mapping_network.trainable_variables))
        style_encoder_optimizer.apply_gradients(zip(style_encoder_grad, style_encoder.trainable_variables))

    return loss



val_l1_loss = tf.keras.metrics.Mean(name='val_l1_loss')
val_ssim = tf.keras.metrics.Mean(name='val_ssim')
def val_step(origin_images, style_target, target_images, target_class, generator, style_encoder, l1_lambda):
    style_code = style_encoder(style_target, target_class, training=False)
    output_images = generator(origin_images, style_code, training=False)

    target_images_renormalized = (target_images + 1.0) / 2
    output_images_renormalized = (output_images + 1.0) / 2

    ssim = tf.image.ssim(target_images_renormalized[0], output_images_renormalized[0], max_val=1.0)

    #mae_loss_object = tf.keras.losses.MeanAbsoluteError()
    #mae_loss = mae_loss_object(target_images_renormalized, output_images_renormalized)
    mae_loss = l1_loss(target_images_renormalized[0], output_images_renormalized[0]) * l1_lambda

    val_l1_loss(mae_loss)
    val_ssim(ssim)


def test_step(origin_images, style_target, target_images, target_class, target_path, generator, style_encoder, output_dir):
    style_code = style_encoder(style_target, target_class, training=False)
    output_images = generator(origin_images, style_code, training=False)

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
    g_learning_rate = args.g_learning_rate
    d_learning_rate = args.d_learning_rate
    e_learning_rate = args.e_learning_rate
    f_learning_rate = args.f_learning_rate
    l1_lambda = args.l1_lambda
    cycle_lambda = args.cycle_lambda
    style_recons_lambda = args.style_recons_lambda
    style_div_lambda = args.style_div_lambda
    #norm_type = args.norm_type
    epochs = args.epochs
    output_dir = args.output_dir
    font_classes = args.font_classes # list
    num_domains = len(font_classes) + 1
    font_nums = 1000
    save_tag = args.save_tag
    #output_feature = args.output_feature

    output_dir = init_out_dir(output_dir, save_tag) # output_dir is current logging saving directory
    initial_logger(os.path.join(output_dir, 'log.txt'))

    logger = get_logger()

    steps_per_epoch = int(len(font_classes) * font_nums / batch_size)

    #if shuffle_size == 0:
    #    shuffle_size = font_classes * font_nums

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    generator = build_generator(dim_in=64, style_dim=64, max_conv_dim=512, repeat_num=4)
    discriminator = build_discriminator(dim_in=64, num_domains=num_domains, max_conv_dim=512, repea_nums=6)
    mapping_network = build_mapping_network(latent_dim=16, style_dim=64, num_domains=num_domains)
    style_encoder = build_style_encoder(dim_in=64, style_dim=64, num_domains=num_domains, max_conv_dim=512, repea_nums=6)

    learning_rate_g = [g_learning_rate, g_learning_rate * 0.1, g_learning_rate * 0.01][:]
    boundaries_g = [int(0.4 * epochs * steps_per_epoch), int(0.6 * epochs * steps_per_epoch)]
    learning_rate_schedules_g = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries_g, learning_rate_g)
    learning_rate_d = [d_learning_rate, d_learning_rate * 0.1, d_learning_rate * 0.01][:]
    boundaries_d = [int(0.4 * epochs * steps_per_epoch), int(0.6 * epochs * steps_per_epoch)]
    learning_rate_schedules_d = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries_d, learning_rate_d)
    learning_rate_e = [e_learning_rate, e_learning_rate * 0.1, e_learning_rate * 0.01][:]
    boundaries_e = [int(0.4 * epochs * steps_per_epoch), int(0.6 * epochs * steps_per_epoch)]
    learning_rate_schedules_e = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries_e, learning_rate_e)
    learning_rate_f = [f_learning_rate, f_learning_rate * 0.1, f_learning_rate * 0.01][:]
    boundaries_f = [int(0.4 * epochs * steps_per_epoch), int(0.6 * epochs * steps_per_epoch)]
    learning_rate_schedules_f = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries_f, learning_rate_f)


    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedules_g, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedules_d, beta_1=0.5)
    style_encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedules_e, beta_1=0.5)
    mapping_network_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedules_f, beta_1=0.5)

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

    demo_ds = iter(train_dataset)
    batch_first = next(demo_ds)
    origin = batch_first['origin']
    style_inputs = batch_first['style_target']
    target = batch_first['target']
    target_class = batch_first['target_class']
    origin_class = tf.zeros_like(target_class)
    z_target = tf.random.normal(shape=[target_class.shape[0], 16])
    z2 = tf.random.normal(shape=[target_class.shape[0], 16])
    first_call(origin,
               style_inputs,
               target,
               target_class,
               origin_class,
               z_target,
               z2,
               generator,
               discriminator,
               mapping_network,
               style_encoder,
               generator_optimizer,
               discriminator_optimizer,
               mapping_network_optimizer,
               style_encoder_optimizer,
               cycle_lambda,
               style_recons_lambda,
               style_div_lambda)
    del demo_ds
    del batch_first

    for epoch in range(epochs):
        # Reset train metrics
        train_gen_z_loss_sum = 0.0
        train_gen_s_loss_sum = 0.0
        train_disc_z_loss_sum = 0.0
        train_disc_s_loss_sum = 0.0
        num_batches = 0

        # Train Loop
        epoch_timer.timeit()
        for batch in tqdm(train_dataset):
            origin = batch['origin']
            style_inputs = batch['style_target']
            target = batch['target']
            target_class = batch['target_class']
            origin_class = tf.zeros_like(target_class)
            z_target = tf.random.normal(shape=[target_class.shape[0], 16])
            z2 = tf.random.normal(shape=[target_class.shape[0], 16])

            discriminator_z_loss = train_discriminator_step(origin_images=origin,
                                                            style_target=style_inputs,
                                                            target_images=target,
                                                            target_class=target_class,
                                                            origin_class=origin_class,
                                                            z_target=z_target,
                                                            generator=generator,
                                                            discriminator=discriminator,
                                                            mapping_network=mapping_network,
                                                            style_encoder=style_encoder,
                                                            discriminator_optimizer=discriminator_optimizer)

            discriminator_s_loss = train_discriminator_step(origin_images=origin,
                                                            style_target=style_inputs,
                                                            target_images=target,
                                                            target_class=target_class,
                                                            origin_class=origin_class,
                                                            z_target=None,
                                                            generator=generator,
                                                            discriminator=discriminator,
                                                            mapping_network=mapping_network,
                                                            style_encoder=style_encoder,
                                                            discriminator_optimizer=discriminator_optimizer)

            generator_z_loss = train_generator_step(origin_images=origin,
                                                    style_target=style_inputs,
                                                    target_images=target,
                                                    target_class=target_class,
                                                    origin_class=origin_class,
                                                    z_target=z_target,
                                                    z2=z2,
                                                    generator=generator,
                                                    discriminator=discriminator,
                                                    mapping_network=mapping_network,
                                                    style_encoder=style_encoder,
                                                    generator_optimizer=generator_optimizer,
                                                    mapping_network_optimizer=mapping_network_optimizer,
                                                    style_encoder_optimizer=style_encoder_optimizer,
                                                    cycle_lambda=cycle_lambda,
                                                    style_recons_lambda=style_recons_lambda,
                                                    style_div_lambda=style_div_lambda)

            generator_s_loss = train_generator_step(origin_images=origin,
                                                    style_target=style_inputs,
                                                    target_images=target,
                                                    target_class=target_class,
                                                    origin_class=origin_class,
                                                    z_target=None,
                                                    z2=None,
                                                    generator=generator,
                                                    discriminator=discriminator,
                                                    mapping_network=mapping_network,
                                                    style_encoder=style_encoder,
                                                    generator_optimizer=generator_optimizer,
                                                    mapping_network_optimizer=mapping_network_optimizer,
                                                    style_encoder_optimizer=style_encoder_optimizer,
                                                    cycle_lambda=cycle_lambda,
                                                    style_recons_lambda=style_recons_lambda,
                                                    style_div_lambda=style_div_lambda)

            train_disc_z_loss_sum += discriminator_z_loss
            train_disc_s_loss_sum += discriminator_s_loss
            train_gen_z_loss_sum += generator_z_loss
            train_gen_s_loss_sum += generator_s_loss
            num_batches += 1
        used_time_train = epoch_timer.timeit()

        train_log_template = (
            'Epoch {}, Train Discriminator With z Loss: {:.4f}, Train Discriminator With s Loss: {:.4f}, '
            'Train Generator With z Loss: {:.4f}, Train Generator With s Loss: {:.4f}, '
            'Train Time: {:.4f} min'
        )
        logger.info(
            train_log_template.format(
                epoch + 1,
                train_disc_z_loss_sum / num_batches,
                train_disc_s_loss_sum / num_batches,
                train_gen_z_loss_sum / num_batches,
                train_gen_s_loss_sum / num_batches,
                used_time_train / 60
            )
        )


        # Reset val metrics
        val_l1_loss.reset_states()
        val_ssim.reset_states()

        # Val Loop
        epoch_timer.timeit()
        for batch in tqdm(val_dataset):
            origin = batch['origin']
            style_inputs = batch['style_target']
            target = batch['target']
            target_class = batch['target_class']

            val_step(origin, style_inputs, target, target_class, generator, style_encoder, l1_lambda)
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
        for batch in tqdm(test_dataset):
            origin = batch['origin']
            style_inputs = batch['style_target']
            target = batch['target']
            target_path = batch['target_path']
            target_class = batch['target_class']
            test_step(origin, style_inputs, target, target_class, target_path, generator, style_encoder, cur_output_dir)
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
        save_path = os.path.join(output_dir, 'checkpoints_weights', 'epoch_{}'.format(epoch + 1))
        tf.io.gfile.makedirs(save_path)

        generator.save_weights(os.path.join(save_path, 'generator_weights.h5'))
        style_encoder.save_weights(os.path.join(save_path, 'style_encoder_weights.h5'))


        # Save Model
        #generator.save(os.path.join(output_dir,
        #                            'checkpoints',
        #                            'epoch_{}'.format(epoch+1)))