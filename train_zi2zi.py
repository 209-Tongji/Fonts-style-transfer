import os
import sys
import argparse
import time
import tensorflow as tf
from tqdm import tqdm

from utils import *
from dataset import get_image_dataset
from Models.Zi2Zi import build_generator, build_discriminator
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
    parser.add_argument('--g-learning-rate', type=float, default=0.001, help='Initial G Learning Rate')
    parser.add_argument('--d-learning-rate', type=float, default=0.001, help='Initial D Learning Rate')
    parser.add_argument('--l1-lambda', type=int, default=100, help='L1 Loss Lambda')
    parser.add_argument('--const-lambda', type=int, default=15, help='Constant Loss Lambda')
    parser.add_argument('--tv-lambda', type=int, default=0, help='TV Loss Lambda')
    parser.add_argument('--category-lambda', type=int, default=1, help='Category Loss Lambda')
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

"""
@tf.function
def first_call(origin_images, style_target, target_images, target_class, origin_class, embedding,
               generator, discriminator, generator_optimizer, discriminator_optimizer,
               l1_lambda, const_lambda, category_lambda):
    with tf.GradientTape(persistent=True) as tape:
        real_D, real_D_logits, real_category_logits = discriminator(origin_images, training=False)

        fake_B, encoded_real_A = generator(origin_images, embedding, target_class, training=False)

    return None
"""


@tf.function
def train_discriminator_step(origin_images, style_target, target_images, target_class, origin_class, embedding,
                             num_domains, generator, discriminator, discriminator_optimizer, category_lambda):

    with tf.GradientTape(persistent=True) as tape:

        # judge origin images
        disc_source, disc_source_logits, disc_source_category_logits = discriminator(origin_images, training=True)
        # judge target images
        disc_target, disc_target_logits, disc_target_category_logits = discriminator(target_images, training=True)

        # generate fake target images from origin images
        _ , gen_images = generator(origin_images, embedding, target_class, training=False)

        # judge fake target images
        disc_gen, disc_gen_logits, disc_gen_category_logits = discriminator(gen_images, training=True)

        # target images' onehot true labels
        target_labels = tf.reshape(tf.one_hot(indices=target_class, depth=num_domains), shape=[target_class.shape[0], num_domains])

        # origin images' onehot true labels  all index is 0
        # origin_labels = tf.reshape(tf.one_hot(indices=origin_class, depth=num_domains), shape=[origin_class.shape[0], num_domains])

        # category loss
        target_category_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_target_category_logits,
                                                                                      labels=target_labels))

        # binary real/fake loss
        disc_loss = multi_discriminator_loss(disc_source_logits, disc_target_logits, disc_gen_logits)

        #total discriminator loss
        d_loss = category_lambda * target_category_loss + disc_loss


    # calculating gradients
    discriminator_grad = tape.gradient(d_loss, discriminator.trainable_variables)

    # apply gradients
    discriminator_optimizer.apply_gradients(zip(discriminator_grad, discriminator.trainable_variables))

    return category_lambda * target_category_loss, disc_loss


@tf.function
def train_generator_step(origin_images, style_target, target_images, target_class, origin_class, embedding, num_domains,
                         generator, discriminator, generator_optimizer, l1_lambda, const_lambda, category_lambda):

    with tf.GradientTape(persistent=True) as tape:

        # generate fake target images from origin images
        encoded_origin_images, gen_images = generator(origin_images, embedding, target_class, training=True)

        # generate gen images' embedding to approach the origin images' embedding
        encoded_gen_images, _ = generator(gen_images, embedding, origin_class, training=True)

        # fool the discriminator
        disc_gen, disc_gen_logits, disc_gen_category_logits = discriminator(gen_images, training=False)

        # encoding constant loss
        # this loss assume that generated imaged and real image
        # should reside in the same space and close to each other
        const_loss = (tf.reduce_mean(tf.square(encoded_origin_images - encoded_gen_images))) * const_lambda

        # L1 loss between real and generated images
        mae_loss = l1_loss(target_images, gen_images)

        # maximize the chance generator fool the discriminator
        cheat_loss = standard_generator_loss(disc_gen_logits)

        # gen images try to category right
        # target images' onehot true labels
        target_labels = tf.reshape(tf.one_hot(indices=target_class, depth=num_domains),
                                   shape=[target_class.shape[0], num_domains])
        fake_category_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_gen_category_logits,
                                                                                    labels=target_labels))

        # cal total loss
        g_loss = const_lambda * const_loss + l1_lambda * mae_loss + cheat_loss + category_lambda * fake_category_loss

    # calculating gradients
    generator_grad = tape.gradient(g_loss, generator.trainable_variables)

    # apply gradients
    generator_optimizer.apply_gradients(zip(generator_grad, generator.trainable_variables))

    return const_lambda * const_loss, l1_lambda * mae_loss, category_lambda * fake_category_loss



val_l1_loss = tf.keras.metrics.Mean(name='val_l1_loss')
val_ssim = tf.keras.metrics.Mean(name='val_ssim')
def val_step(origin_images, style_target, target_images, target_class, generator, embedding, l1_lambda):
    _, output_images = generator(origin_images, embedding, target_class, training=False)

    target_images_renormalized = (target_images + 1.0) / 2
    output_images_renormalized = (output_images + 1.0) / 2

    ssim = tf.image.ssim(target_images_renormalized[0], output_images_renormalized[0], max_val=1.0)

    #mae_loss_object = tf.keras.losses.MeanAbsoluteError()
    #mae_loss = mae_loss_object(target_images_renormalized, output_images_renormalized)
    mae_loss = l1_loss(target_images_renormalized[0], output_images_renormalized[0]) * l1_lambda

    val_l1_loss(mae_loss)
    val_ssim(ssim)


def test_step(origin_images, style_target, target_images, target_class, target_path, generator, embedding, output_dir):
    _, output_images = generator(origin_images, embedding, target_class, training=False)

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
    l1_lambda = args.l1_lambda
    const_lambda = args.const_lambda
    tv_lambda = args.tv_lambda
    category_lambda = args.category_lambda
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

    generator = build_generator()
    discriminator = build_discriminator(num_domains=num_domains)

    learning_rate_g = [g_learning_rate, g_learning_rate * 0.5, g_learning_rate * 0.25][:]
    boundaries_g = [int(0.3 * epochs * steps_per_epoch), int(0.3 * epochs * steps_per_epoch)]
    learning_rate_schedules_g = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries_g, learning_rate_g)
    learning_rate_d = [d_learning_rate, d_learning_rate * 0.5, d_learning_rate * 0.25][:]
    boundaries_d = [int(0.3 * epochs * steps_per_epoch), int(0.6 * epochs * steps_per_epoch)]
    learning_rate_schedules_d = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries_d, learning_rate_d)


    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedules_g, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedules_d, beta_1=0.5)

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

    #demo_ds = iter(train_dataset)
    #batch_first = next(demo_ds)
    #origin = batch_first['origin']
    #style_inputs = batch_first['style_target']
    #target = batch_first['target']
    #target_class = batch_first['target_class']
    #origin_class = tf.zeros_like(target_class)
    #embedding = tf.random.normal(shape=[num_domains, 1, 1, 128], stddev=0.01)
    #first_call(origin,
    #           style_inputs,
    #           target,
    #           target_class,
    #           origin_class,
    #           embedding,
    #           generator,
    #           discriminator,
    #           generator_optimizer,
    #           discriminator_optimizer,
    #           l1_lambda,
    #           const_lambda,
    #           category_lambda)
    #del demo_ds
    #del batch_first

    for epoch in range(epochs):
        # Reset train metrics
        train_gen_const_loss_sum = 0.0
        train_gen_mae_loss_sum = 0.0
        train_gen_category_loss_sum = 0.0
        train_disc_category_loss_sum = 0.0
        train_disc_loss_sum = 0.0
        num_batches = 0

        # Train Loop
        epoch_timer.timeit()
        for batch in tqdm(train_dataset):
            origin = batch['origin']
            style_inputs = batch['style_target']
            target = batch['target']
            target_class = batch['target_class']
            origin_class = tf.zeros_like(target_class)
            embedding = tf.random.normal(shape=[num_domains, 1, 1, 128], stddev=0.01)

            disc_cate_loss, disc_loss = train_discriminator_step(origin_images=origin,
                                                                 style_target=style_inputs,
                                                                 target_images=target,
                                                                 target_class=target_class,
                                                                 origin_class=origin_class,
                                                                 embedding=embedding,
                                                                 num_domains=num_domains,
                                                                 generator=generator,
                                                                 discriminator=discriminator,
                                                                 discriminator_optimizer=discriminator_optimizer,
                                                                 category_lambda=category_lambda)


            gen_const_loss, gen_mae_loss, gen_cate_loss = train_generator_step(origin_images=origin,
                                                                               style_target=style_inputs,
                                                                               target_images=target,
                                                                               target_class=target_class,
                                                                               origin_class=origin_class,
                                                                               embedding=embedding,
                                                                               num_domains=num_domains,
                                                                               generator=generator,
                                                                               discriminator=discriminator,
                                                                               generator_optimizer=generator_optimizer,
                                                                               l1_lambda=l1_lambda,
                                                                               const_lambda=const_lambda,
                                                                               category_lambda=category_lambda)


            train_disc_category_loss_sum += disc_cate_loss
            train_disc_loss_sum += disc_loss
            train_gen_const_loss_sum += gen_const_loss
            train_gen_mae_loss_sum += gen_mae_loss
            train_gen_category_loss_sum += gen_cate_loss
            num_batches += 1
        used_time_train = epoch_timer.timeit()

        train_log_template = (
            'Epoch {}, Train Discriminator Category Loss: {:.4f}, Train Discriminator Disc Loss: {:.4f}, '
            'Train Generator Constant Loss: {:.4f}, Train Generator L1 Loss: {:.4f}, Train Generator Category Loss: {:.4f},'
            'Train Time: {:.4f} min'
        )
        logger.info(
            train_log_template.format(
                epoch + 1,
                train_disc_category_loss_sum / num_batches,
                train_disc_loss_sum / num_batches,
                train_gen_const_loss_sum / num_batches,
                train_gen_mae_loss_sum / num_batches,
                train_gen_category_loss_sum / num_batches,
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
            embedding = tf.random.normal(shape=[num_domains, 1, 1, 128], stddev=0.01)

            val_step(origin, style_inputs, target, target_class, generator, embedding, l1_lambda)
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
            embedding = tf.random.normal(shape=[num_domains, 1, 1, 128], stddev=0.01)
            test_step(origin, style_inputs, target, target_class, target_path, generator, embedding, cur_output_dir)
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

        if epoch % 2 == 0:
            generator.save_weights(os.path.join(save_path, 'generator_weights.h5'))