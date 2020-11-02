import os
import random
import tensorflow as tf

from utils import *


# 将包含多个单一路径名称的list合并为一个连续路径
def gather_path(path_split):
    head = ''
    for path in path_split:
        head = os.path.join(head, path)
    return head


def build_path(target_path, font_tag):
    """输入target路径，返回原始宋体图片路径和随机同风格图片路径

    Args:
        target_path: 目标图片(ground truth)路径
        font_tag: 数据用途, ['train', 'val', 'test']

    Returns:
        origin_path: 宋体图片路径
        style_path: 随机的目标图片同风格图片
    """
    path_split = target_path.split('/')

    origin_path = path_split[:2] + ['0'] + path_split[3:]
    origin_path = gather_path(origin_path)

    if font_tag == 'train':
        style_index = str(random.randint(0, 999)).zfill(4)
    elif font_tag == 'val':
        style_index = str(random.randint(1000, 1199)).zfill(4)
    elif font_tag == 'test':
        style_index = str(random.randint(1200, 1249)).zfill(4)
    style_index = 'hz_{}.png'.format(style_index)
    path_split[3] = style_index
    style_path = gather_path(path_split)

    return origin_path, style_path


def read_decode_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, 3) # [0, 1]
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img * 2.0 - 1.0 # normalize [-1., 1.]
    return img


def load_image(target_path, origin_path, style_path, training):
    target_image = read_decode_image(target_path)
    origin_image = read_decode_image(origin_path)
    style_image = read_decode_image(style_path)

    # img range is [-1., 1.]
    return [target_image, origin_image, style_image]


# 读取图片
def process_image_data(training):

    def process_function(raw_data):
        [target, origin, style] = tf.py_function(
            load_image,
            [raw_data['target'],
             raw_data['origin'],
             raw_data['style_target'],
             training],
            [tf.float32, tf.float32, tf.float32]
        )

        outdata = raw_data
        del raw_data['target']
        outdata['target'] = target
        del raw_data['origin']
        outdata['origin'] = origin
        del raw_data['style_target']
        outdata['style_target'] = style

        return outdata

    return process_function


def get_image_dataset(data_dir, font_classes, font_tag):
    """get image tensorflow Dataset

    Args:
        data_dir: train/val/test data dir
        font_classes: number of target font classes
        font_tag: usage of dataset, choose from ['train', 'val', 'test']

    Returns:
        Dataset
    """

    records = {
        'origin': [],
        'style_target': [],
        'target': [],
        'target_path': [],
    }

    logger = get_logger()

    if font_tag not in ['train', 'val', 'test']:
        font_tag = data_dir.split('/')[1]

    logger.info('Create {} Dataset'.format(font_tag))

    training = True if font_tag == 'train' else False # image augment tag

    for font_class in range(1, font_classes+1):
        font_class_dir = (data_dir + '/{}/*.png').format(str(font_class))
        target_list = tf.io.gfile.glob(font_class_dir)
        for target_path in target_list:
            records['target_path'].append(target_path)
            records['target'].append(target_path)
            origin_path, style_path = build_path(target_path, font_tag)
            records['origin'].append(origin_path)
            records['style_target'].append(style_path)

    dataset = tf.data.Dataset.from_tensor_slices(records)
    dataset = dataset.map(
        process_image_data(training),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset


