import os
import sys
import tensorflow as tf
import json
import random
import time
from utils import *

# 将包含多个单一路径名称的list合并为一个连续路径
def gather_path(path_split):
    head = ''
    for path in path_split:
        head = os.path.join(head, path)
    return head

logger = get_logger()

def build_train_json(train_dir, train_classes, output_dir, save_tag):
    all_target_path_list = []
    for font_class in train_classes:
        font_class_dir = (train_dir + '/{}/*.png').format(str(font_class))
        target_path_list = tf.io.gfile.glob(font_class_dir)
        all_target_path_list.extend(target_path_list)

    random.shuffle(all_target_path_list)

    json_list = []
    for target_path in all_target_path_list:
        temp = dict()
        path_split = target_path.split('/')

        origin_path = path_split[:2] + ['0'] + path_split[3:]
        origin_path = gather_path(origin_path)

        style_index = str(random.randint(0, 999)).zfill(4)
        style_index = 'hz_{}.png'.format(style_index)
        path_split[3] = style_index
        style_path = gather_path(path_split)
        temp['target_path'] = target_path
        temp['origin_path'] = origin_path
        temp['style_path'] = style_path
        json_list.append(temp)

    if not tf.io.gfile.exists(output_dir):
        tf.io.gfile.makedirs(output_dir)

    save_tag = save_tag + '-train'
    file_name = os.path.join(output_dir, save_tag)

    with tf.io.gfile.GFile('{}.json'.format(file_name), 'w') as f:
        for item in json_list:
            item = json.dumps(item)
            f.write(item + '\n')

    logger.info('Create Train Dataset Json File {}'.format(file_name))

    return file_name + '.json'



def build_val_json(val_dir, val_classes, output_dir, save_tag):
    all_target_path_list = []
    for font_class in val_classes:
        font_class_dir = (val_dir + '/{}/*.png').format(str(font_class))
        target_path_list = tf.io.gfile.glob(font_class_dir)
        all_target_path_list.extend(target_path_list)

    # random.shuffle(all_target_path_list)

    json_list = []
    for target_path in all_target_path_list:
        temp = dict()
        path_split = target_path.split('/')

        origin_path = path_split[:2] + ['0'] + path_split[3:]
        origin_path = gather_path(origin_path)

        style_index = str(random.randint(1000, 1199)).zfill(4)
        style_index = 'hz_{}.png'.format(style_index)
        path_split[3] = style_index
        style_path = gather_path(path_split)
        temp['target_path'] = target_path
        temp['origin_path'] = origin_path
        temp['style_path'] = style_path
        json_list.append(temp)

    if not tf.io.gfile.exists(output_dir):
        tf.io.gfile.makedirs(output_dir)

    save_tag = save_tag + '-val'
    file_name = os.path.join(output_dir, save_tag)

    with tf.io.gfile.GFile('{}.json'.format(file_name), 'w') as f:
        for item in json_list:
            item = json.dumps(item)
            f.write(item + '\n')

    logger.info('Create Val Dataset Json File {}'.format(file_name))

    return file_name + '.json'


def build_test_json(test_dir, test_classes, output_dir, save_tag):
    all_target_path_list = []
    for font_class in test_classes:
        font_class_dir = (test_dir + '/{}/*.png').format(str(font_class))
        target_path_list = tf.io.gfile.glob(font_class_dir)
        all_target_path_list.extend(target_path_list)

    # random.shuffle(all_target_path_list)

    json_list = []
    for target_path in all_target_path_list:
        temp = dict()
        path_split = target_path.split('/')

        origin_path = path_split[:2] + ['0'] + path_split[3:]
        origin_path = gather_path(origin_path)

        style_index = str(random.randint(1200, 1249)).zfill(4)
        style_index = 'hz_{}.png'.format(style_index)
        path_split[3] = style_index
        style_path = gather_path(path_split)
        temp['target_path'] = target_path
        temp['origin_path'] = origin_path
        temp['style_path'] = style_path
        json_list.append(temp)

    if not tf.io.gfile.exists(output_dir):
        tf.io.gfile.makedirs(output_dir)

    save_tag = save_tag + '-test'
    file_name = os.path.join(output_dir, save_tag)

    with tf.io.gfile.GFile('{}.json'.format(file_name), 'w') as f:
        for item in json_list:
            item = json.dumps(item)
            f.write(item + '\n')

    logger.info('Create Test Dataset Json File {}'.format(file_name))

    return file_name + '.json'


def load_json(file_name):
    json_list = []
    with tf.io.gfile.GFile(file_name, 'r') as f:
        for item in f.readlines():
            item = item.strip()
            item = json.loads(item)
            json_list.append(item)

    logger.info('Load Dataset From Json File {}'.format(file_name))

    return json_list