import time
import logging
import os
import sys
import tensorflow as tf

_logger_name = 'logger'


def initial_logger(out_path):
    logger = logging.getLogger(_logger_name)
    logger.setLevel('INFO')

    handler1 = logging.StreamHandler()
    log_f = tf.io.gfile.GFile(out_path, 'w')
    handler2 = logging.StreamHandler(log_f)

    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        '%Y-%m-%d %H:%M:%S')

    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)

    logger.addHandler(handler1)
    logger.addHandler(handler2)


def get_logger():
    return logging.getLogger(_logger_name)


def get_time_str():
    """get local time

    Return:
        time str looks like: '2020-10-29_18-51-13'
    """
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    return time_str


def new_folder(dir_path):
    """make a new empty folder

    Args:
        dir_path: folder path
    """
    if tf.io.gfile.isdir(dir_path):
        tf.io.gfile.rmtree(dir_path) # Deletes everything under path recursively.
    tf.io.gfile.mkdir(dir_path)


def init_out_dir(out_dir, tag):
    """init current experiment save folder

    Args:
        out_dir: existed dir to save results
        tag: experiment tag
    """
    time_str = get_time_str()

    if not tag:
        tag = 'notag'
    exp_name = '{}_{}'.format(tag, time_str)
    out_dir = os.path.join(out_dir, exp_name)

    new_folder(out_dir)

    return out_dir


def remove_file(file_path):
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass


class Timer(object):
    """class to do timekeeping

    """

    def __init__(self):
        self.last_time = time.time()

    def timeit(self):
        old_time = self.last_time
        self.last_time = time.time()
        return self.last_time - old_time



def plot_image(output_dir, output_image, font_class, font_index):
    output_dir = os.path.join(output_dir, font_class)
    if not tf.io.gfile.isdir(output_dir):
        tf.io.gfile.mkdir(output_dir)
    output_dir = os.path.join(output_dir, font_index)

    output_image = tf.io.encode_jpeg(output_image)
    tf.io.write_file(output_dir, output_image)




