import os
from functools import partial

import tensorflow as tf


def _decode_jpeg(image_buffer):
    img=tf.io.decode_jpeg(image_buffer,channels=3)
    img=tf.image.convert_image_dtype(img, dtype=tf.float32)
    
    return img

def _normalize_img(image):

    # normalization 
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    mean =tf.constant(mean, tf.float32)
    std =tf.constant(std, tf.float32)

    image = (image - mean) / std

    return tf.convert_to_tensor(image)

def _preprocess_img(image_decoded, is_training):
    image_decoded.set_shape([None,None,None])
    image =tf.image.resize(image_decoded, [224,224])
    
    if is_training == True:
        IMG_SIZE = tf.shape(image)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.resize(
            image,
            [
                tf.math.floordiv(IMG_SIZE[0] * 9, 8),
                tf.math.floordiv(IMG_SIZE[1] * 9, 8),
            ],
        )
        image = tf.image.random_crop(image, size=[IMG_SIZE[0], IMG_SIZE[1], 3])

    return image

def _parse_fn(example_serialized, is_training):

    feature_map={
        'image/encoded' : tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/class/label'  : tf.io.FixedLenFeature([], dtype=tf.int64,  default_value=-1),
        'image/class/text'   : tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
    }

    parsed = tf.io.parse_single_example(example_serialized, feature_map)
    image = _decode_jpeg(parsed['image/encoded'])

    image = _preprocess_img(image, is_training)
    image = _normalize_img(image)

    label = tf.one_hot(parsed['image/class/label'] - 1, 1000, dtype=tf.float32)
    return (image, label)


def get_dataset(tfrecords_dir, subset, batch_size, num_data_workers):
    """Read TFRecords files and turn them into a TFRecordDataset."""
    files = tf.io.matching_files(os.path.join(tfrecords_dir, '%s-*' % subset))
    shards = tf.data.Dataset.from_tensor_slices(files)
    shards = shards.shuffle(tf.cast(tf.shape(files)[0], tf.int64))
    shards = shards.repeat()
    dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length=4)
    dataset = dataset.shuffle(buffer_size=8192)

    parser = partial(_parse_fn, is_training=True if subset == 'train' else False)
    dataset = dataset.map(parser) #apply preprocessing for training images
    dataset = dataset.batch(batch_size) #set batch size
    dataset = dataset.prefetch(batch_size)

    return dataset