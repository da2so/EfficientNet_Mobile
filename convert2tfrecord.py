
#!/usr/bin/python
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Modification by SH Kang <shkang@nota.ai>:
# * removed bbox annotations in the TFRecords
# * reconstruct code for tf 2.4.0


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import random
import sys
import threading

import numpy as np
import six
import tensorflow as tf

from data.imagecoder import ImageCoder



def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if six.PY3 and isinstance(value, six.text_type):
        value = six.binary_type(value, encoding='utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, synset, human,
                        height, width):
    """
    Build an Example proto for an example.
    """
    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/synset': _bytes_feature(synset),
        'image/class/text': _bytes_feature(human),
        'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(os.path.basename(filename)),
        'image/encoded': _bytes_feature(image_buffer)}))
    return example


def _process_image(filename, coder):
    """
    Process a single image file.
    """
    # Read the image file.
    with tf.compat.v1.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()
    """
    # Clean the dirty data.
    if coder._is_png(filename):
        # 1 image is a PNG.
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)
    elif coder._is_cmyk(filename):
        # 22 JPEG images are in CMYK colorspace.
        print('Converting CMYK to RGB for %s' % filename)
        image_data = coder.cmyk_to_rgb(image_data)
    """
    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               synsets, labels, humans, num_shards, output_dir):
    """
    Processes and saves list of images as TFRecord in 1 thread.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads

    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                                ranges[thread_index][1],
                                num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(output_dir, output_filename)
        writer = tf.io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            synset = synsets[i]
            human = humans[i]

            image_buffer, height, width = _process_image(filename, coder)

            example = _convert_to_example(filename, image_buffer, label,
                                            synset, human, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                    (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
            (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
            (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files(name, filenames, synsets, labels, humans,
                         num_shards, num_threads, output_dir):
    """
    Process and save list of images as TFRecord of Example protos.
    """
    assert len(filenames) == len(synsets)
    assert len(filenames) == len(labels)
    assert len(filenames) == len(humans)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames,
                synsets, labels, humans, num_shards, output_dir)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
            (datetime.now(), len(filenames)))
    sys.stdout.flush()


def _find_image_files(data_dir, labels_file):
    """
    Build a list of all images files and labels in the data set.
    """
    print('Determining list of input files and labels from %s.' % data_dir)
    challenge_synsets = [l.strip() for l in tf.compat.v1.gfile.FastGFile(labels_file, 'r').readlines()]

    labels = []
    filenames = []
    synsets = []

    # Leave label index 0 empty as a background class.
    label_index = 1

    # Construct the list of JPEG files and labels.
    for synset in challenge_synsets:
        jpeg_file_path = '%s/%s/*.JPEG' % (data_dir, synset)
        matching_files = tf.io.gfile.glob(jpeg_file_path)

        labels.extend([label_index] * len(matching_files))
        synsets.extend([synset] * len(matching_files))
        filenames.extend(matching_files)

        if not label_index % 100:
            print('Finished finding files in %d of %d classes.' % ( label_index, len(challenge_synsets)))
        label_index += 1

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    synsets = [synsets[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Found %d JPEG files across %d labels inside %s.' %
            (len(filenames), len(challenge_synsets), data_dir))
    return filenames, synsets, labels


def _find_human_readable_labels(synsets, synset_to_human):
    """
    Build a list of human-readable labels.
    """
    humans = []
    for s in synsets:
        assert s in synset_to_human, ('Failed to find: %s' % s)
        humans.append(synset_to_human[s])
    return humans


def _process_dataset(name, directory, num_shards, num_threads,
                    labels_file, synset_to_human, output_dir):
    """
    Process a complete data set and save it as a TFRecord.
    """
    filenames, synsets, labels = _find_image_files(directory, labels_file)
    humans = _find_human_readable_labels(synsets, synset_to_human)
    _process_image_files(name, filenames, synsets, labels,
                        humans, num_shards, num_threads, output_dir)


def _build_synset_lookup(imagenet_metadata_file):
    """
    Build lookup for synset to human-readable label.
    """
    lines = tf.compat.v1.gfile.FastGFile(imagenet_metadata_file, 'r').readlines()
    synset_to_human = {}
    for l in lines:
        if l:
            parts = l.strip().split('\t')
            assert len(parts) == 2
            synset = parts[0]
            human = parts[1]
            synset_to_human[synset] = human
    return synset_to_human




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converting data to tfrecord')

    parser.add_argument('--train_dir', type = str, default = './dataset/raw_data/train/', help = 'Randomly transform image and annotations')
    parser.add_argument('--val_dir', type = str, default = './dataset/raw_data/val/', help = 'Batch size')
    parser.add_argument('--output_dir', type = str, default = './dataset/tfrecord/', help ='ImageNet dataset path' )
    
    parser.add_argument('--train_shards', type=int, default = 1024, help = 'Number of shards in training TFRecord files')
    parser.add_argument('--val_shards', type = int, default = 128, help = 'Number of shards in validation TFRecord files')
    parser.add_argument('--num_threads', type = int, default = 4, help = 'Number of threads to preprocess the images')

    # Arguments related to generator config
    parser.add_argument('--labels_file', type = str, default = './data/imagenet_lsvrc_2015_synsets.txt', help = 'Labels file')
    parser.add_argument('--imagenet_metadata_file', type = str, default = './data/imagenet_metadata.txt', help = 'ImageNet metadata file')

    args = parser.parse_args()

    # Build a map from synset to human-readable label.
    synset_to_human = _build_synset_lookup(args.imagenet_metadata_file)

    # Run it!
    _process_dataset('validation',
                    args.val_dir, 
                    args.val_shards,
                    args.num_threads, 
                    args.labels_file, 
                    synset_to_human,
                    args.output_dir)

    _process_dataset('train', 
                    args.train_dir,
                    args.train_shards,
                    args.num_threads, 
                    args.labels_file, 
                    synset_to_human,
                    args.output_dir)