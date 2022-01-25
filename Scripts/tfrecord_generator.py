from collections import namedtuple
from object_detection.utils import dataset_util
from PIL import Image
import argparse
import io
import os
import pandas as pd
import tensorflow as tf
import random

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--labelmap_file', help='The path of labelmap file.')
parser.add_argument('-o', '--output_path', help='The output path of record.')
parser.add_argument('-i', '--image_dir', help='The images directory.')
parser.add_argument('-csv', '--csv_input', help='The csv input path.')


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def init_names(labelmap):
    items = labelmap.split('item')[1:]
    items_dict = {}
    for item in items:
        name = str(item.split('name')[1].split('"')[1]).strip()
        name_id = int(item.split('name')[1].split('id: ')[1].split('}')[0])
        items_dict[name] = name_id
    return items_dict


class TFRecord:
    def __init__(self, labelmap_file):
        f = open(labelmap_file, "r")
        labelmap = f.read()
        self.class_names = init_names(labelmap)

    def class_text_to_int(self, row_label):
        if self.class_names[row_label] is not None:
            return self.class_names[row_label]

    def create_tf(self, group, path):
        with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size

        filename = group.filename.encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for index, row in group.object.iterrows():
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(self.class_text_to_int(row['class']))

        tf_sample = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_sample

    def generate(self, output_path, image_dir, csv_input):
        writer = tf.io.TFRecordWriter(output_path)
        path = os.path.join(image_dir)
        data = pd.read_csv(csv_input)
        grouped = (split(data, 'filename'))
        random.shuffle(grouped)

        for group in grouped:
            tf_sample = self.create_tf(group, path)
            writer.write(tf_sample.SerializeToString())

        print('Generated tf record.')


def main():
    args = parser.parse_args()
    tf_record = TFRecord(args.labelmap_file)
    tf_record.generate(args.output_path, args.image_dir, args.csv_input)


main()
