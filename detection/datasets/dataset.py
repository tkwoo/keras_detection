
import numpy as np
from glob import glob

from numpy.core.defchararray import center
import tensorflow as tf
import os
from tqdm import tqdm
import itertools
import xml.etree.ElementTree as ET
from .transforms import _resize

AUTOTUNE = tf.data.experimental.AUTOTUNE\


class Dataprocessor:
    def __init__(self, args):
        self.args = args
        self.classes = self.args.MODEL.CLASSES
        self.total_train = self.make_tfrecord(args.TRAIN_DIR)
        self.total_val = self.make_tfrecord(args.VAL_DIR)
        self.train_tfrecords = self.load_tfrecord(args.TRAIN_DIR)
        self.train_tfrecords = self.create_dataset(self.train_tfrecords)
        self.val_tfrecords = self.load_tfrecord(args.VAL_DIR)
        self.val_tfrecords = self.create_dataset(self.val_tfrecords, True)

    @property
    def train_length(self):
        return self.total_train // self.args.BATCH_SIZE

    @property
    def val_length(self):
        return self.total_val // self.args.BATCH_SIZE

    def create_dataset(self, tfrecords, is_val=False):
        # augmentation = DataAugmenter(self.args, is_val)
        if self.args.MODEL.AUTOML:
            dirs = self.args.TRAIN_DIR if not is_val else self.args.VAL_DIR
            train_gen = tf.keras.preprocessing.image.ImageDataGenerator()
            data_gens = [train_gen.flow_from_directory(
                directory=d,
                batch_size=self.args.BATCH_SIZE,
                shuffle=True,
                target_size=(self.args.DATA.SIZE[1], self.args.DATA.SIZE[0]),
            ) for d in dirs]
            data_gens = itertools.chain(*data_gens)

            def callable_iterator(generator):
                for img_batch, targets_batch in generator:
                    yield img_batch, targets_batch

            return tf.data.Dataset.from_generator(lambda: callable_iterator(data_gens), output_types=(tf.float32, tf.float32))

        else:
            if not is_val:
                return tfrecords.repeat()\
                                .shuffle(self.args.DATA.SHUFFLE_SIZE)\
                                .batch(self.args.BATCH_SIZE)\
                                .prefetch(buffer_size=AUTOTUNE)
            else:
                return tfrecords.batch(self.args.BATCH_SIZE)\
                                .prefetch(buffer_size=AUTOTUNE)

    def bounding_box(self, xml_file):
        # img_path_split = image.split('.')[:-1]
        # print(img_path_split)
        # bpath = '{}/{}.xml'.format(root_annots,'.'.join(map(str,img_path_split)))
        tree = ET.parse(xml_file)
        root = tree.getroot()
        objects = root.findall('object')
        xmins, ymins, xmaxs, ymaxs = [], [], [], []
        names = []
        for o in objects:
            bndbox = o.find('bndbox')
            name = o.find('name').text
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            xmins.append(xmin)
            xmaxs.append(xmax)
            ymins.append(ymin)
            ymaxs.append(ymax)
            # boxes.append([xmin, ymin, xmax, ymax])
            names.append(name)
        return xmins, ymins, xmaxs, ymaxs, names

    # def read_classes(self, paths):
    #     classes = self.args.CLASSES

    #     return dirs

    def make_tfrecord(self, paths):
        lengths = []
        for path in paths:
            record_file = os.path.join(path, f'data_{len(self.classes)}.tfrecords')
            if os.path.exists(record_file):
                with open(record_file + f'_{len(self.classes)}.length', 'r') as l:
                    lengths.append(int(l.readline()))
                continue
            
            files = [f for f in glob(os.path.join(path, '*/*')) if f.endswith('jpg') or f.endswith('.png')]
            # print(os.path.join(path, '*/*.{jpg,png}'))
            # print(files)
            np.random.shuffle(files)
            print('making tfrecords..')
            # options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
            length = 0
            with tf.io.TFRecordWriter(record_file) as writer:
                for f in tqdm(files):
                    basedir = os.path.basename(os.path.dirname(f))
                    if not os.path.exists(f.replace(basedir, basedir + '_annotation').replace('.jpg', '.xml')):
                        continue
                    xmins, ymins, xmaxs, ymaxs, names = self.bounding_box(f.replace(basedir, basedir + '_annotation').replace('.jpg', '.xml'))
                    image_string = open(f, 'rb').read()
                    # label = os.path.basename(os.path.dirname(f))
                    tf_example = self.__make_feature_from(image_string, xmins, ymins, xmaxs, ymaxs, names)
                    if tf_example == None:
                        continue
                    length += 1
                    writer.write(tf_example.SerializeToString())
            lengths.append(length)
            with open(record_file + f'_{len(self.classes)}.length', 'w') as l:
                l.write(str(length))

        return sum(lengths)

    def __make_feature_from(self, image_string, xmins, ymins, xmaxs, ymaxs, names):
        labels = []
        xmins_filtered, ymins_filtered, xmaxs_filtered, ymaxs_filtered = [], [], [], []
        for n, xmin, ymin, xmax, ymax in zip(names, xmins, ymins, xmaxs, ymaxs):
            if n in self.classes:
                labels.append(self.classes.index(n) + 1)
                xmins_filtered.append(xmin)
                ymins_filtered.append(ymin)
                xmaxs_filtered.append(xmax)
                ymaxs_filtered.append(ymax)
        if len(labels) == 0:
            return None
        # names = [self.classes.index(label) for label in names]
        image_shape = tf.image.decode_jpeg(image_string).shape
        if isinstance(image_string, type(tf.constant(0))):
            image_string = image_string.numpy()
        h, w, _ = image_shape
        feature = {
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
            'xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins_filtered)),
            'ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins_filtered)),
            'xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs_filtered)),
            'ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs_filtered)),
            'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=labels)),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def load_tfrecord(self, paths):
        records = []
        image_feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'xmin': tf.io.VarLenFeature(tf.float32),
            'ymin': tf.io.VarLenFeature(tf.float32),
            'xmax': tf.io.VarLenFeature(tf.float32),
            'ymax': tf.io.VarLenFeature(tf.float32),
            'labels': tf.io.VarLenFeature(tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string),
        }
        record_paths = [os.path.join(path, f'data_{len(self.classes)}.tfrecords') for path in paths]
        dataset = tf.data.TFRecordDataset(record_paths)
        mean = self.args.DATA.MEAN
        std = self.args.DATA.STD
        final_stride = self.args.FINAL_STRIDE
        image_size = [self.args.DATA.SIZE[1], self.args.DATA.SIZE[0]]
        classes = self.args.MODEL.NUM_CLASSES

        def loss_matching(label):
            # origin_shape = [256, 128]
            # bs, h, w, 1
            # label = bs, None(the number of boxes per image), 5
            logits_shape = [38, 22]#[78, 46]#[int(image_size[0]/final_stride), int(image_size[1]/final_stride)]
            

            x, y = tf.meshgrid(tf.range(logits_shape[1]), tf.range(logits_shape[0]))
            x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)
            x += 0.5
            y += 0.5
            # h, w
            x /= logits_shape[1]
            y /= logits_shape[0]
            # h, w, 4
            grid_cell = tf.stack([y, x], axis=-1)
            grid_cell = tf.reshape(grid_cell, [-1, 2])
            # grid_cell = tf.stack([y, x, y, x], axis=-1)

            # y, x, y, x
            # for center target
            # centers_label = tf.stack([(label[:, 0] + label[:, 2]) / 2, (label[:, 1] + label[:, 3]) / 2], axis=-1)
            # centers_label = tf.stack([
            #     centers_label[:, 0] * logits_shape[0],
            #     centers_label[:, 1] * logits_shape[1]
            # ], axis=-1)

            # centers_label = tf.math.floor(centers_label)
            # centers_label = tf.cast(centers_label, tf.int64)
            
            # for all target
            # h * w, boxes, 4
            box_centers = tf.stack([(label[:, 0] + label[:, 2]) / 2, (label[:, 1] + label[:, 3]) / 2], axis=-1)
            centers_label = tf.stack([
                grid_cell[:, 0, None] - label[:, 0][None],
                grid_cell[:, 1, None] - label[:, 1][None],
                label[:, 2][None] - grid_cell[:, 0, None],
                label[:, 3][None] - grid_cell[:, 1, None]
            ], axis=2)

            centers_label = tf.math.reduce_min(centers_label, axis=-1, keepdims=False)
            
            # h * w, boxes
            is_not_in_boxes = centers_label <= 0
            # tf.print(tf.reduce_sum(tf.cast(is_not_in_boxes, tf.int64)))
            # h * w, boxes
            centers_distance = tf.math.reduce_euclidean_norm(
                grid_cell[:, None, :] - box_centers[None], axis=2, keepdims=False
            )

            # filtering not in box
            # h * w, boxes
            centers_distance = tf.where(is_not_in_boxes, 99999., centers_distance)
            # h * w
            gt_index = tf.math.argmin(centers_distance, axis=1)
            # tf.print(tf.math.reduce_max(gt_index))
            # tf.print(centers_distance.shape)
            # tf.print(label.shape)
            # tf.print(label[:, 4][gt_index[10]])
            logits_label = tf.gather(label[:, 4], gt_index)
            # tf.print(logits_label)
            min_centers_distance = tf.math.reduce_min(centers_distance, axis=1, keepdims=False)
            logits_label = tf.where(min_centers_distance == 99999, -1., logits_label)
            logits_label = tf.cast(logits_label, tf.int64)
            logits_label = tf.one_hot(logits_label, classes-1, on_value=1.0, off_value=0.0, axis=-1)
            # label_class = tf.cast(label[:, 4], tf.int64) - 1
            # logits_label = tf.ones([logits_shape[0], logits_shape[1]], dtype=tf.int64) * -1
            # logits_label = tf.tensor_scatter_nd_update(logits_label, centers_label, label_class)
            # logits_label = tf.one_hot(logits_label, classes-1, on_value=1.0, off_value=0.0, axis=-1)
            
            # for sparse cross entropy
            # label_class = tf.cast(label[:, 4], tf.float32)
            # logits_label = tf.zeros([logits_shape[0], logits_shape[1]], dtype=tf.float32)
            # logits_label = tf.tensor_scatter_nd_update(logits_label, centers_label, label_class)
            # logits_label = tf.expand_dims(logits_label, axis=-1)
            # logits_label = tf.one_hot(logits_label, classes, on_value=1.0, off_value=0.0, axis=-1)
            
            
            # centers_label_index = tf.concat([centers_label, label_class], axis=-1)
            # logits_label = tf.zeros([logits_shape[0], logits_shape[1], classes], dtype=tf.float32)
            # logits_label = tf.tensor_scatter_nd_add(logits_label, centers_label_index, tf.ones([tf.shape(centers_label_index)[0]], tf.float32))
            # tf.print(logits_label)
            # box_label = tf.zeros([logits_shape[0] * logits_shape[1], 4], dtype=tf.float32)
            # box_label = tf.tensor_scatter_nd_update(box_label, centers_label, tf.constant(1))
            # selected_position = tf.gather_nd(grid_cell, centers_label)
            # encoded_box = tf.stack([
            #     selected_position[:, 0] - label[:, 0],
            #     selected_position[:, 1] - label[:, 1],
            #     label[:, 2] - selected_position[:, 2],
            #     label[:, 3] - selected_position[:, 3]
            # ], axis=-1)
            
            # box_label = tf.tensor_scatter_nd_update(box_label, gt_index, label[:, :4])
            # box_label = tf.tensor_scatter_nd_update(box_label, centers)
            # box_targets = tf.concat(box_targets)
            box_label = tf.gather(label[:, :4], gt_index)
            box_label = tf.reshape(box_label, [logits_shape[0], logits_shape[1], 4])
            logits_label = tf.reshape(logits_label, [logits_shape[0], logits_shape[1], -1])
            return tf.concat([box_label, logits_label], axis=-1)

        def _parse_image_function(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            example = tf.io.parse_single_example(example_proto, image_feature_description)
            # image = tf.io.decode_raw(example['image'], tf.uint8)
            image = tf.image.decode_jpeg(example['image'], channels=3, dct_method='INTEGER_ACCURATE')
            image = tf.reshape(image, [example['height'], example['width'], 3])
            labels = tf.sparse.to_dense(example['labels'])
            labels = tf.cast(labels, tf.float32)
            labels = tf.stack(
                        [tf.sparse.to_dense(example['ymin']),
                        tf.sparse.to_dense(example['xmin']),
                        tf.sparse.to_dense(example['ymax']),
                        tf.sparse.to_dense(example['xmax']),labels], axis=-1)
            image = tf.cast(image, tf.float32)
            image = image / 255
            image = image - [[mean]]
            image = image / [[std]]
            image, labels = _resize(image, labels, image_size)
            label = loss_matching(labels)
            return image, label

        return dataset.map(_parse_image_function, num_parallel_calls=AUTOTUNE)


def build_data(cfg):
    return Dataprocessor(cfg)