from __future__ import absolute_import, division, print_function, unicode_literals

from numpy.core.defchararray import encode

import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201

from focal_loss import sparse_categorical_focal_loss


keras_factory = {
    'mobilenet': MobileNetV2,
    'resnet50': ResNet50V2,
    'resnet101': ResNet101V2,
    "resnet152": ResNet152V2,
    'inceptionv3': InceptionV3,
    'inception-resnet': InceptionResNetV2,
    'densenet121': DenseNet121,
    'densenet169': DenseNet169,
    'densenet201': DenseNet201
    # 'efficientnetb0': "https://tfhub.dev/google/efficientnet/b0/feature_vector/1",
    # 'efficientnetb1': "https://tfhub.dev/google/efficientnet/b1/feature_vector/1",
    # 'efficientnetb2': "https://tfhub.dev/google/efficientnet/b2/feature_vector/1"
}


BN_MOMENTUM = 0.1


def unpack_x_y_sample_weight(data):
    """Unpacks user-provided data tuple."""
    if not isinstance(data, tuple):
        return (data, None, None)
    elif len(data) == 1:
        return (data[0], None, None)
    elif len(data) == 2:
        return (data[0], data[1], None)
    elif len(data) == 3:
        return (data[0], data[1], data[2])


class CustomModel(tf.keras.Model):
    def __init__(self, cfg, **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        model_name = cfg.MODEL.NAME
        num_classes = cfg.MODEL.NUM_CLASSES
        if cfg.DATA.RANDOM_CROP:
            size = [cfg.DATA.RANDOM_CROP_SIZE[1], cfg.DATA.RANDOM_CROP_SIZE[0]]
        else:
            size = [cfg.DATA.SIZE[1], cfg.DATA.SIZE[0]]
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        assert model_name in keras_factory
        self.cfg = cfg
        self.deconv_with_bias = cfg.DECONV_WITH_BIAS
        self.model = keras_factory[model_name](
            weights='imagenet', include_top=False, input_tensor=Input(shape= size + [3]))
        self.deconv_layers = self._make_deconv_layer(
            cfg.NUM_DECONV_LAYERS,
            cfg.NUM_DECONV_FILTERS,
            cfg.NUM_DECONV_KERNELS,
        )
        self.box_pred = keras.Sequential([
            layers.Conv2D(
                filters=4,
                kernel_size=3,
                padding='same'#, activation='sigmoid'
            ),
            
        ])
        self.scaler = tf.Variable(1.)
        self.final_relu = layers.Activation('relu')
        
        self.final_logit = keras.Sequential([
            layers.Conv2D(
                filters=cfg.MODEL.NUM_CLASSES-1,
                kernel_size=cfg.FINAL_CONV_KERNEL,
                padding='same', activation='sigmoid'
            )
        ])
        
        self.add_weight_decay(weight_decay)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers_ = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            # print(self.deconv_with_bias)
            # print(output_padding)
            layers_.append(
                layers.Conv2DTranspose(
                    filters=planes,
                    kernel_size=kernel,
                    strides=2,
                    padding='valid',
                    output_padding=output_padding,
                    use_bias=self.deconv_with_bias))
            layers_.append(layers.BatchNormalization(momentum=BN_MOMENTUM))
            layers_.append(layers.Activation('relu'))
            # self.inplanes = planes
        # layers_.append(
        #     keras.Sequential([layers.Conv2D(
        #         filters=256,
        #         kernel_size=3,
        #         padding='same'
        #     ), layers.BatchNormalization(momentum=BN_MOMENTUM),
        #     layers.Activation('relu')]))

        return keras.Sequential(layers_)

    def export(self):
        output = self.model.output
        output = self.fc(output)
        if self.cfg.MODEL.TEMPERATURE_SCALING != 1:
            output = preprocessing.Rescaling(1./self.cfg.MODEL.TEMPERATURE_SCALING)(output)
        output = self.softmax(output)
        return Model(inputs=self.model.input, outputs=output)

    def add_weight_decay(self, weight_decay):
        for layer in self.model.layers:
            layer.trainable = True
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer= l2(weight_decay)
        
        for layer in self.deconv_layers.layers:
            layer.trainable = True
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer= l2(weight_decay)

        for layer in self.box_pred.layers:
            layer.trainable = True
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer= l2(weight_decay)

    def __call__(self, x, training=False):
        self.model.trainable = training
        self.deconv_layers.trainable = training
        self.box_pred.trainable = training
        self.final_logit.trainable = training
        output = self.model(x)
        output = self.deconv_layers(output)
        box_pred = self.box_pred(output)
        box_pred = box_pred * self.scaler
        box_pred = self.final_relu(box_pred)
        logits = self.final_logit(output)
        return logits, box_pred

    def test_step(self, data):
        data, label, weights = unpack_x_y_sample_weight(data)

        logits, box_pred = self(data, False)
        logits_loss, box_loss = self.loss_matching(logits, box_pred, label)
        total_loss = logits_loss + box_loss
        # loss = tf.math.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(label, pred))
        # self.compiled_metrics.update_state(label, pred, weights)
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = total_loss
        
        return results

    def train_step(self, data):
        data, label, weights = unpack_x_y_sample_weight(data)
            
        with tf.GradientTape() as tape:
            logits, box_pred = self(data, True)

            logits_loss, box_loss = self.loss_matching(logits, box_pred, label)
            total_loss = logits_loss + box_loss
            # print(self.trainable_weights)
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            return {
                "loss": total_loss,
                "logits_loss": logits_loss,
                "box_loss": box_loss
            }

    def loss_matching(self, logits, box_pred, label):
        origin_shape = [256, 128]
        # bs, h, w, 1
        # label = bs, None(the number of boxes per image), 5
        logits_shape = tf.cast(tf.shape(logits), tf.float32)
        # logits_label = tf.zeros([logits_shape[0], logits_shape[1], logits_shape[2], 3], dtype=tf.float32)

        x, y = tf.meshgrid(tf.range(logits_shape[2]), tf.range(logits_shape[1]))
        x += 0.5
        y += 0.5
        # h, w
        x /= logits_shape[2]
        y /= logits_shape[1]
        # h, w, 4 (yxyx)
        grid_cell = tf.stack([y, x, y, x], axis=-1)
        grid_cell = tf.expand_dims(grid_cell, axis=0)

        box_label, logits_label = label[..., :4], label[..., 4:]
        # logits_loss = sparse_categorical_focal_loss(logits_label, logits, 2.0, from_logits=True)
        # logits_loss = tf.nn.softmax_cross_entropy_with_logits(tf.cast(logits_label, tf.int64), logits)
        logits_loss = tfa.losses.sigmoid_focal_crossentropy(logits_label, logits, from_logits=False)
        # tf.print(tf.reduce_sum(logits_label))
        # tf.print(tf.size(logits_label))
        # tf.print(grid_cell[..., 0].shape)
        logits_loss = tf.reduce_sum(logits_loss)
        # encoded_box = tf.stack([
        #         grid_cell[..., 0] - box_pred[..., 0],
        #         grid_cell[..., 1] - box_pred[..., 1],
        #         box_pred[..., 2] + grid_cell[..., 2],
        #         box_pred[..., 3] + grid_cell[..., 3]
        #     ], axis=-1)
        
        # encoded_box = tf.clip_by_value(encoded_box, clip_value_min=0, clip_value_max=1)
        # tf.print('here' * 10)
        mask = tf.math.reduce_sum(logits_label, axis=-1)
        mask = tf.math.greater(mask, tf.constant(0.))
        # encoded_box = tf.boolean_mask(encoded_box, mask)

        box_pred = tf.boolean_mask(box_pred, mask)
        box_label = tf.boolean_mask(box_label, mask)
        # tf.print(box_pred)
        # tf.print(box_label)
        box_loss = self.iou_loss(box_label, box_pred)

        # box_loss = tfa.losses.giou_loss(box_target, encoded_box)
        logits_loss = logits_loss / tf.reduce_sum(tf.cast(mask, tf.float32))
        box_loss = box_loss / tf.reduce_sum(tf.cast(mask, tf.float32))
        return logits_loss, box_loss

    def iou_loss(self, target, pred):
        pred_left = pred[:, 1]
        pred_top = pred[:, 0]
        pred_right = pred[:, 3]
        pred_bottom = pred[:, 2]

        target_left = target[:, 1]
        target_top = target[:, 0]
        target_right = target[:, 3]
        target_bottom = target[:, 2]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)

        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)
        w_intersect = tf.math.minimum(pred_left, target_left) + tf.math.minimum(pred_right, target_right)
        h_intersect = tf.math.minimum(pred_bottom, target_bottom) + tf.math.minimum(pred_top, target_top)
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        losses = -tf.math.log(ious)
        return tf.math.reduce_sum(losses)