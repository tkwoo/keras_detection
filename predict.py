import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from detection.model import build_compiled_model
from detection.datasets.dataset import build_data
from detection.configs import cfg

import argparse
from pathlib import Path
import cv2
import numpy as np


def get_model():
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--restart", default=0, type=int)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    print(cfg)
    
    latest = tf.train.latest_checkpoint(cfg.OUTPUT_DIR)
    model = build_compiled_model(cfg)
    model.load_weights(latest)
    # model = model.export()
    print(model.model.layers)
    return model


def put_text(img, texts):
        # You may need to adjust text size and position and size.
        # If your images are in [0, 255] range replace (0, 0, 1) with (0, 0, 255)
    h = 30
    for text in texts:
        img = cv2.putText(img, str(text), (0, h), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)
        h += 30

    return img


def get_grid(h=38, w=22):
    x, y = tf.meshgrid(tf.range(w), tf.range(h))
    x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)
    x += 0.5
    y += 0.5
    # h, w
    x /= w
    y /= h
    # h, w, 4 (yxyx)
    grid_cell = tf.stack([y, x, y, x], axis=-1)
    return grid_cell.numpy()


def predict_and_show(model, ):
    val_dir = ['/home/ocrusr/datasets/ppe/final/upper_cropped_1/']#cfg.VAL_DIR
    # val_dir = cfg.VAL_DIR
    val_dir = Path(val_dir[0])
    class_names = [d.name for d in val_dir.glob('*')]
    val_images = val_dir.glob('*/*.jpg')
    val_images = list(val_images)
    # print(val_images)
    np.random.shuffle(val_images)
    classes = cfg.MODEL.CLASSES
    
    class_names = ['background', 'hwear', 'hunwear']
    corr = 0
    total = 0
    for idx, image_path in enumerate(val_images):
        print(image_path)
        if idx == 6:
            break
        # if 'unk' in str(image_path):
        #     continue
        gt_name = image_path.parent.name
        # gt_label = class_names.index(gt_name)
        # if gt_label == 2:
        #     continue
        total += 1
        image_raw = tf.io.read_file(str(image_path))
        image = tf.image.decode_jpeg(image_raw, dct_method='INTEGER_ACCURATE')
        # print(image)
        # print(type(image))
        # image = tf.image.resize(image, cfg.DATA.SIZE)
        image_ = image / 255
        image_ = image_ - [[cfg.DATA.MEAN]]
        image_ = image_ / [[cfg.DATA.STD]]
        image_ = tf.image.resize(image_, [cfg.DATA.SIZE[1], cfg.DATA.SIZE[0]])
        # print(image_)
        image_ = np.expand_dims(image_, axis=0)
        logits, boxes = model.model(image_, False)
        # logits = tf.nn.softmax(logits)
        logits, boxes = logits.numpy(), boxes.numpy()
        # print(logits)
        # max_value = np.amax(logits)
        # print(max_value)
        # logits = np.amax(logits, axis=-1)
        # mask = logits == max_value
        # # logits = logits[..., 1:]
        logits = logits > 0.5

        mask = np.sum(logits, axis=-1, keepdims=False) > 0
        # print(np.sum(mask))
        grid_cell = get_grid()
        # print(grid_cell[40, 20])
        # print(boxes[0, 0:10,])
        boxes = np.stack([
                grid_cell[..., 0] - boxes[..., 0],
                grid_cell[..., 1] - boxes[..., 1],
                boxes[..., 2] + grid_cell[..., 2],
                boxes[..., 3] + grid_cell[..., 3]
            ], axis=-1)
        # print(boxes.shape)
        # print(boxes[0, 40, 20])
        boxes = boxes[mask].reshape([-1, 4])
        boxes = np.clip(boxes, a_min=0, a_max=1)
        print(boxes)
        image = image.numpy()
        h, w, c = image.shape
        print(h, ' ', w)
        for b in boxes:
            image = cv2.rectangle(image, (int(b[1] * w), int(b[0] * h)), (int(b[3] * w), int(b[2] * h)), (255, 0, 0), 1)
        cv2.imwrite(f'{idx}.jpg', image.astype(np.uint8))
        # # results = model.predict(image_)
        # results = results[0]
        # pred = np.argmax(results)
        # pred_value = np.max(results)
        # if pred_value < 0.7:
        #     pred = 2
        # print(pred)
        # if pred == gt_label:
        #     corr += 1
        # # if gt_label == 2:
        # #     print(pred_value)
        # else:
        #     print(gt_label, ' but ', pred, ' value: ', pred_value)
        # print(results)
    #     texts = [cn + ': ' + str(r) for cn, r in zip(['person', 'falldown'], results.tolist())]
    #     img = put_text(image.numpy().astype(np.uint8), texts)
    #     cv2.imshow('t', img)
    #     k = cv2.waitKey(0)
    #     if k == 27: # esc key
    #         cv2.destroyAllWindows()
    #         break
    #     print(corr/total)
    # print(corr/total)


def main():
    model = get_model()
    predict_and_show(model)

if __name__ == '__main__':
    main()
