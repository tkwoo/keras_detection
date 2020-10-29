from detection.model import build_compiled_model
from detection.datasets.dataset import build_data
from detection.configs import cfg
from detection.callbacks.engine import build_callbacks
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow as tf

import argparse


def main():
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

    # float16, mixed precision
    if cfg.MIXED_PRECISION:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
    
    if cfg.MULTI_GPU:
        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            model = build_compiled_model(cfg)
    else:
        model = build_compiled_model(cfg)
    data = build_data(cfg)
    # print(type(data.train_tfrecords))
    model.fit(
        data.train_tfrecords,
        steps_per_epoch=data.train_length,
        validation_data=data.val_tfrecords,
        use_multiprocessing=True,
        workers=6,
        epochs=cfg.EPOCH,
        callbacks=build_callbacks(cfg)
    )


if __name__ == '__main__':
    main()
