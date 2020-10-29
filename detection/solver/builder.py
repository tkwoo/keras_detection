import tensorflow as tf


__all__ = ['sgd', 'adam']


def SGD(lr):
    return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)


def ADAM(lr):
    return tf.keras.optimizers.Adam(learning_rate=lr)


__pair = {
    'sgd': SGD,
    'adam': ADAM
}

def build_optimizer(cfg):
    assert cfg.SOLVER.NAME in __pair
    
    optimizer = __pair[cfg.SOLVER.NAME](cfg.SOLVER.LR)

    return optimizer
