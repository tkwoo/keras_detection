import tensorflow as tf


def _resize(img, labels, img_dim):
    ''' # resize and boxes coordinate to percent'''
    w_f = tf.cast(tf.shape(img)[1], tf.float32)
    h_f = tf.cast(tf.shape(img)[0], tf.float32)
    locs = tf.stack([labels[:, 0] / h_f,  labels[:, 1] / w_f,
                     labels[:, 2] / h_f,  labels[:, 3] / w_f] ,axis=1)
    locs = tf.clip_by_value(locs, 0, 1.0)
    labels = tf.concat([locs, labels[:, 4][:, tf.newaxis]], axis=1)

    resize_case = tf.random.uniform([], 0, 5, dtype=tf.int32)
    if isinstance(img_dim, int):
        img_dim = (img_dim, img_dim)
    elif isinstance(img_dim,(tuple, list)):
        img_dim = img_dim
    else:
        raise Exception('Type error of input image size format,tuple or int. ')

    def resize(method):
        def _resize():
            #　size h,w
            return tf.image.resize(img, [img_dim[0], img_dim[1]], method=method, antialias=True)
        return _resize

    img = tf.case([(tf.equal(resize_case, 0), resize('bicubic')),
                   (tf.equal(resize_case, 1), resize('area')),
                   (tf.equal(resize_case, 2), resize('nearest')),
                   (tf.equal(resize_case, 3), resize('lanczos3'))],
                  default=resize('bilinear'))

    return img, labels
