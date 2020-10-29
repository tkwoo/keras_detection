import tensorflow_addons as tfa


def build_loss(args):
    if args.TASK == 'classification':
        return 'sparse_categorical_crossentropy'
    elif args.TASK == 'triplet':
        return tfa.losses.TripletSemiHardLoss()
    else:
        assert False, f'unsupported loss: {args.TASK}'
