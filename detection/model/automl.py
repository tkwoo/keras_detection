import autokeras as ak


def make_automl_model(args):
    return ak.ImageClassifier(
        num_classes=args.MODEL.NUM_CLASSES,
        loss='sparse_categorical_crossentropy',
        metrics = ['accuracy'],
        directory=args.OUTPUT_DIR,
        max_trials=args.MODEL.AUTOML_TRIALS,
        objective="val_loss")