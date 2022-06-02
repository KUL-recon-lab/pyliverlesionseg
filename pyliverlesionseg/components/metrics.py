import numpy as np
from keras import backend as K
from keras.utils import to_categorical


def binary_crossentropy(y_true, y_pred, clip=1e-7):
    y_pred = np.clip(y_pred, clip, 1 - clip)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=(1, 2, 3, 4))


def binary_accuracy(y_true, y_pred, weights=None):
    if weights is None:
        w = 1
    else:
        w_n = (1 - np.round(y_true)) * weights[0]
        w_p = np.round(y_true) * weights[1]
        w = w_n + w_p
    return np.mean(w / np.mean(w) * np.equal(np.round(y_true), np.round(y_pred)), axis=(1, 2, 3, 4))


def binary_precision(y_true, y_pred, smooth=1e-7):
    TP = np.sum(np.round(y_true) * np.round(y_pred), axis=(1, 2, 3, 4))
    TP_FP = np.sum(np.round(y_pred), axis=(1, 2, 3, 4))
    return (TP + smooth) / (TP_FP + smooth)


def binary_recall(y_true, y_pred, smooth=1e-7):
    TP = np.sum(np.round(y_true) * np.round(y_pred), axis=(1, 2, 3, 4))
    TP_FN = np.sum(np.round(y_true), axis=(1, 2, 3, 4))
    return (TP + smooth) / (TP_FN + smooth)


def binary_dice(y_true, y_pred, smooth=1e-7):
    intersection = np.sum(np.round(y_true) * np.round(y_pred), axis=(1, 2, 3, 4))
    union = np.sum(np.round(y_true), axis=(1, 2, 3, 4)) + np.sum(np.round(y_pred), axis=(1, 2, 3, 4))
    return (2 * intersection + smooth) / (union + smooth)


def binary_soft_dice(y_true, y_pred, smooth=1e-7):
    intersection = np.sum(y_true * y_pred, axis=(1, 2, 3, 4))
    union = np.sum(y_true, axis=(1, 2, 3, 4)) + np.sum(y_pred, axis=(1, 2, 3, 4))
    return (2 * intersection + smooth) / (union + smooth)

def binary_class_dice(y_true, y_pred, class_i=2, smooth=1e-7):
    intersection = np.sum(np.round(y_true[..., class_i]) * np.round(y_pred[..., class_i]), axis=(1, 2, 3))
    union = np.sum(np.round(y_true[..., class_i]), axis=(1, 2, 3)) + np.sum(np.round(y_pred[..., class_i]), axis=(1, 2, 3))
    return (2 * intersection + smooth) / (union + smooth)

def binary_jaccard(y_true, y_pred, smooth=1e-7):
    intersection = np.sum(np.round(y_true) * np.round(y_pred), axis=(1, 2, 3, 4))
    union = np.sum(np.round(y_true), axis=(1, 2, 3, 4)) + np.sum(np.round(y_pred), axis=(1, 2, 3, 4))
    return (intersection + smooth) / (union - intersection + smooth)

def binary_tversky(y_true, y_pred, alpha=0.5, beta=0.5, per_image=True, smooth=1e-7):
    intersection = np.sum(np.round(y_true) * np.round(y_pred), axis=(1, 2, 3, 4) if per_image else None)
    fp = np.sum(np.round(y_pred), axis=(1, 2, 3, 4) if per_image else None) - intersection
    fn = np.sum(np.round(y_true), axis=(1, 2, 3, 4) if per_image else None) - intersection
    return (intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth) if per_image else [(intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth)]

def binary_soft_jaccard(y_true, y_pred, smooth=1e-7):
    intersection = np.sum(y_true * y_pred, axis=(1, 2, 3, 4))
    union = np.sum(y_true, axis=(1, 2, 3, 4)) + np.sum(y_pred, axis=(1, 2, 3, 4))
    return (intersection + smooth) / (union - intersection + smooth)


def binary_soft_vol_diff(y_true, y_pred, voxel_volume=1):
    return (np.sum(y_pred, axis=(1, 2, 3, 4)) - np.sum(y_true, axis=(1, 2, 3, 4))) * voxel_volume


def binary_vol_diff(y_true, y_pred, voxel_volume=1):
    return (np.sum(np.round(y_pred), axis=(1, 2, 3, 4)) - np.sum(np.round(y_true), axis=(1, 2, 3, 4))) * voxel_volume


def binary_relative_soft_vol_diff(y_true, y_pred):
    return (np.sum(y_pred, axis=(1, 2, 3, 4)) - np.sum(y_true, axis=(1, 2, 3, 4))) / np.sum(y_true, axis=(1, 2, 3, 4))


def binary_relative_vol_diff(y_true, y_pred):
    return (np.sum(np.round(y_pred), axis=(1, 2, 3, 4)) - np.sum(np.round(y_true), axis=(1, 2, 3, 4))) / np.sum(np.round(y_true), axis=(1, 2, 3, 4))


def binary_expected_calibration_error(y_true, y_pred, nb_bins=10):
    ECE = [0] * len(y_true)
    for min in np.arange(0, 1, 1 / nb_bins):
        max = min + 1 / nb_bins
        for i, (y_true_, y_pred_) in enumerate(zip(y_true, y_pred)):
            mask = (min <= y_pred_) * (y_pred_ <= max)
            ECE[i] += np.sum(mask) * np.abs(np.mean(y_pred_[mask]) - np.mean(y_true_[mask])) / y_true_.size
    return ECE


def binary_true_positives(y_true, y_pred):
    return np.sum(np.round(y_true) * np.round(y_pred), axis=(1, 2, 3, 4))


def binary_true_negatives(y_true, y_pred):
    return np.sum(np.round(1 - y_true) * np.round(1 - y_pred), axis=(1, 2, 3, 4))


def binary_false_positives(y_true, y_pred):
    return np.sum(np.round(1 - y_true) * np.round(y_pred), axis=(1, 2, 3, 4))


def binary_false_negatives(y_true, y_pred):
    return np.sum(np.round(y_true) * np.round(1 - y_pred), axis=(1, 2, 3, 4))


def categorical_crossentropy(y_true, y_pred, clip=1e-7):
    y_pred = np.clip(y_pred, clip, 1 - clip)
    ce = np.sum(y_true * np.log(y_pred), axis=-1)
    return -np.mean(ce, axis=(1, 2, 3))


def categorical_accuracy(y_true, y_pred):
    return np.mean(np.equal(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1)), axis=(1, 2, 3))


def categorical_dice(y_true, y_pred, smooth=1e-7):
    nb_classes = y_pred.shape[-1]
    dice = 0
    for class_i in range(1, nb_classes):
        intersection = np.sum(np.round(y_true[..., class_i]) * np.round(y_pred[..., class_i]), axis=(1, 2, 3))
        union = np.sum(np.round(y_true[..., class_i]), axis=(1, 2, 3)) + np.sum(np.round(y_pred[..., class_i]), axis=(1, 2, 3))
        dice += (2 * intersection + smooth) / (union + smooth)
    return dice / (nb_classes - 1)


def categorical_soft_dice(y_true, y_pred, smooth=1e-7):
    nb_classes = y_pred.shape[-1]
    dice = 0
    for class_i in range(1, nb_classes):
        intersection = np.sum(y_true[..., class_i] * y_pred[..., class_i], axis=(1, 2, 3))
        union = np.sum(y_true[..., class_i], axis=(1, 2, 3)) + np.sum(y_pred[..., class_i], axis=(1, 2, 3))
        dice += (2 * intersection + smooth) / (union + smooth)
    return dice / (nb_classes - 1)


def categorical_jaccard(y_true, y_pred, smooth=1e-7):
    nb_classes = y_pred.shape[-1]
    jaccard = 0
    for class_i in range(1, nb_classes):
        intersection = np.sum(np.round(y_true[..., class_i]) * np.round(y_pred[..., class_i]), axis=(1, 2, 3))
        union = np.sum(np.round(y_true[..., class_i]), axis=(1, 2, 3)) + np.sum(np.round(y_pred[..., class_i]), axis=(1, 2, 3))
        jaccard += (intersection + smooth) / (union - intersection + smooth)
    return jaccard / (nb_classes - 1)


def categorical_soft_jaccard(y_true, y_pred, smooth=1e-7):
    nb_classes = y_pred.shape[-1]
    jaccard = 0
    for class_i in range(1, nb_classes):
        intersection = np.sum(y_true[..., class_i] * y_pred[..., class_i], axis=(1, 2, 3))
        union = np.sum(y_true[..., class_i], axis=(1, 2, 3)) + np.sum(y_pred[..., class_i], axis=(1, 2, 3))
        jaccard += (intersection + smooth) / (union - intersection + smooth)
    return jaccard / (nb_classes - 1)

def categorical_tversky(y_true, y_pred, alpha=0.5, beta=0.5, per_image=True, smooth=1e-7):
    nb_classes = y_pred.shape[-1]
    tversky = 0
    for class_i in range(1, nb_classes):
        intersection = np.sum(np.round(y_true[..., class_i]) * np.round(y_pred[..., class_i]), axis=(1, 2, 3))
        fp = np.sum(np.round(y_pred[..., class_i]), axis=(1, 2, 3)) - intersection
        fn = np.sum(np.round(y_true[..., class_i]), axis=(1, 2, 3)) - intersection
        tversky += (intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth)
    return tversky / (nb_classes - 1)

def categorical_soft_tversky(y_true, y_pred, alpha=0.5, beta=0.5, per_image=True, smooth=1e-7):
    nb_classes = y_pred.shape[-1]
    tversky = 0
    for class_i in range(1, nb_classes):
        intersection = np.sum(y_true[..., class_i] * y_pred[..., class_i], axis=(1, 2, 3))
        fp = np.sum(y_pred[..., class_i], axis=(1, 2, 3)) - intersection
        fn = np.sum(y_true[..., class_i], axis=(1, 2, 3)) - intersection
        tversky += (intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth)
    return tversky / (nb_classes - 1)

def categorical_soft_vol_diff(y_true, y_pred):
    nb_classes = y_pred.shape[-1]
    soft_vol_diff = 0
    for class_i in range(1, nb_classes):
        soft_vol_diff += np.sum(y_pred[..., class_i], axis=(1, 2, 3)) - np.sum(y_true[..., class_i], axis=(1, 2, 3))
    return soft_vol_diff / (nb_classes - 1)


def categorical_mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(np.argmax(y_true, axis=-1) - np.argmax(y_pred, axis=-1)), axis=(1, 2, 3))


def categorical_linear_weighted_kappa(y_true, y_pred):
    nb_classes = y_pred.shape[-1]
    y_pred = to_categorical(np.argmax(y_pred, axis=-1)[..., None], nb_classes)
    nb_samples = np.sum(y_true)
    y_true_prior = np.sum(y_true, axis=(0, 1, 2, 3)) / nb_samples
    y_pred_prior = np.sum(y_pred, axis=(0, 1, 2, 3)) / nb_samples
    random_confusion_matrix = np.dot(y_true_prior[..., None], y_pred_prior[None, ...])
    true_confusion_matrix = np.array([np.sum(y_true[..., [class_i]] * y_pred, axis=(0, 1, 2, 3)) for class_i in range(nb_classes)]) / nb_samples
    weight_matrix = np.abs(np.array([np.arange(-class_i, nb_classes - class_i) for class_i in range(nb_classes)]))
    return [1 - np.sum(weight_matrix * true_confusion_matrix) / np.sum(weight_matrix * random_confusion_matrix)]

def regression_mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2, axis=(1,2,3,4))

def regression_mean_absolute_error(y_true, y_pred):
    return np.mean((y_true - y_pred), axis=(1,2,3,4))



lookuptable = {
    "binary_crossentropy":                  binary_crossentropy,
    "binary_accuracy":                      binary_accuracy,
    "binary_precision":                     binary_precision,
    "binary_recall":                        binary_recall,
    "binary_dice":                          binary_dice,
    "binary_class_dice":                    binary_class_dice,
    "binary_soft_dice":                     binary_soft_dice,
    "binary_jaccard":                       binary_jaccard,
    "binary_soft_jaccard":                  binary_soft_jaccard,
    "binary_soft_vol_diff":                 binary_soft_vol_diff,
    "binary_expected_calibration_error":    binary_expected_calibration_error,
    "categorical_crossentropy":             categorical_crossentropy,
    "categorical_accuracy":                 categorical_accuracy,
    "categorical_dice":                     categorical_dice,
    "categorical_soft_dice":                categorical_soft_dice,
    "categorical_tversky":                  categorical_tversky,
    "categorical_soft_tversky":             categorical_soft_tversky,
    "categorical_mean_absolute_error":      categorical_mean_absolute_error,
    "categorical_linear_weighted_kappa":    categorical_linear_weighted_kappa,
    "regression_mean_squared_error":        regression_mean_squared_error,
    "regression_mean_absolute_error":       regression_mean_absolute_error
}
