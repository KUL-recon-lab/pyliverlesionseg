from __future__ import print_function, division
from keras import backend as K
import numpy as np
import tensorflow as tf

def binary_crossentropy(y_true, y_pred, weights=[1, 1], focal=False, sampling_correction=False):
    if sampling_correction:
        correction_weights = y_true[..., 1]
    else:
        correction_weights = 1
    y_pred = y_pred[..., 0]
    y_true = y_true[..., 0]
    if focal:
        weight_mask = weights[0] * y_pred * (1 - y_true) + weights[1] * (1 - y_pred) * y_true
    else:
        weight_mask = weights[0] * (1 - y_true) + weights[1] * y_true
    return K.mean(correction_weights * weight_mask * K.binary_crossentropy(y_true, y_pred), axis=[1, 2, 3])


def binary_accuracy(y_true, y_pred, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)), axis=[1, 2, 3, 4])


def binary_precision(y_true, y_pred, smooth=1e-7, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    TP = K.sum(K.round(y_true) * K.round(y_pred), axis=[1, 2, 3, 4])
    TP_FP = K.sum(K.round(y_pred), axis=[1, 2, 3, 4])
    return (TP + smooth) / (TP_FP + smooth)


def binary_recall(y_true, y_pred, smooth=1e-7, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    TP = K.sum(K.round(y_true) * K.round(y_pred), axis=[1, 2, 3, 4])
    TP_FN = K.sum(K.round(y_true), axis=[1, 2, 3, 4])
    return (TP + smooth) / (TP_FN + smooth)


def binary_dice(y_true, y_pred, per_image=True, smooth=1e-7, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    intersection = K.sum(K.round(y_true) * K.round(y_pred), axis=[1, 2, 3, 4] if per_image else None)
    union = K.sum(K.round(y_true), axis=[1, 2, 3, 4] if per_image else None) + K.sum(K.round(y_pred), axis=[1, 2, 3, 4] if per_image else None)
    return (2 * intersection + smooth) / (union + smooth) if per_image else [(2 * intersection + smooth) / (union + smooth)]

def binary_liver_dice(y_true, y_pred, class_i=1, per_image=True, smooth=1e-7, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    intersection = K.sum(K.round(y_true[..., class_i]) * K.round(y_pred[...,class_i]), axis=[1, 2, 3] if per_image else None)
    union = K.sum(K.round(y_true[..., class_i]), axis=[1, 2, 3] if per_image else None) + K.sum(K.round(y_pred[[...,class_i]]), axis=[1, 2, 3] if per_image else None)
    return (2 * intersection + smooth) / (union + smooth) if per_image else [(2 * intersection + smooth) / (union + smooth)]

def binary_tumor_dice(y_true, y_pred, class_i=2, per_image=True, smooth=1e-7, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    intersection = K.sum(K.round(y_true[..., class_i]) * K.round(y_pred[...,class_i]), axis=[1, 2, 3] if per_image else None)
    union = K.sum(K.round(y_true[..., class_i]), axis=[1, 2, 3] if per_image else None) + K.sum(K.round(y_pred[[...,class_i]]), axis=[1, 2, 3] if per_image else None)
    return (2 * intersection + smooth) / (union + smooth) if per_image else [(2 * intersection + smooth) / (union + smooth)]

def binary_soft_dice(y_true, y_pred, per_image=True, smooth=1e-7, sampling_correction=False, norm='L1'):
    if sampling_correction:
        y_true = y_true[..., :-1]
    if norm == 'L1':
        intersection = K.sum(y_true * y_pred, axis=[1, 2, 3, 4] if per_image else None)
        union = K.sum(y_true, axis=[1, 2, 3, 4] if per_image else None) + K.sum(y_pred, axis=[1, 2, 3, 4] if per_image else None)
    elif norm == 'L2':
        intersection = K.sqrt(K.sum(y_true * y_pred, axis=[1, 2, 3, 4] if per_image else None))
        union = K.sqrt(K.sum(K.square(y_true), axis=[1, 2, 3, 4] if per_image else None)) + K.sqrt(K.sum(K.square(y_pred), axis=[1, 2, 3, 4] if per_image else None))
    return (2 * intersection + smooth) / (union + smooth) if per_image else [(2 * intersection + smooth) / (union + smooth)]


def binary_jaccard(y_true, y_pred, per_image=True, smooth=1e-7, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    intersection = K.sum(K.round(y_true) * K.round(y_pred), axis=[1, 2, 3, 4] if per_image else None)
    union = K.sum(K.round(y_true), axis=[1, 2, 3, 4] if per_image else None) + K.sum(K.round(y_pred), axis=[1, 2, 3, 4] if per_image else None)
    return (intersection + smooth) / (union - intersection + smooth) if per_image else [(intersection + smooth) / (union - intersection + smooth)]


def binary_soft_jaccard(y_true, y_pred, per_image=True, smooth=1e-7, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3, 4] if per_image else None)
    union = K.sum(y_true, axis=[1, 2, 3, 4] if per_image else None) + K.sum(y_pred, axis=[1, 2, 3, 4] if per_image else None)
    return (intersection + smooth) / (union - intersection + smooth) if per_image else [(intersection + smooth) / (union - intersection + smooth)]


def binary_tversky(y_true, y_pred, alpha=0.5, beta=0.5, per_image=True, smooth=1e-7, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    intersection = K.sum(K.round(y_true) * K.round(y_pred), axis=[1, 2, 3, 4] if per_image else None)
    fp = K.sum(K.round(y_pred), axis=[1, 2, 3, 4] if per_image else None) - intersection
    fn = K.sum(K.round(y_true), axis=[1, 2, 3, 4] if per_image else None) - intersection
    return (intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth) if per_image else [(intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth)]


def binary_soft_tversky(y_true, y_pred, alpha=0.5, beta=0.5, per_image=True, smooth=1e-7, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3, 4] if per_image else None)
    fp = K.sum(y_pred, axis=[1, 2, 3, 4] if per_image else None) - intersection
    fn = K.sum(y_true, axis=[1, 2, 3, 4] if per_image else None) - intersection
    return (intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth) if per_image else [(intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth)]

def binary_tumor_soft_tversky(y_true, y_pred, class_i=2, alpha=0.5, beta=0.5, per_image=True, smooth=1e-7, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    intersection = K.sum(y_true[..., class_i] * y_pred[..., class_i], axis=[1, 2, 3] if per_image else None)
    fp = K.sum(y_pred[..., class_i], axis=[1, 2, 3] if per_image else None) - intersection
    fn = K.sum(y_true[..., class_i], axis=[1, 2, 3] if per_image else None) - intersection
    return (intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth) if per_image else [(intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth)]


def binary_vol_diff(y_true, y_pred, per_image=True, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    return K.sum(K.round(y_pred), axis=[1, 2, 3, 4]) - K.sum(K.round(y_true), axis=[1, 2, 3, 4]) if per_image else [K.sum(K.round(y_pred)) - K.sum(K.round(y_true))]


def binary_soft_vol_diff(y_true, y_pred, per_image=True, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    return K.sum(y_pred, axis=[1, 2, 3, 4]) - K.sum(y_true, axis=[1, 2, 3, 4]) if per_image else [K.sum(y_pred) - K.sum(y_true)]


def binary_dice_loss(y_true, y_pred, per_image=True, smooth=1e-7, sampling_correction=False):
    return 1 - binary_soft_dice(y_true, y_pred, per_image=per_image, smooth=smooth, sampling_correction=sampling_correction)


def binary_jaccard_loss(y_true, y_pred, per_image=True, smooth=1e-7, sampling_correction=False):
    return 1 - binary_soft_jaccard(y_true, y_pred, per_image=per_image, smooth=smooth, sampling_correction=sampling_correction)


def binary_tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5, per_image=True, smooth=1e-7, sampling_correction=False):
    return 1 - binary_soft_tversky(y_true, y_pred, alpha=alpha, beta=beta, per_image=per_image, smooth=smooth, sampling_correction=sampling_correction)

def binary_tumor_tversky_loss(y_true, y_pred, class_i=2, alpha=0.5, beta=0.5, per_image=True, smooth=1e-7, sampling_correction=False):
    return 1 - binary_tumor_soft_tversky(y_true, y_pred, class_i=class_i, alpha=alpha, beta=beta, per_image=per_image, smooth=smooth, sampling_correction=sampling_correction)

def categorical_crossentropy(y_true, y_pred, weights=None, sampling_correction=False):
    if sampling_correction:
        correction_weights = y_true[..., -1]
        y_true = y_true[..., :-1]
    else:
        correction_weights = 1
    if weights is None:
        weights = 1
    else:
        weights = K.sum(y_true * weights, axis=-1)
    return K.mean(K.categorical_crossentropy(y_true, y_pred), axis=[1, 2, 3])


def categorical_binary_crossentropy(y_true, y_pred, weights=None, focal=False, exclude_background=False, sampling_correction=False):
    if sampling_correction:
        correction_weights = y_true[..., -1]
        y_true = y_true[..., :-1]
    else:
        correction_weights = 1
    if weights is None:
        nb_classes = K.int_shape(y_pred)[-1]
        weights = [1] * nb_classes
    if exclude_background:
        weights[0] = 0
    if focal:
        weight_mask = K.variable(weights) * (y_true * (1 - y_pred) + (1 - y_true) * y_pred)
    else:
        weight_mask = K.variable(weights) * K.ones_like(y_true)
    return K.mean(correction_weights * K.mean(weight_mask * K.binary_crossentropy(y_true, y_pred), axis=-1), axis=[1, 2, 3])


def categorical_accuracy(y_true, y_pred, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    return K.mean(K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), "float32"), axis=[1, 2, 3])


def categorical_dice(y_true, y_pred, smooth=1e-7, exclude_background=True, per_image=True, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    if exclude_background:
        y_true = y_true[..., 1:]
        y_pred = y_pred[..., 1:]
    intersections = K.sum(K.round(y_true) * K.round(y_pred), axis=[1, 2, 3] if per_image else [0, 1, 2, 3])
    unions = K.sum(K.round(y_true), axis=[1, 2, 3] if per_image else [0, 1, 2, 3]) + K.sum(K.round(y_pred), axis=[1, 2, 3] if per_image else [0, 1, 2, 3])
    dices = (2 * intersections + smooth) / (unions + smooth)
    return K.mean(dices, axis=1) if per_image else K.mean(dices, keepdims=True)


def categorical_soft_dice(y_true, y_pred, smooth=1e-7, exclude_background=True, per_image=True, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    if exclude_background:
        y_true = y_true[..., 1:]
        y_pred = y_pred[..., 1:]
    intersections = K.sum(y_true * y_pred, axis=[1, 2, 3] if per_image else [0, 1, 2, 3])
    unions = K.sum(y_true, axis=[1, 2, 3] if per_image else [0, 1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3] if per_image else [0, 1, 2, 3])
    soft_dices = (2 * intersections + smooth) / (unions + smooth)
    return K.mean(soft_dices, axis=1) if per_image else K.mean(soft_dices, keepdims=True)


def categorical_jaccard(y_true, y_pred, smooth=1e-7, exclude_background=True, per_image=True, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    if exclude_background:
        y_true = y_true[..., 1:]
        y_pred = y_pred[..., 1:]
    intersections = K.sum(K.round(y_true) * K.round(y_pred), axis=[1, 2, 3] if per_image else [0, 1, 2, 3])
    unions = K.sum(K.round(y_true), axis=[1, 2, 3] if per_image else [0, 1, 2, 3]) + K.sum(K.round(y_pred), axis=[1, 2, 3] if per_image else [0, 1, 2, 3])
    jaccards = (intersections + smooth) / (unions - intersections + smooth)
    return K.mean(jaccards, axis=1) if per_image else K.mean(jaccards, keepdims=True)


def categorical_soft_jaccard(y_true, y_pred, smooth=1e-7, exclude_background=True, per_image=True, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    if exclude_background:
        y_true = y_true[..., 1:]
        y_pred = y_pred[..., 1:]
    intersections = K.sum(y_true * y_pred, axis=[1, 2, 3] if per_image else [0, 1, 2, 3])
    unions = K.sum(y_true, axis=[1, 2, 3] if per_image else [0, 1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3] if per_image else [0, 1, 2, 3])
    soft_jaccards = (intersections + smooth) / (unions - intersections + smooth)
    return K.mean(soft_jaccards, axis=1) if per_image else K.mean(soft_jaccards, keepdims=True)


def categorical_tversky(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1e-7, exclude_background=True, per_image=True, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    if exclude_background:
        y_true = y_true[..., 1:]
        y_pred = y_pred[..., 1:]
    intersections = K.sum(K.round(y_true) * K.round(y_pred), axis=[1, 2, 3] if per_image else [0, 1, 2, 3])
    fps = K.sum(K.round(y_pred), axis=[1, 2, 3] if per_image else [0, 1, 2, 3]) - intersections
    fns = K.sum(K.round(y_true), axis=[1, 2, 3] if per_image else [0, 1, 2, 3]) - intersections
    soft_tverskys = (intersections + smooth) / (intersections + alpha * fps + beta * fns + smooth)
    return K.mean(soft_tverskys, axis=1) if per_image else K.mean(soft_tverskys, keepdims=True)


def categorical_soft_tversky(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1e-7, exclude_background=True, per_image=True, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    if exclude_background:
        y_true = y_true[..., 1:]
        y_pred = y_pred[..., 1:]
    intersections = K.sum(y_true * y_pred, axis=[1, 2, 3] if per_image else [0, 1, 2, 3])
    fps = K.sum(y_pred, axis=[1, 2, 3] if per_image else [0, 1, 2, 3]) - intersections
    fns = K.sum(y_true, axis=[1, 2, 3] if per_image else [0, 1, 2, 3]) - intersections
    soft_tverskys = (intersections + smooth) / (intersections + alpha * fps + beta * fns + smooth)
    return K.mean(soft_tverskys, axis=1) if per_image else K.mean(soft_tverskys, keepdims=True)


def categorical_dice_loss(y_true, y_pred, smooth=1e-7, exclude_background=True, per_image=True, sampling_correction=False):
    return 1 - categorical_soft_dice(y_true, y_pred, smooth=smooth, exclude_background=exclude_background, per_image=per_image, sampling_correction=sampling_correction)


def categorical_jaccard_loss(y_true, y_pred, smooth=1e-7, exclude_background=True, per_image=True, sampling_correction=False):
    return 1 - categorical_soft_jaccard(y_true, y_pred, smooth=smooth, exclude_background=exclude_background, per_image=per_image, sampling_correction=sampling_correction)


def categorical_tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1e-7, exclude_background=True, per_image=True, sampling_correction=False):
    return 1 - categorical_soft_tversky(y_true, y_pred, alpha=alpha, beta=beta, smooth=smooth, exclude_background=exclude_background, per_image=per_image, sampling_correction=sampling_correction)


def categorical_vol_diff(y_true, y_pred, exclude_background=True, per_image=True, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    if exclude_background:
        y_true = y_true[..., 1:]
        y_pred = y_pred[..., 1:]
    soft_vol_diffs = K.sum(K.round(y_pred), axis=[1, 2, 3] if per_image else [0, 1, 2, 3]) - K.sum(K.round(y_true), axis=[1, 2, 3] if per_image else [0, 1, 2, 3])
    return K.mean(soft_vol_diffs, axis=1) if per_image else K.mean(soft_vol_diffs, keepdims=True)


def categorical_soft_vol_diff(y_true, y_pred, exclude_background=True, per_image=True, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    if exclude_background:
        y_true = y_true[..., 1:]
        y_pred = y_pred[..., 1:]
    soft_vol_diffs = K.sum(y_pred, axis=[1, 2, 3] if per_image else [0, 1, 2, 3]) - K.sum(y_true, axis=[1, 2, 3] if per_image else [0, 1, 2, 3])
    return K.mean(soft_vol_diffs, axis=1) if per_image else K.mean(soft_vol_diffs, keepdims=True)


def categorical_mean_absolute_error(y_true, y_pred, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    return K.mean(K.abs(K.cast(K.argmax(y_true, axis=-1), dtype="float32") - K.cast(K.argmax(y_pred, axis=-1), dtype="float32")), axis=[1, 2, 3])


def categorical_linear_weighted_kappa(y_true, y_pred, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.one_hot(K.argmax(y_pred, axis=-1), nb_classes)
    nb_samples = K.sum(y_true)
    random_confusion_matrix = K.dot(K.sum(y_true, axis=[0, 1, 2, 3])[..., None], K.sum(y_pred, axis=[0, 1, 2, 3])[None, ...]) / (nb_samples ** 2)
    true_confusion_matrix = []
    weight_matrix = np.zeros((nb_classes, nb_classes))
    for i in range(nb_classes):
        true_confusion_matrix_ = []
        for j in range(nb_classes):
            weight_matrix[i, j] = abs(i - j)
            true_confusion_matrix_.append(K.sum(y_true[..., j] * y_pred[..., i]))
        true_confusion_matrix.append(K.stack(true_confusion_matrix_, axis=-1))
    true_confusion_matrix = K.stack(true_confusion_matrix, axis=-1)
    kappa = 1 - K.sum(true_confusion_matrix * weight_matrix) / K.sum(random_confusion_matrix * weight_matrix)
    return kappa


"""
NEXT CODE IS STILL UNDER DEVELOPMENT! (from https://github.com/bermanmaxim/LovaszSoftmax)
"""


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None, order='BHWC'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """
    if per_image:
        def treat_image(prob_lab):
            prob, lab = prob_lab
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = flatten_probas(prob, lab, ignore, order)
            return lovasz_softmax_flat(prob, lab, classes=classes)
        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore, order), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    C = probas.shape[1]
    losses = []
    present = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = tf.cast(tf.equal(labels, c), probas.dtype)  # foreground for class c
        if classes == 'present':
            present.append(tf.reduce_sum(fg) > 0)
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = tf.abs(fg - class_pred)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
        fg_sorted = tf.gather(fg, perm)
        grad = lovasz_grad(fg_sorted)
        losses.append(
            tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c))
                      )
    if len(class_to_sum) == 1:  # short-circuit mean when only one class
        return losses[0]
    losses_tensor = tf.stack(losses)
    if classes == 'present':
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(losses_tensor, present)
    loss = tf.reduce_mean(losses_tensor)
    return loss


def flatten_probas(probas, labels, ignore=None, order='BHWC'):
    """
    Flattens predictions in the batch
    """
    if len(probas.shape) == 3:
        probas, order = tf.expand_dims(probas, 3), 'BHWC'
    if order == 'BCHW':
        probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
        order = 'BHWC'
    if order != 'BHWC':
        raise NotImplementedError('Order {} unknown'.format(order))
    C = probas.shape[3]
    probas = tf.reshape(probas, (-1, C))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return probas, labels
    valid = tf.not_equal(labels, ignore)
    vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vprobas, vlabels


def binary_lovasz_softmax_loss(y_true, y_pred, exclude_background=True, per_image=True, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    B, X, Y, Z, C = K.int_shape(y_pred)
    def reduce_z_dim(tensor):
        return K.reshape(tensor, (int(X), int(Y * Z), int(C)))
    y_true = tf.map_fn(reduce_z_dim, y_true, dtype=tf.float32)
    y_pred = tf.map_fn(reduce_z_dim, y_pred, dtype=tf.float32)
    y_true = K.cast(K.round(y_true), dtype="int64")
    return lovasz_softmax(y_pred[..., 0], y_true[..., 0], classes=[1] if exclude_background else 'present', per_image=per_image)


def categorical_binary_lovasz_softmax_loss(y_true, y_pred, exclude_background=True, per_image=True, sampling_correction=False):
    if sampling_correction:
        y_true = y_true[..., :-1]
    if exclude_background:
        y_true = y_true[..., 1:]
        y_pred = y_pred[..., 1:]
    B, X, Y, Z, C = K.int_shape(y_pred)
    y_true = K.reshape(y_true, (int(2 * C), int(X), int(Y), int(Z), int(1)))
    y_pred = K.reshape(y_pred, (int(2 * C), int(X), int(Y), int(Z), int(1)))
    def reduce_z_dim(tensor):
        return K.reshape(tensor, (int(X), int(Y * Z), int(1)))
    y_true = tf.map_fn(reduce_z_dim, y_true, dtype=tf.float32)
    y_pred = tf.map_fn(reduce_z_dim, y_pred, dtype=tf.float32)
    y_true = K.cast(K.round(y_true), dtype="int64")
    return lovasz_softmax(y_pred[..., 0], y_true[..., 0], classes=[1] if exclude_background else 'present', per_image=per_image)
