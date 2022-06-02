import os
import sys
import time
import math
import numpy as np
import nibabel as nib
import keras.backend as K
from . import sampling
from pyliverlesionseg.sampling.samplecreator import SampleCreator
from pyliverlesionseg.sampling.sampler import Sampler
from pyliverlesionseg.sampling.imageloader import ImageLoader
from pyliverlesionseg.components.callbacks import MetricNameChanger, History, FullImageSetTester
from pyliverlesionseg.components.layer_resolution import get_layers_scales
from scipy import ndimage
from keras import Model
from keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical
if sys.version_info >= (3, 0):
    import pickle
else:
    import cPickle as pickle
import pdb

class DeepVoxNet(object):

    def __init__(
            self,
            model,
            nb_classes_in_multiclass=None,
            center_sampling=False,
            classification=False
    ):
        """
        :param model: This is the Keras model of your network. Its parameters will be optimized when the train function is called. You can use the model to predict a new instance by using the predict function. Load and save functions allow you to load and save the complete DeepVoxNet model.
        :type model: Model

        :param training_and_prediction_priors: If specified, these should be the fractions of each class present in the training set (i.e. source distribution) and in the prediction set (i.e. target distribution). If the model has multiple outputs, these must be specified for each output in a dictonary.
        :type training_and_prediction_priors: None
        :type training_and_prediction_priors: (list or tuple) of (list or tuple) of float
        :type training_and_prediction_priors: (dict of str: (None or (list or tuple) of (list or tuple) of float))

        :param nb_classes_in_multiclass: If multi-class the number of classes must be specified. If this model has multiple outputs, these must be specified for each output in a dictionary.
        :type nb_classes_in_multiclass: None
        :type nb_classes_in_multiclass: int
        :type nb_classes_in_multiclass: list of int
        :type nb_classes_in_multiclass: (dict of str: int)

        :param center_sampling: Must be set to True when prediction is to be performed only on a segment at the center of the image.
        :type center_sampling: bool

        :param classification: Must be set to True if image classification instead of segmentation is performed. If the model has multiple outputs, this must be specified for each output in a dictionary.
        :type classification: bool
        :type classification: list of bool
        :type classification: (dict of str: bool)
        """

        self.model = model

        if nb_classes_in_multiclass and not isinstance(nb_classes_in_multiclass, dict):
            nb_classes_in_multiclass = {self.model.output_names[0]: nb_classes_in_multiclass}
        self.nb_classes_in_multiclass = nb_classes_in_multiclass

        self.center_sampling = center_sampling

        if classification and not isinstance(classification, dict):
            classification = {self.model.output_names[0]: classification}
        self.classification = classification

    def train(
            self,
            training_x_creator,
            training_y_creator,
            training_sampler,
            validation_x_creator=None,
            validation_y_creator=None,
            validation_sampler=None,
            full_training_x_creator=None,
            full_training_sampler=None,
            full_training_gt_loader=None,
            full_training_output_paths=None,
            full_training_metrics=None,
            full_training_metrics_modes=None,
            full_validation_x_creator=None,
            full_validation_sampler=None,
            full_validation_gt_loader=None,
            full_validation_output_paths=None,
            full_validation_metrics=None,
            full_validation_metrics_modes=None,
            nb_epochs=1,
            nb_subepochs=1,
            nb_runs_per_subepoch=1,
            nb_samples_training=None,
            nb_samples_validation=None,
            sgd_batch_size=1,
            prediction_batch_size=None,
            fixed_learning_rate=None,
            parallel_sample_creation=True,
            path_training_result=None,
            callbacks=None
    ):
        """
        :param training_x_creator: A SampleCreator to generate x samples during training for training.
        :type training_x_creator: SampleCreator

        :param training_y_creator: A SampleCreator to generate y samples during training for training.
        :type training_y_creator: SampleCreator

        :param training_sampler: A Sampler to generate subject ids and coordinate ids during training to generate training x and y samples from.
        :type training_sampler:  Sampler

        :param validation_x_creator: A SampleCreator to generate x samples during training for validation.
        :type validation_x_creator: None
        :type validation_x_creator: SampleCreator

        :param validation_y_creator: A SampleCreator to generate y samples during training for validation.
        :type validation_y_creator: None
        :type validation_y_creator: SampleCreator

        :param validation_sampler: A Sampler to generate subject ids and coordinate ids during training to generate validation x and y samples from.
        :type validation_sampler: None
        :type validation_sampler: Sampler

        :param full_training_x_creator: A SampleCreator to generate x samples during training for full image testing on the training set.
        :type full_training_x_creator: None
        :type full_training_x_creator: SampleCreator

        :param full_training_sampler: This Sampler is used for two purposes when doing full image testing on the training set. First, the underlying ImageLoader is used to have the subject list and to have for each subject the size of the output image that needs to be predicted. Second, if the Sampler has an ROI ImageLoader this is used to avoid excess predictions and mask the prediction.
        :type full_training_sampler: None
        :type full_training_sampler: Sampler

        :param full_training_gt_loader: An ImageLoader to get the ground truth segmentations of the training set for further metric calculations when comparing with the predictions.
        :type full_training_gt_loader: None
        :type full_training_gt_loader: ImageLoader

        :param full_training_output_paths:
        :type full_training_output_paths: None
        :type full_training_output_paths:

        :param full_training_metrics:
        :type full_training_metrics: None
        :type full_training_metrics:

        :param full_training_metrics_modes:
        :type full_training_metrics_modes: None
        :type full_training_metrics_modes:

        :param full_validation_x_creator: A SampleCreator to generate x samples during training for full image testing on the validation set.
        :type full_validation_x_creator: None
        :type full_validation_x_creator: SampleCreator

        :param full_validation_sampler: This Sampler is used for two purposes when doing full image testing on the validation set. First, the underlying ImageLoader is used to have the subject list and to have for each subject the size of the output image that needs to be predicted. Second, if the Sampler has an ROI ImageLoader this is used to avoid excess predictions and mask the prediction.
        :type full_validation_sampler: None
        :type full_validation_sampler: Sampler

        :param full_validation_gt_loader: An ImageLoader to get the ground truth segmentations of the validation set for further metric calculations when comparing with the predictions.
        :type full_validation_gt_loader: None
        :type full_validation_gt_loader: ImageLoader

        :param full_validation_output_paths:
        :type full_validation_output_paths: None
        :type full_validation_output_paths:

        :param full_validation_metrics:
        :type full_validation_metrics: None
        :type full_validation_metrics:

        :param full_validation_metrics_modes:
        :type full_validation_metrics_modes: None
        :type full_validation_metrics_modes:

        :param nb_epochs:
        :type nb_epochs: int

        :param nb_subepochs:
        :type nb_subepochs: int

        :param nb_runs_per_subepoch:
        :type nb_runs_per_subepoch: int

        :param nb_samples_training:
        :type nb_samples_training: None
        :type nb_samples_training: int

        :param nb_samples_validation:
        :type nb_samples_validation: None
        :type nb_samples_validation: int

        :param sgd_batch_size:
        :type sgd_batch_size: int

        :param prediction_batch_size:
        :type prediction_batch_size: None
        :type prediction_batch_size: int

        :param fixed_learning_rate:
        :type fixed_learning_rate: None
        :type fixed_learning_rate:

        :param parallel_sample_creation:
        :type parallel_sample_creation: None
        :type parallel_sample_creation:

        :param path_training_result:
        :type path_training_result: None
        :type path_training_result:

        :param callbacks:
        :type callbacks: None
        :type callbacks:

        :return:
        """

        trainSampleGenerator = sampling.generator.SampleGenerator(training_x_creator, training_y_creator, training_sampler, parallel=parallel_sample_creation)
        trainSampleGenerator.prepare(nb_samples_training, 0)
        if validation_sampler:
            assert validation_x_creator and validation_y_creator
            validationSampleGenerator = sampling.generator.SampleGenerator(validation_x_creator, validation_y_creator, validation_sampler, parallel=parallel_sample_creation)
            validationSampleGenerator.prepare(nb_samples_validation, 0)

        if not callbacks:
            callbacks = []
        keras_callbacks_dict = {callback.__class__.__name__: callback for callback in callbacks}
        if full_validation_metrics or full_validation_output_paths:
            full_validation_x_creator = full_validation_x_creator or validation_x_creator
            full_validation_sampler = full_validation_sampler or validation_sampler
            assert full_validation_x_creator and full_validation_sampler
            fullValidationSetTester = FullImageSetTester(self, "val", full_validation_metrics, full_validation_output_paths, full_validation_x_creator, full_validation_sampler, full_image_set_gt_loader=full_validation_gt_loader, batch_size=prediction_batch_size or sgd_batch_size, period=nb_subepochs * nb_runs_per_subepoch, modes=full_validation_metrics_modes)
            callbacks.insert(0, fullValidationSetTester)
        if full_training_metrics or full_training_output_paths:
            full_training_x_creator = full_training_x_creator or training_x_creator
            full_training_sampler = full_training_sampler or training_sampler
            assert full_training_x_creator and full_training_sampler
            fullTrainingSetTester = FullImageSetTester(self, "train", full_training_metrics, full_training_output_paths, full_training_x_creator, full_training_sampler, full_image_set_gt_loader=full_training_gt_loader, batch_size=prediction_batch_size or sgd_batch_size, period=nb_subepochs * nb_runs_per_subepoch, modes=full_training_metrics_modes)
            callbacks.insert(0, fullTrainingSetTester)
        metricNameChanger = MetricNameChanger()
        callbacks.insert(0, metricNameChanger)
        history = History()
        callbacks.append(history)

        if fixed_learning_rate is not None:
            assert not keras_callbacks_dict.get("LearningRateScheduler", None) and not keras_callbacks_dict.get("ReduceLROnPlateau", None)
            if not isinstance(fixed_learning_rate, (list, tuple)):
                fixed_learning_rate = [fixed_learning_rate] * nb_epochs
            assert len(fixed_learning_rate) == nb_epochs
            for i, lr in enumerate(fixed_learning_rate):
                if not isinstance(lr, (list, tuple)):
                    fixed_learning_rate[i] = [lr] * nb_subepochs
                assert len(fixed_learning_rate[i]) == nb_subepochs

            def fixed_learning_rate_scheduler(fixed_learning_rate_schedule=[subepoch_lr for epoch_lr in fixed_learning_rate for subepoch_lr in epoch_lr]):
                def scheduler(keras_epoch, current_learning_rate):
                    prev_lr = fixed_learning_rate_schedule[keras_epoch - 1] if keras_epoch > 0 else None
                    new_lr = fixed_learning_rate_schedule[keras_epoch]
                    if prev_lr != new_lr:
                        print("Setting learning rate from {} to {}.".format(prev_lr, new_lr))
                    return new_lr
                return scheduler
            learningRateScheduler = LearningRateScheduler(fixed_learning_rate_scheduler(), verbose=False)
            callbacks.append(learningRateScheduler)

        total_number_of_keras_epochs_so_far = 0
        total_number_of_keras_epochs = nb_epochs * nb_subepochs * nb_runs_per_subepoch
        for epoch_i in range(nb_epochs):
            for subepoch_i in range(nb_subepochs):
                print("=== Total # epochs: {}({})/{}({}) == Total # Keras epochs: {}/{} === @ LR: {:.7f} ===".format(epoch_i, subepoch_i, nb_epochs, nb_subepochs, total_number_of_keras_epochs_so_far, total_number_of_keras_epochs, K.eval(self.model.optimizer.lr)))
                
                x, y = trainSampleGenerator.get()
                if not (epoch_i + 1 == nb_epochs and subepoch_i + 1 == nb_subepochs):
                    trainSampleGenerator.prepare(nb_samples_training, total_number_of_keras_epochs_so_far)
                if validation_sampler:
                    x_val, y_val = validationSampleGenerator.get()
                    if not (epoch_i + 1 == nb_epochs and subepoch_i + 1 == nb_subepochs):
                        validationSampleGenerator.prepare(nb_samples_validation, total_number_of_keras_epochs_so_far)
                if self.nb_classes_in_multiclass:
                    for layer_name, nb_classes in self.nb_classes_in_multiclass.items():
                        i = self.model.output_names.index(layer_name)
                        if nb_classes is not None and nb_classes > 1 and y[i].shape[4] == 1:
                            y[i] = to_categorical(y[i], num_classes=nb_classes)
                            if nb_samples_validation != 0:
                                y_val[i] = to_categorical(y_val[i], num_classes=nb_classes)
                self.model.fit(x, y, batch_size=sgd_batch_size, epochs=total_number_of_keras_epochs_so_far + nb_runs_per_subepoch, initial_epoch=total_number_of_keras_epochs_so_far, shuffle=True, callbacks=callbacks, validation_data=[x_val, y_val] if validation_sampler else None)
                training_result = {"training_history": history.history, "full_training_history": fullTrainingSetTester.full_image_set_history if full_training_metrics or full_training_output_paths else None, "full_validation_history": fullValidationSetTester.full_image_set_history if full_validation_metrics or full_validation_output_paths else None}
                if path_training_result:
                    with open(path_training_result, "wb") as f:
                        pickle.dump(training_result, f)
                total_number_of_keras_epochs_so_far += nb_runs_per_subepoch
            if keras_callbacks_dict.get("EarlyStopping", None) and keras_callbacks_dict.get("EarlyStopping", None).stopped_epoch:
                break
                         
        return training_result

    def predict(self, x_creator, sampler, subject_id, out_path=None, verbose=True, batch_size=1, auto_recalibration=True, stack_recalibrated=False, output_layer_idx=[0], include_output_layer_name_in_out_path=False, auto_return_first_only=True):
        
        start_time = time.time()

        input_shape = self.model.input_shape
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        main_input_shape = input_shape[0][1:-1]

        layers = self.model.layers
        layers_scales = get_layers_scales(self.model)
        
        output_layers = [self.model.get_layer(name) for name in self.model.output_names]
        output_layers_scales = [layers_scales[layers.index(layer)] for layer in output_layers]

        nb_requested_output_layers = len(output_layer_idx)
        requested_output_layers = [output_layers[i] for i in output_layer_idx]
        requested_output_layers_scales = [output_layers_scales[i] for i in output_layer_idx]
        requested_output_layers_output_sizes = [[int(o_s) for o_s in layer.output_shape[1:-1]] for layer in requested_output_layers]
        requested_output_layers_nb_features = [int(layer.output_shape[-1]) for layer in requested_output_layers]

        numerators = [[scale_part.numerator for scale_part in scale] for scale in requested_output_layers_scales]
        denominators = [[scale_part.denominator for scale_part in scale] for scale in requested_output_layers_scales]
        lcf = [1, 1, 1]
        for denominator in [(1, 1, 1), *denominators]:
            lcf = [int(lcf_part * denominator_part / math.gcd(lcf_part, denominator_part)) for lcf_part, denominator_part in zip(lcf, denominator)]
        gcd = [numerator_part * int(lcf_part / denominator_part) for numerator_part, lcf_part, denominator_part in zip(numerators[0], lcf, denominators[0])]
        for numerator, denominator in zip([(1, 1, 1), *numerators], [(1, 1, 1), *denominators]):
            gcd = [math.gcd(gcd_part, numerator_part * int(lcf_part / denominator_part)) for gcd_part, numerator_part, lcf_part, denominator_part in zip(gcd, numerator, lcf, denominator)]
        I_fill_resolution = [gcd_part / lcf_part for gcd_part, lcf_part in zip(gcd, lcf)]
        
        requested_output_layers_relative_resolutions = [[int(scale_part / I_fill_resolution_part) for scale_part, I_fill_resolution_part in zip(scale, I_fill_resolution)] for scale in requested_output_layers_scales]
        requested_output_layers_output_sizes_at_I_fill_resolution = [[requested_output_layers_relative_resolution_part * (requested_output_layers_output_size_part - 1) + 1 for requested_output_layers_relative_resolution_part, requested_output_layers_output_size_part in zip(requested_output_layers_relative_resolution, requested_output_layers_output_size)] for requested_output_layers_relative_resolution, requested_output_layers_output_size in zip(requested_output_layers_relative_resolutions, requested_output_layers_output_sizes)]

        I, affine, header = sampler.get_base_image(subject_id)
        roi = sampler.get_roi_image(subject_id)
        if roi is not None:
            roi = roi[0]
        
        I_size = I.shape[:-1]
        I_fill_size = [int((I_size_part - 1) / I_fill_resolution_part) + 1 for I_size_part, I_fill_resolution_part in zip(I_size, I_fill_resolution)]
        outputs_fill = [np.full([*I_fill_size, nb_features], np.nan) for nb_features in requested_output_layers_nb_features]

        sampling_segment_size_at_I_resolution = [I_size_part for I_size_part in I_size]
        if not self.center_sampling:
            for requested_output_layers_output_size, requested_output_layers_scale in zip(requested_output_layers_output_sizes, requested_output_layers_scales):
                for i, (I_size_part, requested_output_layers_output_size_part, requested_output_layers_scale_part) in enumerate(zip(I_size, requested_output_layers_output_size, requested_output_layers_scale)):
                    if requested_output_layers_output_size_part is None:
                        sampling_segment_size_at_I_resolution[i] = I_size_part
                    else:
                        requested_output_layers_output_size_part_at_I_resolution = requested_output_layers_scale_part * (requested_output_layers_output_size_part - 1) + 1
                        if requested_output_layers_output_size_part_at_I_resolution < sampling_segment_size_at_I_resolution[i]:
                            sampling_segment_size_at_I_resolution[i] = math.floor(requested_output_layers_output_size_part_at_I_resolution)

        x_starts_at_I_resolution = range(0, I_size[0], sampling_segment_size_at_I_resolution[0]) if not I_size[0] % sampling_segment_size_at_I_resolution[0] else [*range(0, I_size[0], sampling_segment_size_at_I_resolution[0]), I_size[0] - sampling_segment_size_at_I_resolution[0]]
        y_starts_at_I_resolution = range(0, I_size[1], sampling_segment_size_at_I_resolution[1]) if not I_size[1] % sampling_segment_size_at_I_resolution[1] else [*range(0, I_size[1], sampling_segment_size_at_I_resolution[1]), I_size[1] - sampling_segment_size_at_I_resolution[1]]
        z_starts_at_I_resolution = range(0, I_size[2], sampling_segment_size_at_I_resolution[2]) if not I_size[2] % sampling_segment_size_at_I_resolution[2] else [*range(0, I_size[2], sampling_segment_size_at_I_resolution[2]), I_size[2] - sampling_segment_size_at_I_resolution[2]]
        total_nb_sampling_coordinates = len(x_starts_at_I_resolution) * len(y_starts_at_I_resolution) * len(z_starts_at_I_resolution)

        samples = []
        valid_slices_in_I_fill = []
        valid_slices_in_result = []
        nb_sampling_coordinates = 0
        for x in x_starts_at_I_resolution:
            sampling_x_center_at_I_resolution = math.ceil(x + (sampling_segment_size_at_I_resolution[0] - 1) / 2)  # sampling x coordinate
            x_center_at_I_resolution = sampling_x_center_at_I_resolution if main_input_shape[0] % 2 else sampling_x_center_at_I_resolution - 0.5  # real spacial x center coordinate
            x_center_at_I_fill_resolution = x_center_at_I_resolution / I_fill_resolution[0]
            x_starts_at_I_fill = [x_center_at_I_fill_resolution - (requested_output_layers_output_size_at_I_fill_resolution[0] - 1) / 2 for requested_output_layers_output_size_at_I_fill_resolution in requested_output_layers_output_sizes_at_I_fill_resolution]
            x_ends_at_I_fill = [x_center_at_I_fill_resolution + (requested_output_layers_output_size_at_I_fill_resolution[0] - 1) / 2 for requested_output_layers_output_size_at_I_fill_resolution in requested_output_layers_output_sizes_at_I_fill_resolution]
            undershoot_x_in_I_fill = [max(0, -1 * x_starts_at_I_fill[i]) for i in range(nb_requested_output_layers)]
            overshoot_x_in_I_fill = [max(0, x_ends_at_I_fill[i] - (I_fill_size[0] - 1)) for i in range(nb_requested_output_layers)]
            start_x_slice_in_I_fill = [x_starts_at_I_fill[i] + math.ceil(undershoot_x_in_I_fill[i] / requested_output_layers_relative_resolutions[i][0]) * requested_output_layers_relative_resolutions[i][0] for i in range(nb_requested_output_layers)]
            end_x_slice_in_I_fill = [x_ends_at_I_fill[i] - math.ceil(overshoot_x_in_I_fill[i] / requested_output_layers_relative_resolutions[i][0]) * requested_output_layers_relative_resolutions[i][0] for i in range(nb_requested_output_layers)]
            for y in y_starts_at_I_resolution:
                sampling_y_center_at_I_resolution = math.ceil(y + (sampling_segment_size_at_I_resolution[1] - 1) / 2)  # sampling y coordinate
                y_center_at_I_resolution = sampling_y_center_at_I_resolution if main_input_shape[1] % 2 else sampling_y_center_at_I_resolution - 0.5  # real spacial y center coordinate
                y_center_at_I_fill_resolution = y_center_at_I_resolution / I_fill_resolution[1]
                y_starts_at_I_fill = [y_center_at_I_fill_resolution - (requested_output_layers_output_size_at_I_fill_resolution[1] - 1) / 2 for requested_output_layers_output_size_at_I_fill_resolution in requested_output_layers_output_sizes_at_I_fill_resolution]
                y_ends_at_I_fill = [y_center_at_I_fill_resolution + (requested_output_layers_output_size_at_I_fill_resolution[1] - 1) / 2 for requested_output_layers_output_size_at_I_fill_resolution in requested_output_layers_output_sizes_at_I_fill_resolution]
                undershoot_y_in_I_fill = [max(0, -1 * y_starts_at_I_fill[i]) for i in range(nb_requested_output_layers)]
                overshoot_y_in_I_fill = [max(0, y_ends_at_I_fill[i] - (I_fill_size[1] - 1)) for i in range(nb_requested_output_layers)]
                start_y_slice_in_I_fill = [y_starts_at_I_fill[i] + math.ceil(undershoot_y_in_I_fill[i] / requested_output_layers_relative_resolutions[i][1]) * requested_output_layers_relative_resolutions[i][1] for i in range(nb_requested_output_layers)]
                end_y_slice_in_I_fill = [y_ends_at_I_fill[i] - math.ceil(overshoot_y_in_I_fill[i] / requested_output_layers_relative_resolutions[i][1]) * requested_output_layers_relative_resolutions[i][1] for i in range(nb_requested_output_layers)]
                for z in z_starts_at_I_resolution:
                    nb_sampling_coordinates += 1
                    sampling_z_center_at_I_resolution = math.ceil(z + (sampling_segment_size_at_I_resolution[2] - 1) / 2)  # sampling z coordinate
                    z_center_at_I_resolution = sampling_z_center_at_I_resolution if main_input_shape[2] % 2 else sampling_z_center_at_I_resolution - 0.5  # real spacial z center coordinate
                    z_center_at_I_fill_resolution = z_center_at_I_resolution / I_fill_resolution[2]
                    z_starts_at_I_fill = [z_center_at_I_fill_resolution - (requested_output_layers_output_size_at_I_fill_resolution[2] - 1) / 2 for requested_output_layers_output_size_at_I_fill_resolution in requested_output_layers_output_sizes_at_I_fill_resolution]
                    z_ends_at_I_fill = [z_center_at_I_fill_resolution + (requested_output_layers_output_size_at_I_fill_resolution[2] - 1) / 2 for requested_output_layers_output_size_at_I_fill_resolution in requested_output_layers_output_sizes_at_I_fill_resolution]
                    undershoot_z_in_I_fill = [max(0, -1 * z_starts_at_I_fill[i]) for i in range(nb_requested_output_layers)]
                    overshoot_z_in_I_fill = [max(0, z_ends_at_I_fill[i] - (I_fill_size[2] - 1)) for i in range(nb_requested_output_layers)]
                    start_z_slice_in_I_fill = [z_starts_at_I_fill[i] + math.ceil(undershoot_z_in_I_fill[i] / requested_output_layers_relative_resolutions[i][2]) * requested_output_layers_relative_resolutions[i][2] for i in range(nb_requested_output_layers)]
                    end_z_slice_in_I_fill = [z_ends_at_I_fill[i] - math.ceil(overshoot_z_in_I_fill[i] / requested_output_layers_relative_resolutions[i][2]) * requested_output_layers_relative_resolutions[i][2] for i in range(nb_requested_output_layers)]
                    if roi is None or np.sum(roi[x:x + sampling_segment_size_at_I_resolution[0], y:y + sampling_segment_size_at_I_resolution[1], z:z + sampling_segment_size_at_I_resolution[2]]) > 0:
                        samples.append(x_creator.get_sample(subject_id, (sampling_x_center_at_I_resolution, sampling_y_center_at_I_resolution, sampling_z_center_at_I_resolution), epoch_identifier=1))
                        valid_slices_in_I_fill_, valid_slices_in_result_ = [], []
                        for i, requested_output_layers_relative_resolution in enumerate(requested_output_layers_relative_resolutions):
                            x_slice_in_I_fill = slice(round(start_x_slice_in_I_fill[i] - 1e-7), round(end_x_slice_in_I_fill[i] + 1 - 1e-7), requested_output_layers_relative_resolution[0])
                            y_slice_in_I_fill = slice(round(start_y_slice_in_I_fill[i] - 1e-7), round(end_y_slice_in_I_fill[i] + 1 - 1e-7), requested_output_layers_relative_resolution[1])
                            z_slice_in_I_fill = slice(round(start_z_slice_in_I_fill[i] - 1e-7), round(end_z_slice_in_I_fill[i] + 1 - 1e-7), requested_output_layers_relative_resolution[2])
                            x_slice_in_result = slice(math.ceil(undershoot_x_in_I_fill[i] / requested_output_layers_relative_resolution[0]), -1 * math.ceil(overshoot_x_in_I_fill[i] / requested_output_layers_relative_resolution[0]) or None)
                            y_slice_in_result = slice(math.ceil(undershoot_y_in_I_fill[i] / requested_output_layers_relative_resolution[1]), -1 * math.ceil(overshoot_y_in_I_fill[i] / requested_output_layers_relative_resolution[1]) or None)
                            z_slice_in_result = slice(math.ceil(undershoot_z_in_I_fill[i] / requested_output_layers_relative_resolution[2]), -1 * math.ceil(overshoot_z_in_I_fill[i] / requested_output_layers_relative_resolution[2]) or None)
                            valid_slices_in_I_fill_.append((x_slice_in_I_fill, y_slice_in_I_fill, z_slice_in_I_fill))
                            valid_slices_in_result_.append((x_slice_in_result, y_slice_in_result, z_slice_in_result))
                        valid_slices_in_I_fill.append(valid_slices_in_I_fill_)
                        valid_slices_in_result.append(valid_slices_in_result_)
                    if len(samples) == batch_size or (len(samples) > 0 and nb_sampling_coordinates == total_nb_sampling_coordinates):
                        X = [np.array([sample[sample_part] for sample in samples]) for sample_part, _ in enumerate(samples[0])]
                        Y = self.model.predict(X)
                        if not isinstance(Y, list):
                            Y = [Y]
                        valid_slices_in_I_fill = list(zip(*valid_slices_in_I_fill))
                        valid_slices_in_result = list(zip(*valid_slices_in_result))
                        for o, y_batch in enumerate(Y):
                            if o in output_layer_idx:
                                idx = output_layer_idx.index(o)
                                for valid_slice_in_I_fill, valid_slice_in_result, y_b in zip(valid_slices_in_I_fill[idx], valid_slices_in_result[idx], y_batch):
                                    outputs_fill[idx][valid_slice_in_I_fill] = y_b[valid_slice_in_result]
                        samples = []
                        valid_slices_in_I_fill = []
                        valid_slices_in_result = []
        
        outputs_fill = [output[tuple(ndimage.distance_transform_edt(np.isnan(output), return_distances=False, return_indices=True))] for output in outputs_fill]
       
        if roi is not None: #Apply ROI
            roi_fill = ndimage.interpolation.zoom(roi, zoom=[I_fill_size_part / I_size_part for I_fill_size_part, I_size_part in zip(I_fill_size, I_size)] + [1], order=0)
            for i, output in enumerate(outputs_fill):
                outputs_fill[i] = output * roi_fill
        
        outputs = [output[tuple([slice(0, None, relative_resolution_part) for relative_resolution_part in relative_resolution])] for output, relative_resolution in zip(outputs_fill, requested_output_layers_relative_resolutions)]
        outputs = [output if not self.classification or not self.classification[requested_output_layers[i].name] else np.mean(output, axis=(0, 1, 2), keepdims=True) for i, output in enumerate(outputs)]

        if out_path:
            if not isinstance(out_path, list):
                assert isinstance(out_path, str)
                out_path = [out_path]
            assert len(out_path) == 1 or len(out_path) == len(output_layer_idx)
            if include_output_layer_name_in_out_path:
                if len(output_layer_idx) > len(out_path):
                    out_path = out_path * len(output_layer_idx)
                out_path = [os.path.join(os.path.split(path)[0], "{}_{}".format(layer.name, os.path.split(path)[1])) if path is not None else path for layer, path in zip(requested_output_layers, out_path)]
            for i, (path, output, scale, nb_features_output) in enumerate(zip(out_path, outputs, requested_output_layers_scales, requested_output_layers_nb_features)):
                if path:
                    out_dir_path = os.path.dirname(path)
                    if not os.path.exists(out_dir_path):
                        os.makedirs(out_dir_path)
                    output = output[..., 0 if output.shape[-1] == 1 else slice(None)]  # Make (and save) it (as) a 3D image if possible
                    outputs[i] = output  # hope to be removed in the future
                    affine_new = affine.copy()
                    affine_new[:3, :3] *= [float(s) for s in scale]
                    img_out = nib.Nifti1Image(output, affine_new)
                    nib.save(img_out, path)

        if verbose:
            print("Predicted subject #{} in {:.2f} s".format(subject_id, time.time()-start_time))
                  
        if auto_return_first_only:
            return outputs[0]
        else:
            return outputs
        

    @staticmethod
    def load(path, custom_objects=None, compile=True):
        from keras.models import load_model
        # Load deepvoxnet, without the model
        with open(path + ".DVN", "rb") as f:
            deepvoxnet = pickle.load(f)
        deepvoxnet.model = load_model(path, custom_objects=custom_objects, compile=compile)
        return deepvoxnet
        
    
    def save(self, path, save_weights_only=False):
        # Save keras model
        if save_weights_only:
            self.model.save_weights(path)
        else:
            self.model.save(path)
        # Save deepvoxnet, without the model
        model = self.model
        self.model = None
        with open(path + ".DVN", "wb") as f:
            pickle.dump(self, f)
        self.model = model
