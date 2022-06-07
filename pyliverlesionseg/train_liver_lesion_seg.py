import argparse
import os
import pickle
import numpy as np
import nibabel as nib
import pyliverlesionseg.sampling as sampling
import pyliverlesionseg.components.loss as loss
import pyliverlesionseg.components.metrics as metrics
import keras.backend as K
from shutil import copyfile
from functools import partial, update_wrapper
from pyliverlesionseg.components.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from pyliverlesionseg.general import DeepVoxNet
from pyliverlesionseg.architectures.unet_generalized import create_unet_like_model
from keras.optimizers import SGD, Adam
from keras.models import load_model

def run(parameters):

    """
    
    This script is used to train a CNN model for liver or lesion segmentation. The CNN structure is U-net.
    
    """
    
    # extract all parameters from the input dictionary
    data_path = parameters["data_path"]
    data = parameters["data"]
    mode = parameters["mode"]
    subjects = parameters["subjects"]
    training_indices = parameters["training_indices"]
    validation_indices = parameters["validation_indices"]
    inputs = parameters["inputs"]
    outputs = parameters["outputs"]
    run_folder_name = parameters["run_folder_name"]
    network_architecture_id = parameters["network_architecture_id"]
    segment_size = parameters["segment_size"]
    center_sampling = parameters["center_sampling"]
    sampling_roi = parameters["sampling_roi"]
    objective_function = parameters["objective_function"]
    objective_function_weights = parameters["objective_function_weights"]
    weight_decay = parameters["weight_decay"]
    weighted_loss = parameters["weighted_loss"]
    clip_input = parameters["clip_input"]
    normalize_input = parameters["normalize_input"]
    data_augmentation = parameters["data_augmentation"]
    optimizer = parameters["optimizer"]
    fixed_learning_rate = parameters["fixed_learning_rate"]
    reduce_learning_rate_on_plateau = parameters["reduce_learning_rate_on_plateau"]
    early_stopping = parameters["early_stopping"]
    sgd_batch_size = parameters["sgd_batch_size"]
    prediction_batch_size = parameters["prediction_batch_size"]
    nb_samples_training = parameters["nb_samples_training"]
    nb_samples_validation = parameters["nb_samples_validation"]
    max_number_of_subjects_used_for_training = parameters["max_number_of_subjects_used_for_training"]
    max_number_of_subjects_used_for_validation = parameters["max_number_of_subjects_used_for_validation"]
    nb_epochs = parameters["nb_epochs"]
    nb_subepochs = parameters["nb_subepochs"]
    nb_runs_per_subepoch = parameters["nb_runs_per_subepoch"]
    parallel_sample_creation = parameters["parallel_sample_creation"]
    full_training_testing = parameters["full_training_testing"]
    full_validation_testing = parameters["full_validation_testing"]
    full_image_testing_appendix = parameters["full_image_testing_appendix"]
    model_paths = parameters["model_paths"]
    load_weights_only = parameters["load_weights_only"]
    save_weights_only = parameters["save_weights_only"]
    save_best_model_only = parameters["save_best_model_only"]
        
    #########################
    # Do some sanity checks #
    #########################
    if data == "Training":
        assert outputs
    else:
        assert data == "Testing" and mode == "Testing"
        assert not outputs
    #if model_paths is not None:
    #    assert len(model_paths) == len(fold_numbers)
    if reduce_learning_rate_on_plateau:
        assert not isinstance(fixed_learning_rate, (tuple, list))
        initial_learning_rate = fixed_learning_rate
        fixed_learning_rate = None
    else:
        if not isinstance(fixed_learning_rate, (list, tuple)):
            parameters["fixed_learning_rate"] = fixed_learning_rate = [fixed_learning_rate] * nb_epochs
        assert len(fixed_learning_rate) == nb_epochs
        for i, lr in enumerate(fixed_learning_rate):
            if not isinstance(lr, (list, tuple)):
                fixed_learning_rate[i] = [lr] * nb_subepochs
            assert len(fixed_learning_rate[i]) == nb_subepochs
        initial_learning_rate = fixed_learning_rate[0][0]
    if not isinstance(objective_function, (tuple, list)):
        parameters["objective_function"] = objective_function = [objective_function]
    if objective_function_weights is not None:
        assert len(objective_function_weights) == len(objective_function)
        assert np.sum(objective_function_weights) == 1
    else:
        parameters["objective_function_weights"] = objective_function_weights = [1 / len(objective_function) for _ in objective_function]
    if mode == "Training":
        assert full_image_testing_appendix is None

    ######################################
    # Define input and output data paths #
    ######################################
    input_paths = [[os.path.join(data_path, data, s, "{}.nii".format(input)) for input in inputs] for s in subjects]
    output_paths = [[os.path.join(data_path, data, s, "{}.nii".format(o)) for o in outputs] for s in subjects] if data == "Training" else None

    ####################################################
    # Define sampling, input mask and output mask ROIs #
    ####################################################
    sampling_rois = []
    input_masks = []
    output_masks = []
    for i, (input_path, output_path) in enumerate(zip(input_paths, output_paths)):
        if sampling_roi:
            if data == "Training":
                sampling_roi_ = [np.ones_like(nib.load(input_path[0]).get_data()) * (nib.load(output_path[0]).get_data() == 0), nib.load(output_path[0]).get_data() > 0]
            else:
                sampling_roi_ = np.ones_like(nib.load(input_path[0]).get_data())
        else:
            sampling_roi_ = np.ones_like(nib.load(input_path[0]).get_data())
        sampling_rois.append(sampling_roi_)
        input_masks.append(np.ones_like(nib.load(input_path[0]).get_data()))
        output_masks.append(np.ones_like(nib.load(input_path[0]).get_data()))

    ############################
    # Define the run directory #
    ############################
    run_dir = os.path.join(data_path, data, run_folder_name)
    if not os.path.isdir(run_dir):
        assert not (data == "Training" and mode == "Testing")
        print("This is a new run: {}".format(run_folder_name))
        os.makedirs(run_dir)
        for s in subjects:
            if data == "Training":
                os.makedirs(os.path.join(run_dir, s, "Training"))
                os.makedirs(os.path.join(run_dir, s, "Validation"))
            else:
                os.makedirs(os.path.join(run_dir, s, "Testing"))
    else:
        print("This is an existing run: {}".format(run_folder_name))

    ###################################
    # Define the experiment directory #
    ###################################
    if data == "Training":
        experiment_dir = os.path.join(run_dir, "Experiments")
        if not os.path.isdir(experiment_dir):
            assert mode == "Training"
            print("This is a new experiment")
            os.makedirs(experiment_dir)
            with open(os.path.join(experiment_dir, "parameters.pkl"), "wb") as f:
                pickle.dump(parameters, f)
            with open(os.path.join(experiment_dir, "parameters.txt"), "w") as f:
                f.write(str(parameters))
            round_number = 0
        else:
            print("This is an existing experiment")
            with open(os.path.join(experiment_dir, "parameters.pkl"), "rb") as f:
                existing_parameters = pickle.load(f)
                existing_parameters["data_path"] = parameters["data_path"]
                if mode == "Testing":
                    existing_parameters["mode"] = parameters["mode"]
                    existing_parameters["model_paths"] = parameters["model_paths"]
                    existing_parameters["full_image_testing_appendix"] = parameters["full_image_testing_appendix"]
                    existing_parameters["full_training_testing"] = parameters["full_training_testing"] #added by Xikai
                    existing_parameters["full_validation_testing"] = parameters["full_validation_testing"]  #added by Xikai
                else:
                    existing_parameters["model_paths"] = parameters["model_paths"]
                    existing_parameters["fixed_learning_rate"] = parameters["fixed_learning_rate"]
                    existing_parameters["nb_epochs"] = parameters["nb_epochs"]
                    existing_parameters["nb_subepochs"] = parameters["nb_subepochs"]
            existing_rounds = [r for r in os.listdir(experiment_dir) if r.startswith("Round_") and os.path.isdir(os.path.join(experiment_dir, r))]
            round_number = len(existing_rounds)
            existing_folds = [int(fold.split("_")[-1]) for fold in os.listdir(os.path.join(experiment_dir, existing_rounds[-1])) if fold.startswith("Fold_") and os.path.isdir(os.path.join(experiment_dir, existing_rounds[-1], fold))]

        print("Starting Round: {}".format(round_number))
        round_dir = os.path.join(experiment_dir, "Round_{}".format(round_number))
        if not os.path.isdir(round_dir):
            assert mode == "Training"
            os.makedirs(round_dir)
        copyfile(os.path.realpath(__file__), os.path.join(round_dir, "script.py"))
    else:
        experiment_dir = os.path.join(data_path, "Training", run_folder_name, "Experiments")
        if not os.path.isdir(experiment_dir):
            print("This is a new experiment (we can therefore not find any model for testing)")
        else:
            print("This is an existing experiment")
            existing_rounds = [r for r in os.listdir(experiment_dir) if r.startswith("Round_") and os.path.isdir(os.path.join(experiment_dir, r))]
            round_number = len(existing_rounds) - 1
            round_dir = os.path.join(experiment_dir, "Round_{}".format(round_number))
    
    model_path = os.path.join(experiment_dir, "Round_{}.hdf5".format(round_number))
    
    #########################################################################################################################################
    # Define which subjects are used for training/validation/testing and the corresponding full_training/full_validation/full_testing paths #
    #########################################################################################################################################
    if data == "Training":
        logs_dir = os.path.join(round_dir, "Logs")
        if mode == "Training":
            assert not os.path.isdir(logs_dir)
            os.makedirs(logs_dir)
        path_training_result = os.path.join(round_dir, "training_result.pkl")
        training_subjects = [subjects[i] for i in training_indices]
        validation_subjects = [subjects[i] for i in validation_indices]
        full_training_output_paths = [os.path.join(run_dir, s, "Training", "Round_{}{}.nii".format(round_number, "" if full_image_testing_appendix is None else "_{}".format(full_image_testing_appendix))) for s in training_subjects]
        full_validation_output_paths = [os.path.join(run_dir, s, "Validation", "Round_{}{}.nii".format(round_number, "" if full_image_testing_appendix is None else "_{}".format(full_image_testing_appendix))) for s in validation_subjects]
    else:
        assert os.path.isdir(round_dir)
        testing_subjects = subjects
        full_testing_output_paths = [os.path.join(run_dir, s, "Testing", "Round_{}{}.nii".format(round_number, "" if full_image_testing_appendix is None else "_{}".format(full_image_testing_appendix))) for s in testing_subjects]
    
    ######################################################################################
    # We choose the correct network according to the number specified and the info given #
    ######################################################################################
    if load_weights_only or (data == "Training" and not model_paths):
        # network structure for liver segmentation
        if network_architecture_id == 1:
            model = create_unet_like_model(
                number_input_features=len(inputs),
                subsample_factors_per_pathway=[
                    (1, 1, 1),
                    (3, 3, 3),
                    (9, 9, 9),
                    (27, 27, 27)
                ],
                kernel_sizes_per_pathway=[
                    [[(3, 3, 1), (3, 3, 3)], [(3, 3, 3), (3, 3, 1)]],
                    [[(3, 3, 1), (3, 3, 3)], [(3, 3, 3), (3, 3, 1)]],
                    [[(3, 3, 1), (3, 3, 3)], [(3, 3, 3), (3, 3, 1)]],
                    [[(3, 3, 3), (3, 3, 3)], []]
                ],
                number_features_per_pathway=[
                    [[20, 40], [40, 20]],
                    [[40, 80], [80, 40]],
                    [[80, 160], [160, 80]],
                    [[160, 160], []]
                ],
                output_size=segment_size,
                padding='same',
                upsampling='linear',
                mask_output=True,
                l2_reg=weight_decay or 0.0
            )
        # network structure for liver lesion segmentation
        elif network_architecture_id == 2:
            model = create_unet_like_model(
                number_input_features=len(inputs),
                subsample_factors_per_pathway=[
                    (1, 1, 1),
                    (2, 2, 2),
                    (4, 4, 4),
                    (8, 8, 8)
                    ],
                kernel_sizes_per_pathway=[
                    [[(3, 3, 1), (3, 3, 3)], [(3, 3, 3), (3, 3, 1)]],
                    [[(3, 3, 1), (3, 3, 3)], [(3, 3, 3), (3, 3, 1)]],
                    [[(3, 3, 1), (3, 3, 3)], [(3, 3, 3), (3, 3, 1)]],
                    [[(3, 3, 1), (3, 3, 3)], []]
                ],
                number_features_per_pathway=[
                    [[20, 40], [40, 20]],
                    [[40, 80], [80, 40]],
                    [[80, 160], [160, 80]],
                    [[160, 160], []]
                ],
                output_size=segment_size,
                padding='valid',
                upsampling='copy',
                mask_output=True,
                residual_connections=True,
                l2_reg=weight_decay or 0.0
            )
        if model_paths:
            model.load_weights(model_paths.pop(0))
        elif mode == "Testing":
            model.load_weights(model_path)
    else:
        if model_paths:
            model = load_model(model_paths.pop(0), compile=False)
    
    ######################################################
    # If "Training": Make the correct objective function #
    ######################################################
    if data == "Training" and mode == "Training":
        objective_functions = []
        for objective_function_ in objective_function:
            if "cross-entropy" == objective_function_:
                def wbc(y_true, y_pred):
                    return loss.binary_crossentropy(y_true, y_pred, weights=np.asarray(solutions_weighted_loss if weighted_loss else [1, 1], dtype=np.float32), focal=True if "focal" in objective_function_ else False, sampling_correction=False)
                objective_functions.append(wbc)
            elif "dice" == objective_function_:
                def dcl(y_true, y_pred):
                    return loss.binary_dice_loss(y_true, y_pred, sampling_correction=False)
                objective_functions.append(dcl)
            elif "jaccard" == objective_function_:
                def jcl(y_true, y_pred):
                    return loss.binary_jaccard_loss(y_true, y_pred, sampling_correction=False)
                objective_functions.append(jcl)
            elif "lovasz" == objective_function_:
                def lovasz(y_true, y_pred):
                    return loss.binary_lovasz_softmax_loss(y_true, y_pred, sampling_correction=False)
                objective_functions.append(lovasz)
            elif "tversky" == objective_function_:
                def tversky(y_true, y_pred):
                    return loss.binary_tversky_loss(y_true, y_pred, alpha=weighted_loss[0], beta=weighted_loss[1], sampling_correction=False)
                objective_functions.append(tversky)
        def objective(y_true, y_pred):
            total = 0
            for objective_, objective_function_weight in zip(objective_functions, objective_function_weights):
                total += objective_function_weight * objective_(y_true, y_pred)
            return total
    
    ###########################################
    # Define the sample creators and samplers #
    ###########################################
    # Fixed image transformers
    fixedIntensityClipper = sampling.imagetransformer.FixedIntensityTransformer(
        min_clip=-float('inf') if clip_input else -float('inf'),
        max_clip=float('inf') if clip_input else float('inf')
    )
    fixedIntensityTransformer = sampling.imagetransformer.FixedIntensityTransformer(
        shift=[-0] if normalize_input else 0,
        scale=[1 / 1] if normalize_input else 1,
    )
    if data == "Training":
        # Image transformers
        gaussianNoise                           = sampling.imagetransformer.GaussianNoiseTransformer(sampling.Randomness.PerSubject, [(0, 0.01)], transform_probability=data_augmentation)
        # Sample transformers
        sampleFlipper                           = sampling.sampletransformer.Flipper(sampling.Randomness.PerSample, (0.5, 0, 0) if data_augmentation else (0, 0, 0))
        # Training data loaders
        training_inputLoader                    = sampling.ImageLoader([input_paths[i] for i in training_indices], image_transformers=[gaussianNoise, fixedIntensityClipper, fixedIntensityTransformer])
        training_outputLoader                   = sampling.ImageLoader([output_paths[i] for i in training_indices])
        training_inputMaskLoader                = sampling.ImageLoader([input_masks[i] for i in training_indices])
        training_samplingRoiLoader              = sampling.ImageLoader([sampling_rois[i] for i in training_indices])
        # Validation data loaders
        validation_inputLoader                  = sampling.ImageLoader([input_paths[i] for i in validation_indices], image_transformers=[gaussianNoise, fixedIntensityClipper, fixedIntensityTransformer])
        validation_outputLoader                 = sampling.ImageLoader([output_paths[i] for i in validation_indices])
        validation_inputMaskLoader              = sampling.ImageLoader([input_masks[i] for i in validation_indices])
        validation_samplingRoiLoader            = sampling.ImageLoader([sampling_rois[i] for i in validation_indices])
        # Full training data loaders
        full_training_inputLoader               = sampling.ImageLoader([input_paths[i] for i in training_indices], image_transformers=[fixedIntensityClipper, fixedIntensityTransformer])
        full_training_outputLoader              = sampling.ImageLoader([output_paths[i] for i in training_indices])
        full_training_inputMaskLoader           = sampling.ImageLoader([input_masks[i] for i in training_indices])
        full_training_outputMaskLoader          = sampling.ImageLoader([output_masks[i] for i in training_indices])
        # Full validation data loaders
        full_validation_inputLoader             = sampling.ImageLoader([input_paths[i] for i in validation_indices], image_transformers=[fixedIntensityClipper, fixedIntensityTransformer])
        full_validation_outputLoader            = sampling.ImageLoader([output_paths[i] for i in validation_indices])
        full_validation_inputMaskLoader         = sampling.ImageLoader([input_masks[i] for i in validation_indices])
        full_validation_outputMaskLoader        = sampling.ImageLoader([output_masks[i] for i in validation_indices])
        # X and Y creators
        if network_architecture_id in [1, 2]:
            training_x_creator          = sampling.Concat([
                sampling.TransformSamples(sampling.ExtractSegment3D2(training_inputLoader, model.input_shape[0][1:4]), [sampleFlipper]),
                sampling.TransformSamples(sampling.ExtractSegment3D2(training_inputMaskLoader, segment_size), [sampleFlipper])
            ])

            training_y_creator          = sampling.TransformSamples(sampling.ExtractSegment3D2(training_outputLoader, segment_size), [sampleFlipper])
    
            validation_x_creator        = sampling.Concat([
                sampling.TransformSamples(sampling.ExtractSegment3D2(validation_inputLoader, model.input_shape[0][1:4]), [sampleFlipper]),
                sampling.TransformSamples(sampling.ExtractSegment3D2(validation_inputMaskLoader, segment_size), [sampleFlipper])
            ])

            validation_y_creator        = sampling.TransformSamples(sampling.ExtractSegment3D2(validation_outputLoader, segment_size), [sampleFlipper])
            
            full_training_x_creator     = sampling.Concat([
                sampling.ExtractSegment3D2(full_training_inputLoader, model.input_shape[0][1:4]),
                sampling.ExtractSegment3D2(full_training_inputMaskLoader, segment_size)
            ])
            full_validation_x_creator   = sampling.Concat([
                sampling.ExtractSegment3D2(full_validation_inputLoader, model.input_shape[0][1:4]),
                sampling.ExtractSegment3D2(full_validation_inputMaskLoader, segment_size)
            ])
        # Train and validation samplers
        if network_architecture_id in [1, 2]:
            if center_sampling:
                training_sampler        = sampling.ForcedUniformCenterSampler(training_inputLoader, training_samplingRoiLoader, max_number_of_subjects_used=max_number_of_subjects_used_for_training, parallel_sample_creation=parallel_sample_creation)
                validation_sampler      = sampling.ForcedUniformCenterSampler(validation_inputLoader, validation_samplingRoiLoader, max_number_of_subjects_used=max_number_of_subjects_used_for_validation, parallel_sample_creation=parallel_sample_creation)
                full_training_sampler   = sampling.ForcedUniformCenterSampler(full_training_inputLoader, full_training_outputMaskLoader)
                full_validation_sampler = sampling.ForcedUniformCenterSampler(full_validation_inputLoader, full_validation_outputMaskLoader)
            else:
                training_sampler        = sampling.ClassWeightedRestrictedSampler(training_outputLoader, segment_size, training_samplingRoiLoader, max_number_of_subjects_used=max_number_of_subjects_used_for_training)
                validation_sampler      = sampling.ClassWeightedRestrictedSampler(validation_outputLoader, segment_size, validation_samplingRoiLoader, max_number_of_subjects_used=max_number_of_subjects_used_for_validation)
                full_training_sampler   = sampling.ClassWeightedRestrictedSampler(full_training_outputLoader, segment_size, full_training_outputMaskLoader)
                full_validation_sampler = sampling.ClassWeightedRestrictedSampler(full_validation_outputLoader, segment_size, full_validation_outputMaskLoader)
    else:
        # Testing data loaders
        full_testing_inputLoader = sampling.ImageLoader(input_paths, image_transformers=[fixedIntensityClipper, fixedIntensityTransformer])
        full_testing_inputMaskLoader = sampling.ImageLoader(input_masks)
        full_testing_outputMaskLoader = sampling.ImageLoader(output_masks)
        # X creator
        if network_architecture_id in [1, 2]:
            full_testing_x_creator = sampling.Concat([
                sampling.ExtractSegment3D2(full_testing_inputLoader, model.input_shape[0][1:4]),
                sampling.ExtractSegment3D2(full_testing_inputMaskLoader, segment_size)
            ])
        # Testing sampler
        if network_architecture_id in [1, 2]:
            if center_sampling:
                full_testing_sampler = sampling.ForcedUniformCenterSampler(full_testing_inputLoader, full_testing_outputMaskLoader)
            else:
                full_testing_sampler = sampling.ForcedUniformCoordinateSampler(full_testing_inputLoader, full_testing_outputMaskLoader)
    
    #########################################
    # If "Training": Make the callbacks etc #
    #########################################
    if data == "Training" and mode == "Training":
        callbacks = []
        if save_best_model_only:
            callbacks.append(ModelCheckpoint(os.path.join(round_dir, "model_train_loss.hdf5"), monitor='train_loss', save_best_only=True, period=1, mode='min', save_weights_only=save_weights_only))
            callbacks.append(ModelCheckpoint(os.path.join(round_dir, "model_val_loss.hdf5"), monitor='val_loss', save_best_only=True, period=1, mode='min', save_weights_only=save_weights_only))
            callbacks.append(ModelCheckpoint(os.path.join(round_dir, "model_full_train_binary_jaccard_mean.hdf5"), monitor="full_train_binary_jaccard_mean", save_best_only=True, period=1, mode='max', save_weights_only=save_weights_only))
            callbacks.append(ModelCheckpoint(os.path.join(round_dir, "model_full_train_binary_dice_mean.hdf5"), monitor="full_train_binary_dice_mean", save_best_only=True, period=1, mode='max', save_weights_only=save_weights_only))
            callbacks.append(ModelCheckpoint(os.path.join(round_dir, "model_full_train_binary_crossentropy_mean.hdf5"), monitor="full_train_binary_crossentropy_mean", save_best_only=True, period=1, mode='min', save_weights_only=save_weights_only))
            callbacks.append(ModelCheckpoint(os.path.join(round_dir, "model_full_val_binary_jaccard_mean.hdf5"), monitor="full_val_binary_jaccard_mean", save_best_only=True, period=1, mode='max', save_weights_only=save_weights_only))
            callbacks.append(ModelCheckpoint(os.path.join(round_dir, "model_full_val_binary_dice_mean.hdf5"), monitor="full_val_binary_dice_mean", save_best_only=True, period=1, mode='max', save_weights_only=save_weights_only))
            callbacks.append(ModelCheckpoint(os.path.join(round_dir, "model_full_val_binary_crossentropy_mean.hdf5"), monitor="full_val_binary_crossentropy_mean", save_best_only=True, period=1, mode='min', save_weights_only=save_weights_only))
            callbacks.append(ModelCheckpoint(os.path.join(round_dir, "model_full_val_binary_tversky_mean.hdf5"), monitor="full_val_binary_tversky_mean", save_best_only=True, period=1, mode='max', save_weights_only=save_weights_only))
        else:
            callbacks.append(ModelCheckpoint(os.path.join(round_dir, "model_epoch-{epoch}_val_loss-{val_loss:.4f}.hdf5"), monitor='val_loss', save_best_only=True, period=1, mode='min', save_weights_only=save_weights_only))
            callbacks.append(ModelCheckpoint(os.path.join(round_dir, "model_epoch-{epoch}_full_val_binary_dice_mean-{full_val_binary_dice_mean:.4f}.hdf5"), monitor="full_val_binary_dice_mean", save_best_only=True, period=1, mode='max', save_weights_only=save_weights_only))
            callbacks.append(ModelCheckpoint(os.path.join(round_dir, "model_epoch-{epoch}_full_val_binary_crossentropy_mean-{full_val_binary_crossentropy_mean:.4f}.hdf5"), monitor="full_val_binary_crossentropy_mean", save_best_only=True, period=1, mode='min', save_weights_only=save_weights_only))
        if reduce_learning_rate_on_plateau:
            if objective_function[0] == "cross-entropy":
                callbacks.append(ReduceLROnPlateau(monitor='full_val_binary_crossentropy_mean', factor=0.2, patience=150, verbose=True, mode='min', min_delta=0.0001, cooldown=0, min_lr=0))
            elif objective_function[0] == "dice":
                callbacks.append(ReduceLROnPlateau(monitor='full_val_binary_dice_mean', factor=0.2, patience=600, verbose=True, mode='max', min_delta=0.0001, cooldown=0, min_lr=5e-6))
            elif objective_function[0] == "jaccard" or objective_function[0] == "lovasz":
                callbacks.append(ReduceLROnPlateau(monitor='full_val_binary_jaccard_mean', factor=0.2, patience=150, verbose=True, mode='max', min_delta=0.0001, cooldown=0, min_lr=0))
            elif objective_function[0] == "tversky":
                callbacks.append(ReduceLROnPlateau(monitor='full_val_binary_tversky_mean', factor=0.2, patience=150, verbose=True, mode='max', min_delta=0.0001, cooldown=0, min_lr=0))
        if early_stopping:
            if objective_function[0] == "cross-entropy":
                callbacks.append(EarlyStopping(monitor='full_val_binary_crossentropy_mean', min_delta=0.0001, patience=300, verbose=True, mode='min', baseline=None))
            elif objective_function[0] == "dice":
                callbacks.append(EarlyStopping(monitor='full_val_binary_dice_mean', min_delta=0.0001, patience=300, verbose=True, mode='max', baseline=None))
            elif objective_function[0] == "jaccard" or objective_function[0] == "lovasz":
                callbacks.append(EarlyStopping(monitor='full_val_binary_jaccard_mean', min_delta=0.0001, patience=300, verbose=True, mode='max', baseline=None))
            elif objective_function[0] == "tversky":
                callbacks.append(EarlyStopping(monitor='full_val_binary_tversky_mean', min_delta=0.0001, patience=300, verbose=True, mode='max', baseline=None))
        callbacks.append(TensorBoard(log_dir=logs_dir, histogram_freq=0, batch_size=64, write_graph=False, write_grads=False, write_images=False))
    
    ####################################
    # If "Training": Compile the model #
    ####################################
    if data == "Training" and mode == "Training":
        def wrapped_partial(func, *args, **kwargs):
            partial_func = partial(func, *args, **kwargs)
            update_wrapper(partial_func, func)
            return partial_func
        model.compile(
            loss=objective,
            optimizer=SGD(lr=initial_learning_rate, momentum=0.9, nesterov=True) if optimizer == "SGD" else Adam(lr=initial_learning_rate),
            metrics=[
                wrapped_partial(loss.binary_crossentropy, sampling_correction=False),
                wrapped_partial(loss.binary_accuracy, sampling_correction=False),
                wrapped_partial(loss.binary_precision, sampling_correction=False),
                wrapped_partial(loss.binary_recall, sampling_correction=False),
                wrapped_partial(loss.binary_dice, sampling_correction=False),
                wrapped_partial(loss.binary_jaccard, sampling_correction=False),
                wrapped_partial(loss.binary_tversky, alpha=weighted_loss[0] if objective_function[0] == "tversky" else 0.5, beta=weighted_loss[1] if objective_function[0] == "tversky" else 0.5, sampling_correction=False),
                wrapped_partial(loss.binary_soft_vol_diff, sampling_correction=False),
                wrapped_partial(loss.binary_vol_diff, sampling_correction=False)
            ]
        )
    
    #############################
    # CREATE AND RUN DEEPVOXNET #
    #############################
    deepVoxNet = DeepVoxNet(model, center_sampling=center_sampling)
    if data == "Training" and mode == "Training":
        deepVoxNet.train(
            training_x_creator=training_x_creator,
            training_y_creator=training_y_creator,
            training_sampler=training_sampler,
            validation_x_creator=validation_x_creator,
            validation_y_creator=validation_y_creator,
            validation_sampler=validation_sampler,
            full_training_x_creator=full_training_x_creator,
            full_training_sampler=full_training_sampler,
            full_training_gt_loader=full_training_outputLoader if full_training_testing else None,
            full_training_output_paths=full_training_output_paths if full_training_testing else None,
            full_training_metrics=[metrics.binary_crossentropy, metrics.binary_accuracy, metrics.binary_precision, metrics.binary_recall, metrics.binary_dice, metrics.binary_jaccard, metrics.binary_soft_vol_diff, metrics.binary_vol_diff] if full_training_testing else None,
            full_training_metrics_modes=['min', 'max', 'max', 'max', 'max', 'max', 'min', 'min'] if full_training_testing else None,
            full_validation_x_creator=full_validation_x_creator,
            full_validation_sampler=full_validation_sampler,
            full_validation_gt_loader=full_validation_outputLoader if full_validation_testing else None,
            full_validation_output_paths=full_validation_output_paths if full_validation_testing else None,
            full_validation_metrics=[metrics.binary_crossentropy, metrics.binary_accuracy, metrics.binary_precision, metrics.binary_recall, metrics.binary_dice, metrics.binary_jaccard, metrics.binary_soft_vol_diff, metrics.binary_vol_diff] if full_validation_testing else None,
            full_validation_metrics_modes=['min', 'max', 'max', 'max', 'max', 'max', 'min', 'min'] if full_validation_testing else None,
            nb_epochs=nb_epochs,
            nb_subepochs=nb_subepochs,
            nb_runs_per_subepoch=nb_runs_per_subepoch,
            nb_samples_training=nb_samples_training,
            nb_samples_validation=nb_samples_validation,
            sgd_batch_size=sgd_batch_size,
            prediction_batch_size=prediction_batch_size,
            fixed_learning_rate=fixed_learning_rate,
            parallel_sample_creation=parallel_sample_creation,
            path_training_result=path_training_result,
            callbacks=callbacks
        )
        deepVoxNet.save(model_path, save_weights_only=save_weights_only)
    elif data == "Training" and mode == "Testing":
        if full_training_testing:
            for i, out_path in enumerate(full_training_output_paths):
                deepVoxNet.predict(
                    x_creator=full_training_x_creator,
                    sampler=full_training_sampler,
                    subject_id=i,
                    out_path=out_path,
                    verbose=True,
                    batch_size=1,
                    auto_recalibration=False,
                    stack_recalibrated=False,
                    output_layer_idx=[0],
                    include_output_layer_name_in_out_path=False,
                    auto_return_first_only=True
                )
        if full_validation_testing:
            for i, out_path in enumerate(full_validation_output_paths):
                deepVoxNet.predict(
                    x_creator=full_validation_x_creator,
                    sampler=full_validation_sampler,
                    subject_id=i,
                    out_path=out_path,
                    verbose=True,
                    batch_size=1,
                    auto_recalibration=False,
                    stack_recalibrated=False,
                    output_layer_idx=[0],
                    include_output_layer_name_in_out_path=False,
                    auto_return_first_only=True
                )
    else:
        for i, out_path in enumerate(full_testing_output_paths):
            deepVoxNet.predict(
                x_creator=full_testing_x_creator,
                sampler=full_testing_sampler,
                subject_id=i,
                out_path=out_path,
                verbose=True,
                batch_size=1,
                auto_recalibration=False,
                stack_recalibrated=False,
                output_layer_idx=[0],
                include_output_layer_name_in_out_path=False,
                auto_return_first_only=True
            )

    K.clear_session()

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

# function definition needed to define entry_points such that script can be installed
def main():
    # parsing parameters
    parser = argparse.ArgumentParser(description='This script is used to train a U-net model for liver or lesion segmentation. ')
    parser.add_argument('data_path', help = 'Directory of the folder containing the training and/or test datasets.')
    parser.add_argument('--data', 
                    default = "Training", 
                    help = '"data" can be "Training" or "Testing". The input data need to organized as follows: there should be a folder (its directory is specified in the positional argument "data_path") containing one subfolder "Training" for training datasets and/or one subfolder "Testing" for testing datasets; in each subfolder (e.g., "Training"), there should be subfolders "case_0", "case_1", "case_2", ..., where each subfolder contains a pre-processed image in NIFTI format (The file name is specified in the optional argument "inputs") and a pre-processed ground-truth segmentation in NIFTI format (The file name is specified in the optional argument "outputs"). ')    
    parser.add_argument('--mode', 
                    default = "Training", 
                    help = 'Status for training or prediction. "mode" can be "Training" or "Testing".')                    
    parser.add_argument('--nb_subjects', 
                    default = 196, 
                    type = int,
                    help = 'Names of the folders containing the image and ground truth of one data.')   
    parser.add_argument('--training_index_range', 
                    default = [0,169], 
                    nargs = '+',
                    type = int,
                    help = 'indice range for training datasets.')          
    parser.add_argument('--validation_index_range', 
                    default = [169, 196], 
                    nargs = '+', 
                    type = int,
                    help = 'indice range for validation datasets.')       
    parser.add_argument('--inputs', 
                    default = ["CT"],
                    nargs = '+', 
                    help = 'Prefix of the image file name.') 
    parser.add_argument('--outputs', 
                    default = ["GT"],
                    nargs = '+',  
                    help = 'Prefix of the ground truth file name.') 
    parser.add_argument('--run_folder_name', 
                    default = "Runs_liver_seg_output_size_163_136_136", 
                    help = 'Folder for saving the outputs and the trained model parameters.')                    
    parser.add_argument('--network_architecture_id', 
                    default = 1, 
                    type = int,
                    help = 'Index of network architecture.')           
    parser.add_argument('--segment_size', 
                    default = [163,136,136], 
                    type = int,
                    nargs = 3,
                    help = 'List containing the output patch size.')  
    parser.add_argument('--no_center_sampling', 
                    help = 'whether to sample the input patch in the coordinate center of the image.',
                    action = 'store_true')       
    parser.add_argument('--sampling_roi', 
                    help = 'whether to define a ROI for sampling.',
                    action = 'store_true')      
    parser.add_argument('--objective_function', 
                    default = "dice", 
                    help = 'loss function.')    
    parser.add_argument('--objective_function_weights', 
                    default = None, 
                    help = 'Weights for each loss function if multiple loss functions are combined as one objective function.')                       
    parser.add_argument('--weight_decay', 
                    default = 1e-5, 
                    type = float,
                    help = 'Weight for L2 regularization.')
    parser.add_argument('--weighted_loss', 
                    default = None, 
                    nargs = '+',
                    help = 'List of weights for the loss function (e.g., binary crossentropy loss, tversky loss)')    
    parser.add_argument('--clip_input', 
                    help = 'Whether to clip the input image intensity.',
                    action = 'store_true')          
    parser.add_argument('--normalize_input', 
                    help = 'Whether to normalize the input image intensity.',
                    action = 'store_true')              
    parser.add_argument('--no_data_augmentation', 
                    help = 'Whether to perform data augmentation.',
                    action = 'store_true')          
    parser.add_argument('--optimizer', 
                    default = "ADAM", 
                    help = 'Optimizer for CNN optimization.')                        
    parser.add_argument('--fixed_learning_rate', 
                    default = 1e-3, 
                    type = float,
                    help = 'Learning rate.')     
    parser.add_argument('--no_reduce_learning_rate_on_plateau', 
                    help = 'Learning rate.',
                    action = 'store_true')                          
    parser.add_argument('--early_stopping', 
                    help = 'Stop training when the metric change on full validation datasets is smaller than a certain value.',
                    action = 'store_true')     
    parser.add_argument('--sgd_batch_size', 
                    default = 1, 
                    type = int,
                    help = 'Batch size for training.')    
    parser.add_argument('--prediction_batch_size', 
                    default = 1, 
                    type = int,
                    help = 'Batch size for prediction.')                      
    parser.add_argument('--nb_samples_training', 
                    default = 40, 
                    type = int,
                    help = 'Number of sampled input patches for training.')     
    parser.add_argument('--nb_samples_validation', 
                    default = 40, 
                    type = int,
                    help = 'Number of sampled input patches for prediction.')          
    parser.add_argument('--max_number_of_subjects_used_for_training', 
                    default = None, 
                    type = int,
                    help = 'Maximum number of input images for training.')                  
    parser.add_argument('--max_number_of_subjects_used_for_validation', 
                    default = None, 
                    type = int,
                    help = 'Maximum number of input images for prediction.')         
    parser.add_argument('--nb_epochs', 
                    default = 450, 
                    type = int,
                    help = 'Number of epochs.')        
    parser.add_argument('--nb_subepochs', 
                    default = 10, 
                    type = int,
                    help = 'Number of sub-epochs. Validation on the full images is performed every N sub-epochs.')       
    parser.add_argument('--nb_runs_per_subepoch', 
                    default = 1, 
                    type = int,
                    help = 'Number of iterations in each sub-epoch. The total epochs = nb_epochs * nb_subepochs * nb_runs_per_subepoch.')                     
    parser.add_argument('--parallel_sample_creation', 
                    default = False, 
                    type = int,
                    help = 'Number of parallel computation for sampling.')             
    parser.add_argument('--full_training_testing', 
                    help = 'Whether to predict for training datasets.',
                    action = 'store_true')                
    parser.add_argument('--no_full_validation_testing', 
                    help = 'Whether to predict for validation datasets.',
                    action = 'store_true')          
    parser.add_argument('--full_image_testing_appendix', 
                    default = None, 
                    help = 'Appendix for the files of the predictions.')                   
    parser.add_argument('--model_paths', 
                    default = None, 
                    help = 'Path of the weights of the trained model for initialization of a new training.') 
    parser.add_argument('--no_load_weights_only', 
                    help = 'Whether to load weights.',
                    action = 'store_true')    
    parser.add_argument('--no_save_weights_only', 
                    help = 'Whether to save weights.',
                    action = 'store_true')       
    parser.add_argument('--no_save_best_model_only', 
                    help = 'Whether to save the best model.',
                    action = 'store_true')     
    args = parser.parse_args()
    
    # define a dictionary of parameters    
    parameters = {
        "data_path": args.data_path,
        "data": args.data,
        "mode": args.mode,
        "subjects": ["case_{}".format(s) for s in range(args.nb_subjects)],
        "training_indices": list(range(args.training_index_range[0], args.training_index_range[1])) if len(args.training_index_range)>1 else list(range(args.training_index_range[0])),
        "validation_indices": list(range(args.validation_index_range[0], args.validation_index_range[1])) if len(args.validation_index_range)>1 else list(range(args.validation_index_range[0])),
        "inputs": args.inputs,
        "outputs": args.outputs,
        "run_folder_name": args.run_folder_name,
        "network_architecture_id": args.network_architecture_id,
        "segment_size": args.segment_size,
        "center_sampling": not args.no_center_sampling,
        "sampling_roi": args.sampling_roi,
        "objective_function": args.objective_function,
        "objective_function_weights": args.objective_function_weights,
        "weight_decay": args.weight_decay,
        "weighted_loss": args.weighted_loss,
        "clip_input": args.clip_input,
        "normalize_input": args.normalize_input,
        "data_augmentation": not args.no_data_augmentation,
        "optimizer": args.optimizer,
        "fixed_learning_rate": args.fixed_learning_rate,
        "reduce_learning_rate_on_plateau": not args.no_reduce_learning_rate_on_plateau,
        "early_stopping": args.early_stopping,
        "sgd_batch_size": args.sgd_batch_size,
        "prediction_batch_size": args.prediction_batch_size,
        "nb_samples_training": args.nb_samples_training,
        "nb_samples_validation": args.nb_samples_validation,
        "max_number_of_subjects_used_for_training": args.max_number_of_subjects_used_for_training,
        "max_number_of_subjects_used_for_validation": args.max_number_of_subjects_used_for_validation,
        "nb_epochs": args.nb_epochs,
        "nb_subepochs": args.nb_subepochs,
        "nb_runs_per_subepoch": args.nb_runs_per_subepoch,
        "parallel_sample_creation": args.parallel_sample_creation,
        "full_training_testing": args.full_training_testing,
        "full_validation_testing": not args.no_full_validation_testing,
        "full_image_testing_appendix": args.full_image_testing_appendix,
        "model_paths": args.model_paths,
        "load_weights_only": not args.no_load_weights_only,
        "save_weights_only": not args.no_save_weights_only,
        "save_best_model_only": not args.no_save_best_model_only
    }

    run(parameters)

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

if __name__ == '__main__':
  main()
