import pyliverlesionseg
import numpy as np
from keras.utils import to_categorical
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping


class ModelCheckpoint(ModelCheckpoint):
    """
        Identical to Keras' ModelCheckpoint callback
    """
    pass


class TensorBoard(TensorBoard):
    """
        Identical to Keras' TensorBoard callback
    """
    pass


class ReduceLROnPlateau(ReduceLROnPlateau):
    """
        Identical to Keras' ReduceLROnPlateau callback, without setting wait counter to zero each time fit function is called
    """
    def on_train_begin(self, logs=None):
        pass


class EarlyStopping(EarlyStopping):
    """
        Identical to Keras' EarlyStopping callback, without setting the wait counter to zero each time fit function is called
    """
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None):  # , restore_best_weights=False):
        super(EarlyStopping, self).__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode, baseline=baseline)  # , restore_best_weights=restore_best_weights)
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_train_begin(self, logs=None):
        pass


class MetricNameChanger(Callback):
    """
        Prepends "train_" to training metrics
    """
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs_ = logs.copy()
        for k, v in logs_.items():
            if not k.startswith("full_") and not k.startswith("val_") and not k.startswith("train_"):
                logs["train_" + k] = logs.pop(k)


class History(Callback):
    """
        Similar to Keras' History Callback but this one keeps its history (*)
    """
    def on_train_begin(self, logs=None):
        if not hasattr(self, 'epoch'): # (*)
            self.epoch = []
            self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


class FullImageSetTester(Callback):

    def __init__(self, deepvoxnet, full_image_set_name, full_image_set_metrics, full_image_set_output_paths, full_image_set_x_creator, full_image_set_sampler, full_image_set_gt_loader=None, batch_size=1, period=1, verbose=True, modes=None):
        assert full_image_set_metrics or full_image_set_output_paths
        self.deepvoxnet = deepvoxnet
        self.full_image_set_name = full_image_set_name
        self.full_image_set_x_creator = full_image_set_x_creator
        self.full_image_set_sampler = full_image_set_sampler
        self.batch_size = batch_size
        self.period = period
        self.verbose = verbose
        self.full_image_set_history = {}
        if not isinstance(full_image_set_gt_loader, dict):
            if full_image_set_gt_loader is not None:
                full_image_set_gt_loader = {self.deepvoxnet.model.output_names[0]: full_image_set_gt_loader}
            else:
                full_image_set_gt_loader = {}
        self.full_image_set_gt_loader = full_image_set_gt_loader
        if not isinstance(full_image_set_metrics, dict):
            assert full_image_set_metrics is None or isinstance(full_image_set_metrics, list)
            if full_image_set_metrics:
                full_image_set_metrics = {self.deepvoxnet.model.output_names[0]: full_image_set_metrics}
            else:
                full_image_set_metrics = {}
        for k, v in full_image_set_metrics.items():
            full_image_set_metrics[k] = [pyliverlesionseg.components.metrics.lookuptable.get(full_image_set_metric, full_image_set_metric) for full_image_set_metric in v]
        self.full_image_set_metrics = full_image_set_metrics
        if len(self.full_image_set_metrics) < 2:
            self.store_layer_name_in_metric_name = False
        else:
            self.store_layer_name_in_metric_name = True
        if not isinstance(modes, dict):
            assert modes is None or isinstance(modes, list)
            if modes:
                modes = {self.deepvoxnet.model.output_names[0]: modes}
            else:
                modes = {}
        self.modes = modes
        if not isinstance(full_image_set_output_paths, dict):
            assert full_image_set_output_paths is None or isinstance(full_image_set_output_paths, list)
            if full_image_set_output_paths:
                full_image_set_output_paths = {self.deepvoxnet.model.output_names[0]: full_image_set_output_paths}
            else:
                full_image_set_output_paths = {}
        self.full_image_set_output_paths = full_image_set_output_paths
        self.requested_output_layers_names = [name for name in self.deepvoxnet.model.output_names if name in self.full_image_set_metrics or name in self.full_image_set_output_paths]
        self.requested_output_layers_idx = [self.deepvoxnet.model.output_names.index(name) for name in self.requested_output_layers_names]
        for name in self.requested_output_layers_names:
            if name in self.full_image_set_metrics and name not in self.full_image_set_output_paths:
                self.full_image_set_output_paths[name] = [None] * self.full_image_set_sampler.nb_subjects
            elif name in self.full_image_set_output_paths and name not in self.full_image_set_metrics:
                self.full_image_set_metrics[name] = []
                self.full_image_set_gt_loader[name] = None
            if name in self.full_image_set_metrics and name not in self.modes:
                self.modes[name] = ['min'] * len(self.full_image_set_metrics[name])
        assert list(self.full_image_set_gt_loader.keys()) == list(self.full_image_set_metrics.keys()) == list(self.full_image_set_output_paths.keys()) == list(self.modes.keys()) == self.requested_output_layers_names
        for k, v in self.full_image_set_gt_loader.items():
            if v:
                assert self.full_image_set_sampler.nb_subjects == v.get_number_of_subjects()
        super(FullImageSetTester, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if not (epoch + 1) % self.period:
            self._test(logs)
            self._fill_None()
        elif not epoch:
            for k, v in self.full_image_set_metrics.items():
                for metric, mode in zip(v, self.modes[k]):
                    metric_name = self._get_metric_name(metric, "mean", k)
                    logs[metric_name] = np.float32(-np.Inf) if mode is 'max' else np.float32(np.Inf)
                    self.full_image_set_history[metric_name] = self.full_image_set_history.get(metric_name, []) + [np.float32(-np.Inf) if mode is 'max' else np.float32(np.Inf)]
                    metric_name = self._get_metric_name(metric, "median", k)
                    logs[metric_name] = np.float32(-np.Inf) if mode is 'max' else np.float32(np.Inf)
                    self.full_image_set_history[metric_name] = self.full_image_set_history.get(metric_name, []) + [np.float32(-np.Inf) if mode is 'max' else np.float32(np.Inf)]
        else:
            self._fill_repeat(logs)

    def _test(self, logs):
        subjects_values = {}
        subjects_values_recalibrated = {}
        for subject_id in range(self.full_image_set_sampler.nb_subjects):
            out_path = [self.full_image_set_output_paths[name][subject_id] for name in self.requested_output_layers_names]
            pred = self.deepvoxnet.predict(self.full_image_set_x_creator, self.full_image_set_sampler, subject_id, out_path=out_path, output_layer_idx=self.requested_output_layers_idx, include_output_layer_name_in_out_path=True if len(out_path) > 1 and out_path[0] == out_path[1] else False, verbose=self.verbose, batch_size=self.batch_size, stack_recalibrated=True, auto_return_first_only=False)
            metric_printout = []
            metric_printout_recalibrated = []
            for k, v in self.full_image_set_metrics.items():
                if v:
                    gt, _affine, _header = self.full_image_set_gt_loader[k].load(subject_id)
                    if self.deepvoxnet.nb_classes_in_multiclass:
                        if k in self.deepvoxnet.nb_classes_in_multiclass and self.deepvoxnet.nb_classes_in_multiclass[k] and gt.shape[3] == 1:
                            gt = to_categorical(gt, num_classes=self.deepvoxnet.nb_classes_in_multiclass[k])
                    pred_ = pred[self.requested_output_layers_names.index(k)]
                    subjects_values[k] = subjects_values.get(k, []) + [[metric(gt[None], pred_[None] if pred_.ndim == 4 else pred_[None, ..., None])[0] for metric in v]]
                    metric_printout += ["{} {:.3f}  ".format(self._get_metric_name(metric, None, k), value) for metric, value in zip(v, subjects_values[k][-1])]
            print("Subject #{}:  {}".format(subject_id, "".join(metric_printout)))
            if metric_printout_recalibrated:
                print("Subject #{} recalibrated:  {}".format(subject_id, "".join(metric_printout_recalibrated)))
        metric_printout_mean = []
        metric_printout_median = []
        metric_printout_recalibrated_mean = []
        metric_printout_recalibrated_median = []
        for k, v in self.full_image_set_metrics.items():
            if v:
                mean_subject_values = np.mean(subjects_values[k], axis=0)
                median_subject_values = np.median(subjects_values[k], axis=0)
                for i, metric in enumerate(v):
                    metric_name = self._get_metric_name(metric, "mean", k)
                    logs[metric_name] = mean_subject_values[i]
                    self.full_image_set_history[metric_name] = self.full_image_set_history.get(metric_name, []) + [mean_subject_values[i]]
                    metric_name = self._get_metric_name(metric, "median", k)
                    logs[metric_name] = median_subject_values[i]
                    self.full_image_set_history[metric_name] = self.full_image_set_history.get(metric_name, []) + [median_subject_values[i]]
                metric_printout_mean += ["{} {:.3f}  ".format(self._get_metric_name(metric, None, k), value) for metric, value in zip(v, mean_subject_values)]
                metric_printout_median += ["{} {:.3f}  ".format(self._get_metric_name(metric, None, k), value) for metric, value in zip(v, median_subject_values)]
                if k in subjects_values_recalibrated:
                    mean_subject_values_recalibrated = np.mean(subjects_values_recalibrated[k], axis=0)
                    median_subject_values_recalibrated = np.median(subjects_values_recalibrated[k], axis=0)
                    for i, metric in enumerate(v):
                        metric_name = self._get_metric_name(metric, "mean", k) + "_recalibrated"
                        logs[metric_name] = mean_subject_values_recalibrated[i]
                        self.full_image_set_history[metric_name] = self.full_image_set_history.get(metric_name, []) + [mean_subject_values_recalibrated[i]]
                        metric_name = self._get_metric_name(metric, "median", k) + "_recalibrated"
                        logs[metric_name] = median_subject_values_recalibrated[i]
                        self.full_image_set_history[metric_name] = self.full_image_set_history.get(metric_name, []) + [median_subject_values_recalibrated[i]]
                    metric_printout_recalibrated_mean += ["{} {:.3f}  ".format(self._get_metric_name(metric, None, k), value) for metric, value in zip(v, mean_subject_values_recalibrated)]
                    metric_printout_recalibrated_median += ["{} {:.3f}  ".format(self._get_metric_name(metric, None, k), value) for metric, value in zip(v, median_subject_values_recalibrated)]
        print("Mean:  {}".format("".join(metric_printout_mean)))
        print("Median:  {}".format("".join(metric_printout_median)))
        if metric_printout_recalibrated_mean:
            print("Mean recalibrated:  {}".format("".join(metric_printout_recalibrated_mean)))
            print("Median recalibrated:  {}".format("".join(metric_printout_recalibrated_median)))

    def _fill_None(self):
        for k, v in self.full_image_set_metrics.items():
            for metric in v:
                metric_name = self._get_metric_name(metric, "mean", k)
                self.full_image_set_history[metric_name][-self.period:-1] = [None] * (self.period - 1)
                metric_name = self._get_metric_name(metric, "median", k)
                self.full_image_set_history[metric_name][-self.period:-1] = [None] * (self.period - 1)

    def _fill_repeat(self, logs):
        for k, v in self.full_image_set_metrics.items():
            for metric in v:
                metric_name = self._get_metric_name(metric, "mean", k)
                logs[metric_name] = self.full_image_set_history[metric_name][-1]
                self.full_image_set_history[metric_name] = self.full_image_set_history.get(metric_name, []) + [self.full_image_set_history[metric_name][-1]]
                metric_name = self._get_metric_name(metric, "median", k)
                logs[metric_name] = self.full_image_set_history[metric_name][-1]
                self.full_image_set_history[metric_name] = self.full_image_set_history.get(metric_name, []) + [self.full_image_set_history[metric_name][-1]]

    def _get_metric_name(self, metric, moment=None, layer_name=None):
        moment = "_{}".format(moment) if moment else ""
        layer_name = "_{}".format(layer_name) if layer_name and self.store_layer_name_in_metric_name else ""
        return "full_{}{}_{}{}".format(self.full_image_set_name, layer_name, metric.__name__, moment)
