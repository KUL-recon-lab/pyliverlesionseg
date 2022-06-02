import numpy as np
import time
import random
import threading
import copy
try:
    import Queue #python 2
except ImportError:
    import queue as Queue


class SampleGenerator(object):

    def __init__(
            self,
            x_creator,
            y_creator,
            sampler,
            parallel=False,
            verbose=True
    ):
        """
        number of threads to use for the sample generation
        Note that for parallel > 1; (sampler, x_creator, y_creator) will be deepcopied for each requested thread
        This is necessary to let the caching mechanism in (sampler, x_creator, y_creator) to work
        This has other side effects, e.g. each thread's sampler will use max_number_of_subjects_used; resulting in a total of parallel*max_number_of_subjects_used subjects sampled

        :param x_creator:
        :param y_creator:
        :param sampler:
        :param parallel:
        :param verbose:
        """
        self.x_creator = x_creator
        self.y_creator = y_creator
        self.sampler = sampler
        self.parallel = parallel
        if self.parallel:
            if self.parallel == 1:
                self.nb_threads = 2
            else:
                self.nb_threads = self.parallel

        self.verbose = True
        #if multithreaded, create the (sampler,x_creator,y_creator) for each thread
        if self.parallel:
            if self.sampler.__class__.__name__ is 'ForcedUniformSampler':
                self.sampler_x_creator_y_creator_per_thread = [copy.copy((self.sampler,)) + copy.deepcopy((self.x_creator, self.y_creator)) for _i in range(self.nb_threads)]
            else:
                self.sampler_x_creator_y_creator_per_thread = [copy.deepcopy((self.sampler, self.x_creator, self.y_creator)) for _i in range(self.nb_threads)]

    #Internal variables used for storing the samples and targets, so they are shared between the different threads
    _samples, _targets = [], []
    def _actual_preparation_thread(self, nb, epoch_identifier, sampler, x_creator, y_creator, thread_i=None):
        #print("Double check randomness: {}".format(np.random.rand()))
        #start_time = time.time()
        for subject_id, coordinates in sampler.iter_random_sample_coordinates(nb, epoch_identifier=epoch_identifier, thread_i=thread_i):
            self._samples.append(x_creator.get_sample(subject_id, coordinates, epoch_identifier=epoch_identifier))
            self._targets.append(y_creator.get_sample(subject_id, coordinates, epoch_identifier=epoch_identifier))
        #print("Thread prepared {} samples in {} s".format(nb,time.time()-start_time))

    def _prepare(self, nb, epoch_identifier):
        start_time = time.time()
        #Put newly created samples and targets in self._samples and self._targets
        self._samples, self._targets = [], []
        if self.parallel:
            assert nb is None or nb % self.nb_threads == 0
            threads = []
            for thread_i in range(self.nb_threads):
                thread_sampler, thread_x_creator, thread_y_creator = self.sampler_x_creator_y_creator_per_thread[thread_i]
                thread = threading.Thread(target=self._actual_preparation_thread, args=(nb // self.nb_threads if nb is not None else nb, epoch_identifier, thread_sampler, thread_x_creator, thread_y_creator, thread_i))
                thread.start()
                threads.append(thread)
            [thread.join() for thread in threads]
        else:
            self._actual_preparation_thread(nb, epoch_identifier, self.sampler, self.x_creator, self.y_creator)
        #shuffle the samples and targets
        c = list(zip(self._samples, self._targets))
        random.shuffle(c)
        self._samples, self._targets = zip(*c)
        #ensure that the different samples have the same dimensions (can be a problem if we work with full images)
        for sample in self._samples:
            for sample_part_i, _sample_part in enumerate(sample):
                if not sample[sample_part_i].shape == self._samples[0][sample_part_i].shape:
                    raise ValueError("The sample_part.shape is not identical for all samples. Set nb_samples to 1 or fix the shape.")
        #create the continuous x and y
        #start_time_np = time.time()
        x = [np.array([sample[sample_part_i] for sample in self._samples], np.float32) for sample_part_i, _ in enumerate(self._samples[0])]
        y = [np.array([target[target_part_i] for target in self._targets], np.float32) for target_part_i, _ in enumerate(self._targets[0])]
        #print("Numpy generation took {} s. Shape: {} {}".format(time.time()-start_time_np, x[0].shape, y.shape))
        self._samples, self._targets = [], [] #No longer needed, clear memory
        #put results back in queue
        self.q.put(((x, y), time.time() - start_time))
            
    def prepare(self, nb, epoch_identifier):
        '''
        Prepare nb samples for epoch epoch_identifier.
        '''
        if self.parallel:
            self.q = Queue.Queue()
            self.p = threading.Thread(target=self._prepare, args=(nb, epoch_identifier))
            self.p.start()
        else:
            self.p = False
            self.q = Queue.Queue()
            self._prepare(nb, epoch_identifier)

    def get(self):
        '''
        Get the samples that were previously asked to be prepared.
        '''
        start_time_data_transfer = time.time()
        x_and_y, duration_generation = self.q.get()
        duration_data_transfer = time.time() - start_time_data_transfer
        if self.p:
            self.p.join()
        if self.verbose:
            print("Samples generated in {:.2f} s and transferred in {:.2f} s".format(duration_generation, duration_data_transfer))
        return x_and_y
