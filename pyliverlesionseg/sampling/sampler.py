'''
Created on May 26, 2017

@author: drobbe1
'''


import numpy as np
from random import sample, shuffle, seed, getstate


class Sampler(object):
    ''' Uniform sampling. '''
    def __init__(self, base_image_loader, roi_loader=None, max_number_of_subjects_used=None):
        self.base_image_loader = base_image_loader
        self.nb_subjects = base_image_loader.get_number_of_subjects()
        self.roi_loader = roi_loader
        if max_number_of_subjects_used is None or max_number_of_subjects_used > self.nb_subjects:
            max_number_of_subjects_used = self.nb_subjects
        self.max_number_of_subjects_used = max_number_of_subjects_used
        
    def get_random_sample_coordinates(self, nb_sample_coordinates=100, epoch_identifier=None):
        return list(self.iter_random_sample_coordinates(self, nb_sample_coordinates=nb_sample_coordinates, epoch_identifier=epoch_identifier))
    
    def iter_random_sample_coordinates(self, nb_sample_coordinates=100, epoch_identifier=None, thread_i=None):
        ## First we construct an array with the subject ids of the subjects that will be used
        subject_ids = np.arange(self.nb_subjects)
        np.random.shuffle(subject_ids)
        max_number_of_subjects_used_ = min(self.max_number_of_subjects_used, nb_sample_coordinates)
        if max_number_of_subjects_used_ < self.nb_subjects:
            subject_ids = subject_ids[:max_number_of_subjects_used_]
        ## Now we construct an array that tells us how many times each of these subjects will be used
        nb_times_subject_ids = np.ones_like(subject_ids) * (nb_sample_coordinates // len(subject_ids)) # factor is minimum 1
        if nb_sample_coordinates % len(subject_ids):
            nb_times_subject_ids[:int(nb_sample_coordinates % len(subject_ids))] += 1 # depending on the remainder some subjects will be used more to match the number of requested samples
        for subject_id, nb_times_subject_id in zip(subject_ids, nb_times_subject_ids):
            if self.roi_loader is not None:
                roi, _wc, _o = self.roi_loader.load(subject_id, epoch_identifier)
            else:
                base_image, _wc, _o = self.base_image_loader.load(subject_id, epoch_identifier)
                roi = np.ones_like(base_image, dtype=np.bool)
            ## First we construct an array with the possible coordinates to sample from
            possible_coordinates = np.array(list(zip(*np.nonzero(roi))))
            possible_coordinates_ids = np.random.randint(0, len(possible_coordinates), size=nb_times_subject_id)
            for i in possible_coordinates_ids:
                yield (subject_id, tuple(possible_coordinates[i]))
        #     np.random.shuffle(possible_coordinates)
        #     if nb_times_subject_id <= len(possible_coordinates): # this is expected to be the case
        #         possible_coordinates = possible_coordinates[:int(nb_times_subject_id)]
        #     ## Then we construct the array that tells us how many times each coordinate needs to be used
        #     nb_times_possible_coordinates = np.ones(len(possible_coordinates)) * (nb_times_subject_id // len(possible_coordinates)) # in most cases factor will be 1
        #     nb_times_possible_coordinates[:int(nb_times_subject_id % len(possible_coordinates))] += 1 # in most cases remained will be 0
        #     for (x, y, z, _), nb_times_possible_coordinate in zip(possible_coordinates, nb_times_possible_coordinates):
        #         for i in range(int(nb_times_possible_coordinate)):
        #             yield (subject_id, (x, y, z))
        # ## Watch out: the way this is constructed if multiple times the same coordinate needs to be used, there will be NO data augmentation on those!
            
    def get_base_image(self, subject_id):
        return self.base_image_loader.load(subject_id, None)

    def get_roi_image(self, subject_id):
        return self.roi_loader.load(subject_id, None) if self.roi_loader else None


class ForcedUniformSampler(object):
    '''Uniform sampling made more uniform. '''
    def __init__(self, base_image_loader, roi_loader=None, max_number_of_subjects_used=50, nb_threads=1, segment_size=[1, 1, 1]):
        self.base_image_loader = base_image_loader
        self.nb_subjects = base_image_loader.get_number_of_subjects()
        self.roi_loader = roi_loader
        self.max_number_of_subjects_used = max_number_of_subjects_used
        self.list_of_all_subjects = list(range(self.nb_subjects))
        self.list_of_subjects_to_sample_from = []
        self.list_of_coordinate_counts_per_subject = [None]*self.nb_subjects
        self.nb_threads = nb_threads or 1
        self.threads_status = [False]*self.nb_threads
        self.sampled_in_this_step = []
        self.segment_size = segment_size
        self.segment = np.ones(self.segment_size)
        
    def get_next_subject(self, current_list_of_subjects):
        if not self.list_of_subjects_to_sample_from:
            self.list_of_subjects_to_sample_from += sample(self.list_of_all_subjects, self.nb_subjects)
        counts = [sum(1 if s == s_ else 0 for s_ in self.sampled_in_this_step) for s in self.list_of_subjects_to_sample_from]
        try:
            next_subject = self.list_of_subjects_to_sample_from.pop(np.argsort(counts)[0])
            current_list_of_subjects.append(next_subject)
            self.sampled_in_this_step.append(next_subject)
        except:
            pass
        
    def get_next_coordinates_and_update_counts(self, subject, possible_coordinates, nb_samples):
        if self.list_of_coordinate_counts_per_subject[subject] is None:
            self.list_of_coordinate_counts_per_subject[subject] = possible_coordinates.astype(dtype='uint8')
        else:
            self.list_of_coordinate_counts_per_subject[subject] = np.maximum(self.list_of_coordinate_counts_per_subject[subject], possible_coordinates.astype(dtype='uint8'))
        nonzero_coordinates = np.nonzero(self.list_of_coordinate_counts_per_subject[subject])
        nonzero_elements = self.list_of_coordinate_counts_per_subject[subject][nonzero_coordinates]
        nonzero_coordinates = list(zip(*nonzero_coordinates))
        _ = list(zip(nonzero_coordinates, nonzero_elements))
        # shuffle(_) # is very inefficient; takes a lot of computation time
        nonzero_coordinates, nonzero_elements = zip(*_)
        samples = [nonzero_coordinates[i] for i in np.argsort(nonzero_elements)][:nb_samples]
        for coordinate in samples:
            self.update_counts(subject, coordinate)
        return samples
    
    def update_counts(self, subject, coordinate):
        x, y, z = coordinate
        X, Y, Z = self.list_of_coordinate_counts_per_subject[subject].shape
        sx, sy, sz = self.segment_size
        valid_slice_in_counts = (slice(max(0, x - sx // 2), min(X, x + sx // 2)), slice(max(0, y - sy // 2), min(Y, y + sy // 2)), slice(max(0, z - sz // 2), min(Z, z + sz // 2)))
        self.list_of_coordinate_counts_per_subject[subject][valid_slice_in_counts] += 1 
        
    def get_random_sample_coordinates(self, nb_sample_coordinates=100, epoch_identifier=None):
        return list(self.iter_random_sample_coordinates(self, nb_sample_coordinates=100, epoch_identifier=None))
    
    def iter_random_sample_coordinates(self, nb_sample_coordinates=100, epoch_identifier=None, thread_i=None):
        max_number_of_subjects_used_ = min(self.max_number_of_subjects_used, nb_sample_coordinates)
        assert not nb_sample_coordinates % max_number_of_subjects_used_
        subjects = []
        while len(subjects) < max_number_of_subjects_used_:
            self.get_next_subject(subjects)
        self.threads_status[thread_i or 0] = True
        if np.all(self.threads_status):
            self.threads_status[:] = [True]*self.nb_threads
        nb_samples_per_subject = nb_sample_coordinates // max_number_of_subjects_used_
        for s in subjects:
            if self.roi_loader is not None:
                roi, _wc, _o = self.roi_loader.load(s, epoch_identifier)
            else:
                base_image, _wc, _o = self.base_image_loader.load(s, epoch_identifier)
                roi = np.ones_like(base_image, dtype=np.bool)
            sample_coordinates = self.get_next_coordinates_and_update_counts(s, roi[..., 0], nb_samples_per_subject)
            for c in sample_coordinates:
                yield (s, c)
            
    def get_base_image(self, subject_id, ):
        return self.base_image_loader.load(subject_id,None)

    def get_roi_image(self, subject_id, ):
        if self.roi_loader is None:
            return None
        return self.roi_loader.load(subject_id,None)
    

class CenterSampler(Sampler):
    """
        Returns center coordinates. Can be used in conjunction with very large segments that cover the complete image.
    """
    def __init__(self, gt_loader, max_number_of_subjects_used=50):
        super(CenterSampler, self).__init__(gt_loader, max_number_of_subjects_used=max_number_of_subjects_used)
        self.gt_loader = gt_loader
        
    def iter_random_sample_coordinates(self, nb_sample_coordinates=100, epoch_identifier=None, thread_i=None):
        ## First we construct an array with the subject ids of the subjects that will be used
        subject_ids = np.arange(self.nb_subjects)
        # np.random.shuffle(subject_ids)
        max_number_of_subjects_used_ = min(self.max_number_of_subjects_used, nb_sample_coordinates)
        if max_number_of_subjects_used_ < self.nb_subjects:
            subject_ids = subject_ids[:int(max_number_of_subjects_used_)]
        ## Now we construct an array that tells us how many times each of these subjects will be used
        nb_times_subject_ids = np.ones_like(subject_ids) * (nb_sample_coordinates // len(subject_ids))
        nb_times_subject_ids[:int(nb_sample_coordinates % len(subject_ids))] += 1
        for subject_id, nb_times_subject_id in zip(subject_ids, nb_times_subject_ids):
            base_image, _wc, _o = self.base_image_loader.load(subject_id, epoch_identifier)
            for i in range(int(nb_times_subject_id)):
                yield (subject_id, [c // 2 for c in base_image.shape[:-1]])
        ## Watch out: if one requests more samples than available subjects, of the same subject the extra augmentation is only performed on SampleTransformer level!


class ForcedUniformCenterSampler(Sampler):
    """
        Returns center coordinates.
    """
    def __init__(self, base_image_loader, roi_loader=None, max_number_of_subjects_used=None, parallel_sample_creation=False):
        super(ForcedUniformCenterSampler, self).__init__(base_image_loader, roi_loader, max_number_of_subjects_used)
        if parallel_sample_creation == 0:
            self.nb_threads = 1
        elif parallel_sample_creation == 1:
            self.nb_threads = 2
        else:
            self.nb_threads = parallel_sample_creation
        self.subject_list = list(range(self.nb_subjects))
        self.sample_subject_list = []
        for i in range(10000 // self.nb_subjects):
            self.sample_subject_list += sample(self.subject_list, self.nb_subjects)

    def iter_random_sample_coordinates(self, nb_sample_coordinates=None, epoch_identifier=None, thread_i=None):
        thread_i = thread_i or 0
        seed(0)  # Due to the threading they seem to be using the same random module, and thus their seeds do not stay coherent --> we choose to permutate over and over according to seed(0)
        if nb_sample_coordinates is None:
            next_sample_subjects = self.sample_subject_list[:self.max_number_of_subjects_used]
            next_sample_subjects = sample(next_sample_subjects, len(next_sample_subjects))
            self.sample_subject_list = self.sample_subject_list[self.max_number_of_subjects_used:] + next_sample_subjects
            nb_sample_coordinates = self.max_number_of_subjects_used // self.nb_threads + 1 if self.max_number_of_subjects_used % self.nb_threads > thread_i else self.max_number_of_subjects_used // self.nb_threads
        else:
            next_sample_subjects = self.sample_subject_list[:nb_sample_coordinates * self.nb_threads]
            next_sample_subjects = sample(next_sample_subjects, len(next_sample_subjects))
            self.sample_subject_list = self.sample_subject_list[nb_sample_coordinates * self.nb_threads:] + next_sample_subjects
        for i in range(nb_sample_coordinates):
            subject_id = next_sample_subjects[thread_i + i * self.nb_threads]
            base_image, _wc, _o = self.base_image_loader.load(subject_id, epoch_identifier)
            yield (subject_id, [c // 2 for c in base_image.shape[:-1]])


class ForcedUniformCoordinateSampler(Sampler):
    """
        Returns random coordinates.
    """
    def __init__(self, base_image_loader, roi_loader=None, max_number_of_subjects_used=None, weighted_sampling_weights=None, parallel_sample_creation=False):
        super(ForcedUniformCoordinateSampler, self).__init__(base_image_loader, roi_loader, max_number_of_subjects_used)
        if parallel_sample_creation == 0:
            self.nb_threads = 1
        elif parallel_sample_creation == 1:
            self.nb_threads = 2
        else:
            self.nb_threads = parallel_sample_creation
        self.subject_list = list(range(self.nb_subjects))
        self.sample_subject_list = []
        for i in range(10000 // self.nb_subjects):
            self.sample_subject_list += sample(self.subject_list, self.nb_subjects)
        if weighted_sampling_weights:
            assert roi_loader is not None
            assert len(weighted_sampling_weights) == roi_loader.load(0)[0].shape[-1]
            weighted_sampling_weights = [weight / sum(weighted_sampling_weights) for weight in weighted_sampling_weights]
        self.weighted_sampling_weights = weighted_sampling_weights

    def iter_random_sample_coordinates(self, nb_sample_coordinates=None, epoch_identifier=None, thread_i=None):
        thread_i = thread_i or 0
        seed(0)  # Due to the threading they seem to be using the same random module, and thus their seeds do not stay coherent --> we choose to permutate over and over according to seed(0)
        if nb_sample_coordinates is None:
            next_sample_subjects = self.sample_subject_list[:self.max_number_of_subjects_used]
            next_sample_subjects = sample(next_sample_subjects, len(next_sample_subjects))
            self.sample_subject_list = self.sample_subject_list[self.max_number_of_subjects_used:] + next_sample_subjects
            number_of_subjects_used = self.max_number_of_subjects_used // self.nb_threads + 1 if self.max_number_of_subjects_used % self.nb_threads > thread_i else self.max_number_of_subjects_used // self.nb_threads
            factor = 1
        else:
            if nb_sample_coordinates > self.max_number_of_subjects_used:
                assert nb_sample_coordinates % self.max_number_of_subjects_used == 0
                number_of_subjects_used = self.max_number_of_subjects_used
                factor = nb_sample_coordinates // number_of_subjects_used
            else:
                number_of_subjects_used = nb_sample_coordinates
                factor = 1
            next_sample_subjects = self.sample_subject_list[:number_of_subjects_used * self.nb_threads]
            next_sample_subjects = sample(next_sample_subjects, len(next_sample_subjects))
            self.sample_subject_list = self.sample_subject_list[number_of_subjects_used * self.nb_threads:] + next_sample_subjects
        for i in range(number_of_subjects_used):
            subject_id = next_sample_subjects[thread_i + i * self.nb_threads]
            if self.roi_loader is not None:
                roi, _wc, _o = self.roi_loader.load(subject_id, epoch_identifier)
            else:
                base_image, _wc, _o = self.base_image_loader.load(subject_id, epoch_identifier)
                roi = np.ones_like(base_image, dtype=np.bool)
            possible_coordinates = [list(zip(*np.nonzero(roi[..., roi_i]))) for roi_i in range(roi.shape[-1])]
            if self.weighted_sampling_weights:
                roi_ids = np.random.choice(roi.shape[-1], size=factor, p=self.weighted_sampling_weights)
            else:
                total = np.sum([len(c) for c in possible_coordinates])
                roi_ids = np.random.choice(roi.shape[-1], size=factor, p=[len(c) / total for c in possible_coordinates])
            roi_ids, roi_counts = np.unique(roi_ids, return_counts=True)
            for roi_i, roi_count in zip(roi_ids, roi_counts):
                coordinate_ids = [*range(len(possible_coordinates[roi_i]))] * (roi_count // len(possible_coordinates[roi_i]))
                coordinate_ids += list(np.random.choice(len(possible_coordinates[roi_i]), size=roi_count % len(possible_coordinates[roi_i]), replace=False))
                for coordinate_id in coordinate_ids:
                    yield (subject_id, tuple(possible_coordinates[roi_i][coordinate_id]))

            
class ClassWeightedSampler(Sampler):
    ''' Samples per image a predefined fraction of each class. '''
    def __init__(self, gt_loader, roi_loader=None, nb_classes=2, class_weights=None, max_number_of_subjects_used=50):
        super(ClassWeightedSampler,self).__init__(gt_loader, roi_loader, max_number_of_subjects_used)
        self.gt_loader = gt_loader
        self.nb_classes = nb_classes
        self.class_weights = class_weights if class_weights is not None else [1./nb_classes]*nb_classes
        
    def iter_random_sample_coordinates(self,nb_sample_coordinates=100,epoch_identifier=None, thread_i=None):
        #select subjects from which to sample
        if self.max_number_of_subjects_used>self.nb_subjects:
            subject_ids = np.arange(self.nb_subjects)
        else:
            subject_ids = np.unique(np.random.choice(np.arange(self.nb_subjects),size=self.max_number_of_subjects_used,replace=False))
        nb_sample_coordinates_per_subject = max(1, int(round(nb_sample_coordinates/len(subject_ids))))
        for subject_id in subject_ids:
            gt,_wc,_o = self.gt_loader.load(subject_id,epoch_identifier)
            if self.roi_loader is not None:
                roi,_wc,_o = self.roi_loader.load(subject_id,epoch_identifier)
            else:
                roi = np.ones_like(gt,dtype=np.bool)
            for class_i in range(self.nb_classes):
                nb_sample_coordinates_per_subject_per_class = int(round(self.class_weights[class_i]*nb_sample_coordinates_per_subject))
                possible_coordinatess = np.nonzero(np.logical_and(gt==class_i,roi))
                if len(possible_coordinatess[0])>0: #sometimes an image does not contain a certain class
                    for i in np.random.randint(0,len(possible_coordinatess[0]),size=nb_sample_coordinates_per_subject_per_class):
                        yield (subject_id,(possible_coordinatess[0][i],possible_coordinatess[1][i],possible_coordinatess[2][i]))
                        
class ClassWeightedRestrictedSampler(Sampler):
    ''' 
      Samples per image a predefined fraction of each class. 
      The possible coordinateds for sampling are restricted to a bbox smaller than the input image size
      so that the output segment only contains voxels in the input image (no padding on the boundary)
     '''
    def __init__(self, gt_loader, segment_size, roi_loader=None, nb_classes=2, class_weights=None, max_number_of_subjects_used=50):
        super(ClassWeightedRestrictedSampler,self).__init__(gt_loader, roi_loader, max_number_of_subjects_used)
        self.gt_loader = gt_loader
        self.segment_size = segment_size
        self.nb_classes = nb_classes
        self.class_weights = class_weights if class_weights is not None else [1./nb_classes]*nb_classes
        
    def iter_random_sample_coordinates(self,nb_sample_coordinates=100,epoch_identifier=None, thread_i=None):
        #select subjects from which to sample
        if self.max_number_of_subjects_used>self.nb_subjects:
            #subject_ids = np.arange(self.nb_subjects)
            subject_ids = np.unique(np.random.choice(np.arange(self.nb_subjects),size=self.nb_subjects,replace=False))
        else:
            subject_ids = np.unique(np.random.choice(np.arange(self.nb_subjects),size=self.max_number_of_subjects_used,replace=False))
        nb_sample_coordinates_per_subject = max(1, int(round(nb_sample_coordinates/len(subject_ids))))
        for subject_id in subject_ids:
            gt,_wc,_o = self.gt_loader.load(subject_id,epoch_identifier)                     
            if self.roi_loader is not None:
                roi,_wc,_o = self.roi_loader.load(subject_id,epoch_identifier)
            else:
                roi = np.ones_like(gt,dtype=np.bool)
            img_size = gt.shape
            half_segment_size = [ss//2 for ss in self.segment_size]
            if self.segment_size[0] < img_size[0]:
              x_start = half_segment_size[0]
              x_end = img_size[0]-1 - half_segment_size[0] if self.segment_size[0]%2==1 else img_size[0]-1 - half_segment_size[0] + 1
            else: 
              x_start = img_size[0]//2
              x_end = img_size[0]//2
            
            if self.segment_size[1] < img_size[1]:
              y_start = half_segment_size[1]
              y_end = img_size[1]-1 - half_segment_size[1] if self.segment_size[1]%2==1 else img_size[1]-1 - half_segment_size[1] + 1
            else: 
              y_start = img_size[1]//2
              y_end = img_size[1]//2
              
            if self.segment_size[2] < img_size[2]:
              z_start = half_segment_size[2]
              z_end = img_size[2]-1 - half_segment_size[2] if self.segment_size[2]%2==1 else img_size[2]-1 - half_segment_size[2] + 1
            else: 
              z_start = img_size[2]//2
              z_end = img_size[2]//2
            sample_mask = np.zeros_like(gt)
            sample_mask[x_start:x_end+1, y_start:y_end+1, z_start:z_end+1,:] = 1
            nb_sample_coordinates_per_subject_last_class_left = 0
            for class_i in range(self.nb_classes):
                class_i = self.nb_classes-1 - class_i
                nb_sample_coordinates_per_subject_per_class = int(round(self.class_weights[class_i]*nb_sample_coordinates_per_subject)) + nb_sample_coordinates_per_subject_last_class_left
                gt_roi = np.logical_and(gt==class_i,roi)
                gt_roi_sample = np.logical_and(gt_roi,sample_mask)
                possible_coordinatess = np.nonzero(gt_roi_sample)
                if len(possible_coordinatess[0])==0:
                    nb_sample_coordinates_per_subject_last_class_left = nb_sample_coordinates_per_subject_per_class
                else: #sometimes an image does not contain a certain class
                    if nb_sample_coordinates_per_subject_per_class <= len(possible_coordinatess[0]):
                      nb_sample_coordinates_per_subject_per_class_actual = nb_sample_coordinates_per_subject_per_class  
                    else:
                      nb_sample_coordinates_per_subject_per_class_actual = len(possible_coordinatess[0])
                    nb_sample_coordinates_per_subject_last_class_left = nb_sample_coordinates_per_subject_per_class - nb_sample_coordinates_per_subject_per_class_actual
                    for i in np.random.randint(0,len(possible_coordinatess[0]),size=nb_sample_coordinates_per_subject_per_class_actual):
                        yield (subject_id,(possible_coordinatess[0][i],possible_coordinatess[1][i],possible_coordinatess[2][i]))

class GlobalClassWeightedSampler(Sampler):
    ''' Like uniform sampling, but with fixed over/under sampling per class. '''
    def __init__(self, gt_loader, roi_loader=None, nb_classes=2, class_weights=None, max_number_of_subjects_used=50):
        super(GlobalClassWeightedSampler,self).__init__(gt_loader, roi_loader, max_number_of_subjects_used)
        self.gt_loader = gt_loader
        self.nb_classes = nb_classes
        self.class_weights = np.array(class_weights if class_weights is not None else [1.]*nb_classes, np.float)
        
    def iter_random_sample_coordinates(self,nb_sample_coordinates=100,epoch_identifier=None, thread_i=None):
        #select subjects from which to sample
        if self.max_number_of_subjects_used>self.nb_subjects:
            subject_ids = np.arange(self.nb_subjects)
        else:
            subject_ids = np.unique(np.random.choice(np.arange(self.nb_subjects),size=self.max_number_of_subjects_used,replace=False))
        nb_sample_coordinates_per_subject = max(1, int(round(nb_sample_coordinates/len(subject_ids))))
        for subject_id in subject_ids:
            gt,_wc,_o = self.gt_loader.load(subject_id,epoch_identifier)
            if self.roi_loader is not None:
                roi,_wc,_o = self.roi_loader.load(subject_id,epoch_identifier)
            else:
                roi = np.ones_like(gt,dtype=np.bool)
            #number samples per class
            nb_samples_per_class_in_image = [np.sum(np.logical_and(gt==class_i,roi)) for class_i in range(self.nb_classes)]
            fraction_samples_to_make_per_class = (self.class_weights * nb_samples_per_class_in_image)/np.sum(self.class_weights* nb_samples_per_class_in_image)
            nb_samples_to_make_per_class = np.around(fraction_samples_to_make_per_class * nb_sample_coordinates_per_subject).astype(np.int)
            
            for class_i in range(self.nb_classes):
                possible_coordinatess = np.nonzero(np.logical_and(gt==class_i,roi))
                if len(possible_coordinatess[0])>0: #sometimes an image does not contain a certain class
                    for i in np.random.randint(0,len(possible_coordinatess[0]),size=nb_samples_to_make_per_class[class_i]):
                        yield (subject_id,(possible_coordinatess[0][i],possible_coordinatess[1][i],possible_coordinatess[2][i]))
