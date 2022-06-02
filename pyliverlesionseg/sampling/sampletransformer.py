'''
Created on May 24, 2017

Classes to transform samples deterministically (usually for preprocessing) and stochastically (usually for data augmentation).

@author: drobbe1
'''

import numpy as np
import scipy.ndimage
from .randomness import Randomness
import random
    
    
class SampleTransformer(object):

    """
    SampleTransformer is an abstract class.
    Subclasses that want to use the standard transform(), should call __init__ and implement _randomize(), _transform() and get_dimensions().
    """
    
    def __init__(self, randomness):
        """
        :param randomness:
        """
        self.randomness = randomness
        self.old_subject_id = None
        self.old_coordinates = None
        self.old_epoch_identifier = None
        
    def transform(self, sample, subject_id, coordinates, epoch_identifier):
        if (self.randomness>=Randomness.PerEpoch and epoch_identifier!=self.old_epoch_identifier) \
             or (self.randomness>=Randomness.PerSubject and subject_id!=self.old_subject_id) \
             or (self.randomness>=Randomness.PerSample and coordinates!=self.old_coordinates):
            self._randomize()
            self.old_subject_id = subject_id
            self.old_coordinates = coordinates
            self.old_epoch_identifier = epoch_identifier
        for sample_part_i, sample_part in enumerate(sample):
            if self.randomness == Randomness.PerSamplePart:
                self._randomize()
            if self.randomness==Randomness.PerModality:
                raise ValueError("For loop probably not correct.")
                for modality_i, _modality in enumerate(sample_part):
                    self._randomize()
                    sample = self._transform(sample, sample_part_i, modality_i)
            else:
                sample = self._transform(sample, sample_part_i)
        return sample

    def _transform(self,sample,sample_part_i,modality_i=None):
        raise NotImplementedError("This is an abstract class.")
    def get_dimensions(self,initial_dimensions):
        raise NotImplementedError("This is an abstract class.")


class AxesSwapper(SampleTransformer):
    def __init__(self, axes, randomness=1, probability=1):
        super(AxesSwapper, self).__init__(randomness)
        self.axes = axes
        self.probability = probability
        self.randomize = 1
        
    def _randomize(self):
        self.randomize = random.uniform(0, 1) < self.probability
    
    def _transform(self, sample, sample_part_i, modality_i=None):
        if self.randomize:
            if modality_i is None:
                sample[sample_part_i] = np.swapaxes(sample[sample_part_i], *self.axes)
            else:
                sample[sample_part_i][..., modality_i] = np.swapaxes(sample[sample_part_i][..., modality_i], *self.axes)
        return sample
    
    def get_dimensions(self, initial_dimensions):
        initial_dimensions[self.axes[0]], initial_dimensions[self.axes[1]] = initial_dimensions[self.axes[1]], initial_dimensions[self.axes[0]]
        return initial_dimensions
    
    
class IntensityTransformer(SampleTransformer):
    def __init__(self, randomness, mean_shift, std_shift, mean_scale, std_scale):
        super(IntensityTransformer, self).__init__(randomness)
        self.mean_shift = mean_shift if isinstance(mean_shift, (tuple, list)) else [mean_shift]
        self.std_shift = std_shift if isinstance(std_shift, (tuple, list)) else [std_shift]
        self.mean_scale = mean_scale if isinstance(mean_scale, (tuple, list)) else [mean_scale]
        self.std_scale = std_scale if isinstance(std_scale, (tuple, list)) else [std_scale]
        # state
        self.shift = None
        self.scale = None
        
    def _randomize(self):
        self.shift = [np.random.normal(mean_shift, std_shift) for mean_shift, std_shift in zip(self.mean_shift, self.std_shift)]
        self.scale = [np.random.normal(mean_scale, std_scale) for mean_scale, std_scale in zip(self.mean_scale, self.std_scale)]
    
    def _transform(self, sample, sample_part_i, modality_i=None):
        if modality_i is not None:
            sample[sample_part_i][..., modality_i] += self.shift[modality_i if len(self.shift) > 1 else 0]
            sample[sample_part_i][..., modality_i] *= self.scale[modality_i if len(self.scale) > 1 else 0]
        else:
            sample[sample_part_i] *= self.scale
            sample[sample_part_i] += self.shift
        return sample
            
    def get_dimensions(self,initial_dimensions):
        return initial_dimensions


class MetadataTransformer(SampleTransformer):
    
    def __init__(self, shifts, randomness=Randomness.PerSamplePart):
        super(MetadataTransformer,self).__init__(randomness)
        self.shifts_min_max = shifts
        self.shifts = None
        
    def _randomize(self):
        self.shifts = [np.random.uniform(s[0], s[1]) for s in self.shifts_min_max]
    
    def _transform(self, sample, sample_part_i):
        sample[sample_part_i] += self.shifts
        return sample
    
    def get_dimensions(self,initial_dimensions):
        return initial_dimensions
    

class ContrastTransformer(SampleTransformer):
    def __init__(self, randomness, mean_log_scale,std_log_scale):
        super(ContrastTransformer,self).__init__(randomness)
        self.mean_log_scale = mean_log_scale
        self.std_log_scale = std_log_scale
        #state
        self.scale = None
        
    def _randomize(self):
        self.scale = np.random.lognormal(self.mean_log_scale,self.std_log_scale)
    
    def _transform(self,sample,sample_part_i,modality_i=None):
        if modality_i is not None:
            raise ValueError("Raising contrast needs to be done on all modalities at the same time")
        else:
            contrast = sample[sample_part_i][:,:,:,1:] - sample[sample_part_i][:,:,:,0:1]
            #change contrast and add to the signal
            sample[sample_part_i][:,:,:,1:]+= (self.scale-1)*contrast
        return sample
    
    def get_dimensions(self,initial_dimensions):
        return initial_dimensions


class TimeShifter(SampleTransformer):
    def __init__(self, randomness, max_shift_earlier,max_shift_later):
        if max_shift_earlier<0:
            print("Warning: negative max_shift_earlier: the time series will always be delayed!")
        super(TimeShifter,self).__init__(randomness)
        self.max_shift_earlier = max_shift_earlier
        self.max_shift_later = max_shift_later
        #state
        self.shift = None
        
    def _randomize(self):
        self.shift = np.random.randint(-self.max_shift_earlier,self.max_shift_later+1)
    
    def _transform(self,sample,sample_part_i,modality_i=None):
        if modality_i is not None:
            raise ValueError("Time shifting needs to be done on all modalities at the same time")
        else:
            if self.shift>0: #make things later, shift to right
                sample[sample_part_i][:,:,:,self.shift:] = sample[sample_part_i][:,:,:,:-self.shift]
                sample[sample_part_i][:,:,:,:self.shift] = sample[sample_part_i][:,:,:,self.shift:self.shift+1]  #repeat first
            
            if self.shift<0: #make this earlier, shift to left
                sample[sample_part_i][:,:,:,:self.shift] = sample[sample_part_i][:,:,:,-self.shift:]
                sample[sample_part_i][:,:,:,self.shift:] = sample[sample_part_i][:,:,:,self.shift-1:self.shift]  #repeat last
        return sample
        
    def get_dimensions(self,initial_dimensions):
        return initial_dimensions


class ExtrapolateTimeSeries(SampleTransformer):
    '''
    Give a variable length timeseries, extrapolate the end to get a fixed length time series.
    '''
    def __init__(self, target_length, extrapolation_method):
        '''
        @param target_length:           how long should the time series be
        @param extrapolation_method:    how to extrapolate: 
                                            if a float is given, that number is used
                                            if "nearest", the last value is repeated
        '''
        super(ExtrapolateTimeSeries,self).__init__(Randomness.Never)
        self.target_length = target_length
        self.extrapolation_method = extrapolation_method

        
    def _randomize(self):
        pass
    
    def _transform(self,sample,sample_part_i,modality_i=None):
        if modality_i is not None:
            raise ValueError("ExtrapolateTimeSeries needs to be done on all modalities at the same time")
        else:
            new_part = np.empty( tuple(sample[sample_part_i].shape[:-1]) + (self.target_length,), dtype=np.float32 )
            #First extrapolate
            if type(self.extrapolation_method) is float:
                new_part[:] = self.extrapolation_method
            elif self.extrapolation_method=="nearest":
                new_part[:,:,:,:] = sample[sample_part_i][:,:,:,-1:]
            else:
                raise ValueError("Unknown extrapolation_method.")
            #Fill in values
            new_part[:,:,:,:sample[sample_part_i].shape[-1]] = sample[sample_part_i]
            sample[sample_part_i] = new_part
        return sample
    
    def get_dimensions(self,initial_dimensions):
        return (initial_dimensions[0],initial_dimensions[1],initial_dimensions[2],self.target_length)
        
    
class RemoveModality(SampleTransformer):
    '''
    Randomly remove (part of) one of the modalities.
    TODO: implement partial removal
    '''
    def __init__(self, randomness, probabilities, removed_values):
        '''
        @param probabilities: probability for each modality to be absent
        @param removed_values: value that a removed modality will have 
        '''
        super(RemoveModality,self).__init__(randomness)
        self.probabilities = probabilities
        self.removed_values = removed_values
        #state
        self.removed = None
        
    def _randomize(self):
        self.removed = [random.uniform(0,1)<probability for probability in self.probabilities]
    
    def _transform(self,sample,sample_part_i,modality_i=None):
        if modality_i is not None:
            if self.removed[modality_i]:
                sample[sample_part_i][:,:,:,modality_i] = self.removed_values[modality_i]
        else:
            for modality_i,removed in enumerate(self.removed):
                if removed:
                    sample[sample_part_i][:,:,:,modality_i] = self.removed_values[modality_i]
        return sample
    
    def get_dimensions(self,initial_dimensions):
        return initial_dimensions


class SimulateInvalidValuesMotionCorrection(SampleTransformer):
    '''
    Randomly part of the images
    '''
    def __init__(self, randomness, probabilities, removed_values):
        '''
        @param probabilities: probability for each modality to be absent
        @param removed_values: value that a removed modality will have 
        '''
        super(RemoveModality,self).__init__(randomness)
        self.probabilities = probabilities
        self.removed_values = removed_values
        #state
        self.removed = None
        
    def _randomize(self):
        self.removed = [random.uniform(0,1)<probability for probability in self.probabilities]
    
    def _transform(self,sample,sample_part_i,modality_i=None):
        if modality_i is not None:
            if self.removed[modality_i]:
                sample[sample_part_i][:,:,:,modality_i] = self.removed_values[modality_i]
        else:
            for modality_i,removed in enumerate(self.removed):
                if removed:
                    sample[sample_part_i][:,:,:,modality_i] = self.removed_values[modality_i]
        return sample
    
    def get_dimensions(self,initial_dimensions):
        return initial_dimensions    
    
    
class Flipper(SampleTransformer):        
    def __init__(self, randomness=Randomness.PerSample, flipxyz_probabilities=(0.5, 0, 0)):
        super(Flipper, self).__init__(randomness)
        self.flipxyz_probabilities = flipxyz_probabilities
        #state
        self.flipxyz = None
        
    def _randomize(self):
        self.flipxyz = [np.random.binomial(1, flip_p) for flip_p in self.flipxyz_probabilities]
    
    def _transform(self,sample,sample_part_i,modality_i=None):
        directionality = tuple(-1 if flip else 1 for flip in self.flipxyz)
        if modality_i is not None:
            sample[sample_part_i][:, :, :, modality_i]= sample[sample_part_i][::directionality[0], ::directionality[1], ::directionality[2], modality_i]
        else:
            sample[sample_part_i] = np.array(sample[sample_part_i][::directionality[0], ::directionality[1], ::directionality[2], :])
        return sample
    
    def get_dimensions(self,initial_dimensions):
        return initial_dimensions


class LabelFlipper(Flipper):
    def __init__(self,flipper,flip_pairs=[]):
        self.flipper=flipper
        self.flip_pairs=flip_pairs

    def __getattr__(self,name):
        if name=="flipper" or name=="flip_pairs":                 #necessary check to avoid endless loops when pickling
            raise AttributeError(name)
        return getattr(self.flipper,name)
           
    def __setattr__(self,name,value):
        #There is an assymmetry in __getattr__ and __setattr__,
        # the former is only called when an attribute does not exist, the other is always called
        #Hence we explicitly check whether we want to store an attribute locally or in the elastic_transformer
        if name=="flipper" or name=="flip_pairs":
            self.__dict__[name] = value
            return object.__setattr__(self, name, value)
        return setattr(self.flipper,name,value)
            
    def _transform(self,sample,sample_part_i,modality_i=None):
        directionality = tuple(-1 if flip else 1 for flip in self.flipxyz)
        if modality_i is not None:
            sample[sample_part_i][:,:,:,modality_i]= sample[sample_part_i][::directionality[0],::directionality[1],::directionality[2],modality_i]
            if directionality[0]==-1:
                for a,b in self.flip_pairs:
                    mask_a = (sample[sample_part_i][:,:,:,modality_i]==a)
                    mask_b = (sample[sample_part_i][:,:,:,modality_i]==b)
                    sample[sample_part_i][:,:,:,modality_i][mask_b]=a 
                    sample[sample_part_i][:,:,:,modality_i][mask_a]=b 
        else:
            sample[sample_part_i]= np.array( sample[sample_part_i][::directionality[0],::directionality[1],::directionality[2],:] )
            if directionality[0]==-1:
                for a,b in self.flip_pairs:
                    mask_a = (sample[sample_part_i]==a)
                    mask_b = (sample[sample_part_i]==b)
                    sample[sample_part_i][mask_b]=a
                    sample[sample_part_i][mask_a]=b
        return sample
    

def _remove_border(I, border_dimensions):
    ((cx0, cx1), (cy0, cy1), (cz0, cz1),) = border_dimensions
    sx,sy,sz=I.shape[:-1]
    return I[cx0:sx-cx1,cy0:sy-cy1,cz0:sz-cz1,:]
  
        
class Cropper(SampleTransformer):
    def __init__(self, crop):
        '''
        crop = ((cropx0,cropx1),(cropy0,cropy1),(cropz0,cropz1),)
        '''
        super(Cropper,self).__init__(Randomness.Never)
        self.crop = crop
        
    def _randomize(self):
        pass
    
    def _transform(self,sample,sample_part_i,modality_i=None):
        if modality_i is not None:
            raise ValueError("Cropper cannot work on the individual modalities.")
        sample[sample_part_i]= _remove_border(sample[sample_part_i], self.crop)
        return sample
    
    def get_dimensions(self,initial_dimensions):
        return tuple(initial_dimensions[d]-self.crop[d][0]-self.crop[d][1] for d in range(3)) + (initial_dimensions[-1],)


class MultiScaleTransformer(SampleTransformer):        
    def __init__(self, scales):
        '''
        scales = [(subsample_factor,segment_size)] = [(sfx,sfy,sfz),(ssx,ssy,ssz)]
        '''
        self.scales = scales
        
    def transform(self, sample, subject_id, coordinates, epoch_identifier):
        sample_new = []
        for sample_part in sample:
            #Do the subsampling/cropping on this sample_part
            for (sfx,sfy,sfz),(ssx,ssy,ssz) in self.scales:
                sample_part_subsampled = scipy.ndimage.uniform_filter(sample_part,(sfx,sfy,sfz,1))[sfx//2::sfx,sfy//2::sfy,sfz//2::sfz,:]
                border_x,border_y,border_z = (np.array(sample_part_subsampled.shape[:-1]) - np.array((ssx,ssy,ssz)))//2
                sample_new.append( _remove_border(sample_part_subsampled, ((border_x,border_x),(border_y,border_y),(border_z,border_z))) )
        #Replace in place
        [sample.pop() for _ in range(len(sample))]
        [sample.append(sample_part) for sample_part in sample_new]
        return sample
        
    def get_dimensions(self,initial_dimensions):
        return initial_dimensions
