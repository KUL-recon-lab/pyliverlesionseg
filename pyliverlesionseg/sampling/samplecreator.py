'''
Created on May 23, 2017

Contains the required classes for the generation of the training and testing samples, including data augmentation.
    
A sample is a single example that can be given to the classifier for training or testing.
To allow for classifiers that have multiple pathways,
    a sample is a list of 4D numpy arrays for the given coordinates.
For the numpy arrays, the order is assumed to be x,y,z,c

    
@author: drobbe1
'''


import numpy as np
import nibabel as nib
import itertools
import scipy.ndimage


def flatten(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))
    

class SampleCreator(object):
    '''
    Abstract class that creates training/testing samples.
    '''
    
    def get_sample(self, subject_id, coordinates, epoch_identifier):
        '''
        Return a sample, which is a list of 4D numpy arrays for the given coordinates
        For the numpy arrays, the order is assumed to be x,y,z,c
        '''
        raise NotImplementedError("This is an abstract class.")
    
    def get_dimensions(self):
        raise NotImplementedError("This is an abstract class.")
    

class Concat(SampleCreator):
    '''
    Concatenate several SampleCreators, giving a sample that has more 4D arrays.
    '''
    def __init__(self, sample_creators):
        self.sample_creators = sample_creators
        
    def get_sample(self, subject_id, coordinates, epoch_identifier):
        return flatten([sample_creator.get_sample(subject_id, coordinates, epoch_identifier) for sample_creator in self.sample_creators])
    
    def get_dimensions(self):
        return flatten([sample_creator.get_dimensions() for sample_creator in self.sample_creators])
    

class MultimodalConcat3D(SampleCreator):    
    '''
    Concatenate several SampleCreators to obtain 1 large 4D array. 
    '''
    def __init__(self, sample_creators):
        self.sample_creators = sample_creators
#        self.spatial_size = self.sample_creators[0].get_dimensions()[0][:-1]
#        #Sanity check
#        all_dimensions = [sample_creator.get_dimensions() for sample_creator in self.sample_creators]
#        all_spatial_sizes = [ sample_part_dim[:-1] for sample_part_dim in flatten(all_dimensions)]
#        for spatial_size in all_spatial_sizes:
#            if not self.spatial_size==spatial_size: # np.allclose(self.spatial_size,spatial_size)
#                raise ValueError("Not all sample_creators have the same spatial size. Impossible to do a multimodal concat.")
        
    def get_sample(self, subject_id, coordinates, epoch_identifier):
        sample_parts = flatten([sample_creator.get_sample(subject_id, coordinates,epoch_identifier) for sample_creator in self.sample_creators])
        sample = [ np.concatenate([sample_part for sample_part in sample_parts],axis=-1) ]
        return sample

    def get_dimensions(self):
        return [ self.spatial_size + (len(self.sample_creators),) ]


class FullImageAsFixedSizeSegment(SampleCreator):
    '''
    Returns a fixed size segment with the full image in its center.
    Nb, in case the center is a float, round down.
    
    !!! Quite problematic in testing phase 
        --> therefore you should use combination of FullImageAsFixedSizeSegmentCropper (ImageTransformer) and FullImage (SampleCreator)
    '''
    def __init__(self, image_loader, segment_size, default_value=0):
        self.image_loader = image_loader
        self.segment_size = tuple(segment_size)
        self.default_value = default_value
        self.prev_subject_id = None
        self.prev_I = None
                
    def get_sample(self, subject_id, coordinates, epoch_identifier):                    
        if self.prev_subject_id == subject_id:
            I = self.prev_I
        else:
            I, _wc, _o = self.image_loader.load(subject_id, epoch_identifier)
            self.prev_subject_id = subject_id
            self.prev_I = I
        S_size = self.segment_size + (I.shape[-1],)
        S = self.default_value * np.ones(S_size, dtype=np.float32)
        idx_I = [slice(None)] * I.ndim
        idx_S = [slice(None)] * S.ndim
        for i, (d_I, d_S) in enumerate(zip(I.shape[:-1], S_size[:-1])):
            _ = abs(d_I - d_S) // 2
            if d_I > d_S:
                idx_I[i] = slice(_, _ + d_S)
            else:
                idx_S[i] = slice(_, _ + d_I)
        S[tuple(idx_S)] = I[tuple(idx_I)]
        return [S]
        
    def get_dimensions(self):
        old_subject_id = self.image_loader.old_subject_id
        old_epoch_identifier = self.image_loader.old_epoch_identifier
        old_img = self.image_loader.old_img
        for i in range(len(self.image_loader.image_paths_per_subject)):
            if i == 0:
                n_features = self.image_loader.load(i)[0].shape[-1]
            else:
                if n_features != self.image_loader.load(i)[0].shape[-1]:
                    n_features = None
                    break
        self.image_loader.old_subject_id = old_subject_id
        self.image_loader.old_epoch_identifier = old_epoch_identifier
        self.image_loader.old_img = old_img
        return [self.segment_size + (n_features,)]
    
    
class FullImage(SampleCreator):
    '''
    Returns the full image as a segment.
    For example: 
        To use in conjunction with FullImageAsFixedSizeSegmentCropper (ImageTransformer).
        Job of FullImageAsFixedSizeSegmentCropper: transforms the image to a fixed size segment with the full image in its center.
        Job of this one: make it a sample.
    '''
    def __init__(self, image_loader, binary_threshold=None):
        self.image_loader = image_loader
        self.prev_subject_id = None
        self.prev_I = None
        self.binary_threshold = binary_threshold
                
    def get_sample(self, subject_id, coordinates, epoch_identifier):
        if self.prev_subject_id == subject_id:
            I = self.prev_I
        else:
            I, _wc, _o = self.image_loader.load(subject_id, epoch_identifier)
            self.prev_subject_id = subject_id
            self.prev_I = I
        if self.binary_threshold is not None:
            I = np.ones_like(I) * (I > self.binary_threshold)
        return [I.copy()]
        
    def get_dimensions(self):
        old_subject_id = self.image_loader.old_subject_id
        old_epoch_identifier = self.image_loader.old_epoch_identifier
        old_img = self.image_loader.old_img
        for i in range(len(self.image_loader.image_paths_per_subject)):
            I_size = self.image_loader.load(i)[0].shape
            if i == 0:
                output_size = I_size
            else:
                for j, d in enumerate(I_size):
                    if output_size[j] != d:
                        output_size[j] = None
        self.image_loader.old_subject_id = old_subject_id
        self.image_loader.old_epoch_identifier = old_epoch_identifier
        self.image_loader.old_img = old_img
        return [tuple(output_size)]
     
        
class ExtractSegment3D2(SampleCreator):

    def __init__(self, image_loader, segment_size, subsample_factor=(1, 1, 1), default_value=0, mask=False, center_sampling=False):
        self.image_loader = image_loader
        self.segment_size = tuple(segment_size)
        self.subsample_factor = tuple(subsample_factor)
        self.default_value = default_value
        self.prev_subject_id = None
        self.prev_I = None
        self.mask = mask
        self.center_sampling = center_sampling
                
    def get_sample(self, subject_id, coordinates, epoch_identifier):                    
        if self.prev_subject_id == subject_id:
            I = self.prev_I
        else:
            I, _wc, _o = self.image_loader.load(subject_id, epoch_identifier)
            self.prev_subject_id = subject_id
            self.prev_I = I
        coordinates = coordinates if not self.center_sampling else [s // 2 for s in I.shape[:-1]]
        S_size = self.segment_size + (I.shape[-1],)
        S = self.default_value * np.ones(S_size, dtype=np.float32)
        idx_I = [slice(None)] * I.ndim
        idx_S = [slice(None)] * S.ndim
        for i, (d_I, d_S, c, s_f) in enumerate(zip(I.shape[:-1], S_size[:-1], coordinates, self.subsample_factor)):
            n_left_I = c
            n_right_I = d_I - c - 1
            n_left_S = d_S // 2
            n_right_S = d_S // 2
            if d_S % 2 == 0:
                n_right_S -= 1 
            if n_left_I < n_left_S * s_f:
                n = n_left_I // s_f
                start_S = d_S // 2 - n
                start_I = c - n * s_f
            else:
                start_S = 0
                start_I = c - n_left_S * s_f           
            if n_right_I < n_right_S * s_f:
                n = n_right_I // s_f
                end_S = d_S // 2 + n
                end_I = c + n * s_f
            else:
                end_S = d_S - 1
                end_I = c + n_right_S * s_f  
            idx_I[i] = slice(start_I, end_I + 1, s_f)
            idx_S[i] = slice(start_S, end_S + 1)
        if self.mask:
            S[tuple(idx_S)] = 1
        else:
            S[tuple(idx_S)] = I[tuple(idx_I)]
        return [S]
        
    def get_dimensions(self):
        old_subject_id = self.image_loader.old_subject_id
        old_epoch_identifier = self.image_loader.old_epoch_identifier
        old_img = self.image_loader.old_img
        for i in range(len(self.image_loader.image_paths_per_subject)):
            if i == 0:
                n_features = self.image_loader.load(i)[0].shape[-1]
            else:
                if n_features != self.image_loader.load(i)[0].shape[-1]:
                    n_features = None
                    break
        self.image_loader.old_subject_id = old_subject_id
        self.image_loader.old_epoch_identifier = old_epoch_identifier
        self.image_loader.old_img = old_img
        return [self.segment_size + (n_features,)]
    
    
class ExtractSegment3D(SampleCreator):
    '''
    Extracts 3D segments from images.
    get_sample() will give for each subject_id and coordinate, a sample with a 3D volume around that coordinate.
    If the image loader returns a 4D volume, all channels will be returned.
    If wanted, it can extract a subvolume of the subsampled image (useful for multiscale processing).
    '''
    
    def __init__(self, image_loader, segment_size,subsample_factor=(1,1,1),coordinate_transformer=None,extrapolation_method=0):
        '''
        @param image_loader:         deepvoxnet.sampling.imageloader
        @param segment_size:         size of the segment to extract
        @param subsample_factor:     optional, subsampling factor in x, y and z dimension
        @param extrapolation_method: the value for the part of the sample that is outside the image domain
                                            currently only floating point values supported
                                            future: nearest, mirror
        '''
        self.image_loader = image_loader
        self.segment_size = tuple(segment_size)
        self.half_segment_size = tuple(s//2 for s in segment_size)
        self.subsample_factor = tuple(subsample_factor)
        self.coordinate_transformer = coordinate_transformer
        self.extrapolation_method = extrapolation_method
        #Caching
        self.prev_subject_id = None
        self.prev_I = None
        self.prev_full_size=None
        #Courtesy
        if not all([s%2 for s in segment_size]):
            raise ValueError("Only odd segment sizes are supported.")
        if not all([s%2 for s in subsample_factor]):
            raise ValueError("Only odd subsample factors are supported.")
                
    def get_sample(self, subject_id, coordinates, epoch_identifier):
        sfx,sfy,sfz = self.subsample_factor
        if self.prev_subject_id == subject_id:
            I = self.prev_I
            full_size = self.prev_full_size
        else:
            I,_wc,_o = self.image_loader.load(subject_id,epoch_identifier)
            if len(I.shape)==3:
                I = I[...,np.newaxis]
            full_size = I.shape
            I = scipy.ndimage.uniform_filter(I,(sfx, sfy, sfz, 1))[sfx//2::sfx, sfy//2::sfy, sfz//2::sfz, :] #remains float32
            self.prev_subject_id = subject_id
            self.prev_I = I
            self.prev_full_size = full_size
        if self.coordinate_transformer is not None:
            if self.coordinate_transformer=="flipx":
                coordinates = (coordinates[0],
                               full_size[1]-coordinates[1],
                               coordinates[2])
            else:
                raise ValueError("This coordinate transformer is not yet supported.")      
        Ix,Iy,Iz,nb_channels=I.shape
        ssx,ssy,ssz = self.segment_size
        hssx,hssy,hssz = self.half_segment_size
        x,y,z = coordinates
        #account for subsampling!
        x//=sfx
        y//=sfy
        z//=sfz
        segment = np.zeros(self.segment_size+(nb_channels,),dtype=np.float32)
        segment[:] = self.extrapolation_method
        segment[-min(0,x-hssx):ssx+min(0,Ix-(x+hssx+1)),
                -min(0,y-hssy):ssy+min(0,Iy-(y+hssy+1)),
                -min(0,z-hssz):ssz+min(0,Iz-(z+hssz+1)),
                : ] = \
                        I[max(0,x-hssx):min(Ix,x+hssx+1),
                          max(0,y-hssy):min(Iy,y+hssy+1),
                          max(0,z-hssz):min(Iz,z+hssz+1),
                          :]
        return [segment]
        
    def get_dimensions(self):
        return [self.segment_size + (1,) ]


class Metadata(SampleCreator):
    '''
    Add metadata from a numpy array
    '''
    def __init__(self, array):
        '''
        @param array:    metadata, 2D: one row per subject.
        '''
        if len(array.shape)!=2:
            raise ValueError("The array should be 2D: one row per subject.")
        if array.dtype!=np.float32:
            print("Warning: Metadata will be cast to float32.")
        self.array = array.astype(np.float32)
        
    def get_sample(self, subject_id, coordinates, epoch_identifier):
        return [ np.array( [[[self.array[subject_id, :]]]] ) ]
    
    def get_dimensions(self):
        return [ (1,1,1) + (self.array.shape[1],) ] 


class Metadata4D(SampleCreator):
    '''
    Add metadata from a list of numpy arrays
    '''
    def __init__(self, metadata):
        '''
        @param metadata:    list with oen array per subject, x, y, z, feature
        '''
        for i,array in enumerate(metadata):
            if len(array.shape)!=4:
                raise ValueError("Each array should be 4D: x, y, z, feature.")
            if array.dtype!=np.float32:
                print("Warning: Metadata will be cast to float32.")
            metadata[i] = array.astype(np.float32)
        self.metadata = metadata
        
    def get_sample(self, subject_id, coordinates, epoch_identifier):
        return [ self.metadata[subject_id] ]
    
    def get_dimensions(self):
        return [ self.metadata[0].shape ] 
        
        
class ArrayConcat(SampleCreator):    
    '''
    Concatenate the output of several SampleCreators in an existing or new dimension. 
    '''
    def __init__(self, sample_creators, axis, new_axis=False):
        self.sample_creators = sample_creators
        self.axis = axis
        self.new_axis = new_axis
        
    def get_sample(self, subject_id, coordinates, epoch_identifier):
        sample_parts = flatten([sample_creator.get_sample(subject_id, coordinates, epoch_identifier) for sample_creator in self.sample_creators])
        if self.new_axis:
            sample = [np.stack([sample_part for sample_part in sample_parts], axis=self.axis)]
        else:
            sample = [np.concatenate([sample_part for sample_part in sample_parts], axis=self.axis)]
        return sample

    def get_dimensions(self):
        if self.new_axis:
            new_dimension_size = len(self.all_sizes)
            new_size = self.all_sizes[0][:self.axis] + [new_dimension_size] + self.all_sizes[0][self.axis:]
        else:
            new_dimension_size = 0
            for s in self.all_sizes:
                if s[self.axis] is None:
                    new_dimension_size = None
                    break
                else:
                    new_dimension_size += s[self.axis]
            new_size = [s for s in self.all_sizes[0]]
            new_size[self.axis] = new_dimension_size
        return [tuple(new_size)]


class ArrayMultiply(SampleCreator):
    '''
    Multiply the output of several SampleCreators.
    '''

    def __init__(self, sample_creators):
        self.sample_creators = sample_creators

    def get_sample(self, subject_id, coordinates, epoch_identifier):
        sample_parts = flatten([sample_creator.get_sample(subject_id, coordinates, epoch_identifier) for sample_creator in self.sample_creators])
        sample = [np.multiply(*sample_parts)]
        return sample

    def get_dimensions(self):
        return self.sample_creators[0].get_dimensions()

    
class TransformSamples(SampleCreator):
    '''
    Transform samples.
    '''
    def __init__(self, sample_creator, transformers):
        '''
        @param sample_creator:    the input sample_creator, whose samples need to be transformed
        @param transformers:    a list of deepvoxnet.sampling.sampletransformer
        '''
        self.sample_creator = sample_creator
        #courtesy
        if not isinstance(transformers,(list,tuple)):
            transformers = [transformers]
        self.transformers = transformers
        self.active = True
        
    def get_sample(self, subject_id, coordinates, epoch_identifier):
        sample = self.sample_creator.get_sample(subject_id, coordinates, epoch_identifier)
        if self.active:
            for transformer in self.transformers:
                sample = transformer.transform(sample, subject_id, coordinates, epoch_identifier)
        return sample
    
    def get_dimensions(self):
        [dimensions] = self.sample_creator.get_dimensions()
        for t in self.transformers:
            dimensions = t.get_dimensions(dimensions)
        return [dimensions]
