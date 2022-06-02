'''
Created on May 24, 2017

Transform complete images.

@author: drobbe1
'''


import numpy as np
from .randomness import Randomness

    
class ImageTransformer(object):
    """
        ImageTransformer is an abstract class.
        Subclasses that want to use the standard transform(), should call __init__ and implement _randomize(), _transform() and get_dimensions().
    """
    def __init__(self, randomness):
        self.randomness = randomness
        self.old_epoch_identifier = None
        self.old_subject_id = None
            
    def transform(self, I, subject_id, epoch_identifier):
        if (self.randomness >= Randomness.PerEpoch and epoch_identifier != self.old_epoch_identifier) or (self.randomness >= Randomness.PerSubject and subject_id != self.old_subject_id):
            self._randomize(I)
            self.old_subject_id = subject_id
            self.old_epoch_identifier = epoch_identifier
        for modality_i in range(I.shape[-1]):
            if self.randomness == Randomness.PerModality:
                self._randomize(I)
            self._transform(I, modality_i)
        return I


class IntensityTransformer(ImageTransformer):
    def __init__(self, randomness, mean_shift,std_shift, mean_scale,std_scale):
        super(IntensityTransformer,self).__init__(randomness)
        self.mean_shift = mean_shift
        self.std_shift = std_shift
        self.mean_scale = mean_scale
        self.std_scale = std_scale
        
    def _randomize(self, _I):
        self.shift = np.random.normal(self.mean_shift,self.std_shift)
        self.scale = np.random.normal(self.mean_scale,self.std_scale)
    
    def _transform(self,I, modality_i):
        I[:,:,:,modality_i]*= self.scale
        I[:,:,:,modality_i]+= self.shift
        return I
    
    def get_dimensions(self,initial_dimensions):
        return initial_dimensions


class Cropper(ImageTransformer):
    '''
    Crop the image.
    TODO: it's probably cleaner to really crop the image, but imageloader does not yet support that.
     Hence we set everything outside the cropped part to zero.
    '''
    
    def __init__(self, randomness, transform_probability, max_crop_x0,max_crop_x1,max_crop_y0,max_crop_y1,max_crop_z0,max_crop_z1):
        '''
        @param transform_probability: what percentage of images needs to be cropped
        @param : to what exent needs each dimension to be cropped maximally; two crops per dimension (low and high)
        '''
        super(Cropper,self).__init__(randomness)
        self.transform_probability=transform_probability
        self.max_crop = (max_crop_x0,max_crop_x1,max_crop_y0,max_crop_y1,max_crop_z0,max_crop_z1)
        
    def _randomize(self, I):
        
        if (random.uniform(0,1)<self.transform_probability):
            self.mask_set_to_zero = np.ones(I.shape[:3],np.bool)
            #Make sure there is still some part of the image left, and maximally crop half of the image
            Ix,Iy,Iz,_Im = I.shape
            max_crop_for_I = np.minimum(self.max_crop, 
                                        np.array([Ix//2,Ix//2,Iy//2,Iy//2,Iz//2,Iz//2]) ).astype(np.int)
            crops = [random.randint(0,max_crop_for_I[i])
                      for i in range(6)]
            
            self.mask_set_to_zero[crops[0]:Ix-crops[1],
                                  crops[2]:Iy-crops[3],
                                  crops[4]:Iz-crops[5],
                                  ] = 0
        else:
            self.mask_set_to_zero = np.zeros(I.shape[:3],np.bool)
            
    def _transform(self,I, modality_i):
        I[:,:,:,modality_i][self.mask_set_to_zero] = 0
        return I
    
    def get_dimensions(self,initial_dimensions):
        return initial_dimensions


class FullImageAsFixedSizeSegmentCropper(ImageTransformer):
    '''
    Returns a fixed size segment with the full image in its center (in case the center is a float, round down).
    '''
    
    def __init__(self, segment_size, default_value=0):
        self.randomness = 1
        self.segment_size = tuple(segment_size)
        self.default_value = default_value
        
    def transform(self, I, subject_id, epoch_identifier):
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
        return S
        
    def get_dimensions(self, initial_dimensions):
        return self.segment_size + (None,)


class FullImageCropper(ImageTransformer):
    '''
    Returns a fixed crop from an image.
    '''

    def __init__(self, x_min=None, x_max=None, y_min=None, y_max=None, z_min=None, z_max=None):
        self.randomness = 1
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max

    def transform(self, I, subject_id, epoch_identifier):
        return I[self.x_min:self.x_max, self.y_min:self.y_max, self.z_min:self.z_max, :]

    def get_dimensions(self, initial_dimensions):
        dimensions = []
        for d, (i, o) in zip(initial_dimensions[:-1], [(self.x_min, self.x_max), (self.y_min, self.y_max), (self.z_min, self.z_max)]):
            if i and o:
                dimensions.append(o - i)
            elif i:
                dimensions.append(d - i)
            elif o:
                dimensions.append(o)
            else:
                dimensions.append(d)
        return tuple(dimensions) + (initial_dimensions[-1],)
   
    
class FixedIntensityTransformer(ImageTransformer):
    """
        Perform a series of fixed intensity transformations in the following order:
            - clipping
            - shifting
            - scaling
    """
    def __init__(self, shift=0, scale=1, min_clip=-float('inf'), max_clip=float('inf'), ignore_value=None):
        super(FixedIntensityTransformer, self).__init__(Randomness.Never)
        if not isinstance(shift, (list, tuple)):
            shift = [shift]
        self.shift = shift
        if not isinstance(scale, (list, tuple)):
            scale = [scale]
        self.scale = scale
        if not isinstance(min_clip, (list, tuple)):
            min_clip = [min_clip]
        self.min_clip = min_clip
        if not isinstance(max_clip, (list, tuple)):
            max_clip = [max_clip]
        self.max_clip = max_clip
        if not isinstance(ignore_value, (list, tuple)):
            ignore_value = [ignore_value]
        self.ignore_value = np.array(ignore_value)
        
    def transform(self, I, subject_id, epoch_identifier):
        I = np.clip(I, self.min_clip, self.max_clip)
        I_ = I.copy()
        I = I + self.shift
        I = I * self.scale
        I[I_ == self.ignore_value] = I_[I_ == self.ignore_value]
        return I
    
    def get_dimensions(self,initial_dimensions):
        return initial_dimensions


class GaussianNoiseTransformer(ImageTransformer):
    def __init__(self, randomness, sigma_range=[(0, 0.01)], transform_probability=1.0):
        super(GaussianNoiseTransformer, self).__init__(randomness)
        if not isinstance(sigma_range[0], (tuple, list)):
            sigma_range = [sigma_range]
        self.sigma_range = sigma_range
        self.transform_probability = transform_probability
        self.sigma = [0 for _ in self.sigma_range]
        
    def _randomize(self, I):
        if random.uniform(0, 1) < self.transform_probability:
            self.sigma = [random.uniform(*sigma_range_) for sigma_range_ in self.sigma_range]
        else:
            self.sigma = [0 for _ in self.sigma_range]

    def _transform(self, I, modality_i):
        I[:, :, :, modality_i] += np.random.normal(0, self.sigma[0 if len(self.sigma) == 1 else modality_i], I.shape[:3])

    def get_dimensions(self,initial_dimensions):
        return initial_dimensions
    
    
from . import randomdeformation
import random
class ElasticTransformer(ImageTransformer):
    def __init__(self, randomness,
                 deformation_MAD_range = (0,4),
                 smooth_sigma_range = (30,50),
                 transform_probability=1.0,
                 voxel_size = None,
                 max_smooth_sigma_wo_interpolation = 10
                 ):
        '''
            deformation_MAD:      expected mean abs of deformation field, can be a tuple
            smooth_sigma:         spatial smoothing, can be a tuple 
        '''
        super(ElasticTransformer,self).__init__(randomness)
        self.deformation_MAD_range = deformation_MAD_range
        self.smooth_sigma_range = smooth_sigma_range
        self.transform_probability = transform_probability
        self.voxel_size = voxel_size
        self.max_smooth_sigma_wo_interpolation = max_smooth_sigma_wo_interpolation
        
    def _randomize(self, I):
        shape = I.shape[:3]
        self.do_deformation = (random.uniform(0,1)<self.transform_probability)
        if self.do_deformation:
            self.deformation_MAD = random.uniform(*self.deformation_MAD_range)
            self.smooth_sigma = random.uniform(*self.smooth_sigma_range)
            self.deformation_field = randomdeformation.get_random_deformation(shape, 
                 self.deformation_MAD, self.smooth_sigma,
                 voxel_size=self.voxel_size, max_smooth_sigma_wo_interpolation=self.max_smooth_sigma_wo_interpolation)

    def _transform(self,I, modality_i):
        if self.do_deformation:
            I[:,:,:,modality_i] = randomdeformation.apply_deformation(I[:,:,:,modality_i], self.deformation_field)

    def get_dimensions(self,initial_dimensions):
        return initial_dimensions


class ElasticGroundTruthTransformer(ImageTransformer):
    def __init__(self,elastic_transformer):
        self.elastic_transformer=elastic_transformer

    def __getattr__(self,name):
        if name=="elastic_transformer":                 #necessary check to avoid endless loops when pickling
            raise AttributeError(name)
        return getattr(self.elastic_transformer,name)
           

    def __setattr__(self,name,value):
        #There is an assymmetry in __getattr__ and __setattr__,
        # the former is only called when an attribute does not exist, the other is always called
        #Hence we explicitly check whether we want to store an attribute locally or in the elastic_transformer
        if name=="elastic_transformer":
            self.__dict__[name] = value
            return object.__setattr__(self, name, value)
        return setattr(self.elastic_transformer,name,value)
            
    def _transform(self,I, modality_i):
        if self.do_deformation:
            I[:,:,:,modality_i] = randomdeformation.apply_deformation(I[:,:,:,modality_i], self.elastic_transformer.deformation_field,order=0)
        
        
from . import affinedeformation
class AffineTransformer(ImageTransformer):        
    def __init__(self, randomness=Randomness.PerSample, 
                 voxel_size = (1,1,1),
                 transform_probability=1.0,
                 mean_scaling=None, mean_rotation=None, mean_translation=None,
                 std_scaling=None, std_rotation=None, std_translation=None,
                 transformation_to_original = True,
                 order=1, mode = "nearest", cval = 0
                 ):
        super(AffineTransformer,self).__init__(randomness)
        self.voxel_size=voxel_size
        self.transform_probability = transform_probability
        self.mean_scaling=mean_scaling
        self.mean_rotation=mean_rotation
        self.mean_translation=mean_translation
        self.std_scaling=std_scaling
        self.std_rotation=std_rotation
        self.std_translation=std_translation
        self.transformation_to_original = transformation_to_original
        self.order = order
        self.mode = mode
        self.cval = cval
        #state
        self.scaling = None
        self.rotation = None
        self.translation = None

    def _randomize(self, I):
        shape = I.shape[:3]
        self.do_deformation = (random.uniform(0,1)<self.transform_probability)
        if self.do_deformation:
            if self.mean_scaling is not None:
                self.scaling     = [np.random.normal(mean,std) for mean,std in zip(self.mean_scaling,self.std_scaling)]
            if self.mean_rotation is not None:
                self.rotation    = [np.random.normal(mean,std) for mean,std in zip(self.mean_rotation,self.std_rotation)]
            if self.mean_translation is not None:
                self.translation = [np.random.normal(mean,std) for mean,std in zip(self.mean_translation,self.std_translation)]

    def _transform(self,I, modality_i):
        if self.do_deformation:
            I[:,:,:,modality_i] = affinedeformation.apply_deformation(I[:,:,:,modality_i],
                                                                      scaling = self.scaling,
                                                                      rotation= self.rotation,
                                                                      translation=self.translation,
                                                                      voxel_size=self.voxel_size,
                                                                      transformation_to_original=self.transformation_to_original,
                                                                      order=self.order,
                                                                      mode=self.mode,
                                                                      cval=self.cval)
            
    def get_dimensions(self,initial_dimensions):
        return initial_dimensions


class AffineGroundTruthTransformer(ImageTransformer):
    def __init__(self,affine_transformer):
        self.affine_transformer=affine_transformer

    def __getattr__(self,name):
        if name=="affine_transformer":                 #necessary check to avoid endless loops when pickling
            raise AttributeError(name)
        return getattr(self.affine_transformer,name)
           

    def __setattr__(self,name,value):
        #There is an assymmetry in __getattr__ and __setattr__,
        # the former is only called when an attribute does not exist, the other is always called
        #Hence we explicitly check whether we want to store an attribute locally or in the elastic_transformer
        if name=="affine_transformer":
            self.__dict__[name] = value
            return object.__setattr__(self, name, value)
        return setattr(self.affine_transformer,name,value)
            
    def _transform(self,I, modality_i):
        if self.do_deformation:
            I[:,:,:,modality_i] = affinedeformation.apply_deformation(I[:,:,:,modality_i], 
                                                                      scaling = self.scaling,
                                                                      rotation= self.rotation,
                                                                      voxel_size=self.voxel_size,
                                                                      translation=self.translation,
                                                                      transformation_to_original=self.transformation_to_original,
                                                                      order=0,
                                                                      mode="nearest")
    
    
class Flipper(ImageTransformer):        
    def __init__(self, randomness=Randomness.PerSubject, flipxyz_probabilities=(0.5,0,0)):
        super(Flipper,self).__init__(randomness)
        self.flipxyz_probabilities = flipxyz_probabilities
        
    def _randomize(self, _I):
        self.flipxyz = [np.random.binomial(1,flip_p) for flip_p in self.flipxyz_probabilities]
    
    def _transform(self,I, modality_i=None):
        directionality = tuple(-1 if flip else 1 for flip in self.flipxyz)
        if modality_i is not None:
            I[:,:,:,modality_i] = I[::directionality[0],::directionality[1],::directionality[2],modality_i]
        else:
            I[:,:,:,:]          = I[::directionality[0],::directionality[1],::directionality[2],:]
    
    def get_dimensions(self,initial_dimensions):
        return initial_dimensions
       

class Thresholder(ImageTransformer):        
    def __init__(self, threshold):
        '''
        Threshold at what value?
        - float: absolute value
        - "min"
        '''
        super(Thresholder, self).__init__(Randomness.Never)
        self.threshold = threshold
        
    def _transform(self, I, modality_i=None):
        if self.threshold == "min":
            value = np.min(I)
        else:
            value = self.threshold
        if modality_i is not None:
            I[..., modality_i] = I[..., modality_i] > value
        else:
            I[...] = I > value
    
    def get_dimensions(self, initial_dimensions):
        return initial_dimensions
    
    
class SamplingWeigher(ImageTransformer):
    '''
    This ImageTransformer can (and should) be used on the ROI that is used for global class weighted sampling (when implemented).
    You can also use it when you do not mask the network output for the same reason. 
        --> If no ROI is used in this case, the complete image should be presented as the ROI (making use of e.g. ImageAsROIMarker).
    It should be a "categorical" ROI.
    Don't forger to mask-multiply the output of the network with the weights calculated here.
    '''
    def __init__(self, segment_size):
        super(SamplingWeigher, self).__init__(randomness=1)
        self.segment_size = segment_size
        self.segment = np.ones(self.segment_size)
        
    def _transform(self, I, modality_i):
        for c in zip(*I[..., modality_i].nonzero()):
            I[tuple([slice(d - s//2, d + s//2 + 1) for d, s in zip(c, self.segment_size)] + [modality_i])] += 1
        I[..., modality_i] = 1 / np.clip(I[..., modality_i], 1, None)
        
    def get_dimensions(self, initial_dimensions):
        return initial_dimensions


class FullImageAsROIMarker(ImageTransformer):
    '''
    This ImageTransformer simply makes the entire input the ROI or the part of the input where values are larger than a certain threshold. 
    Can be used to get an output mask. Can be used in a loader together with SamplingWeigher to correct change in P(X|Y) near borders.
    '''
    def __init__(self, threshold=None):
        self.threshold = threshold
        super(FullImageAsROIMarker, self).__init__(randomness=1)
        
    def _transform(self, I, modality_i):
        if self.threshold is not None:
            I[..., modality_i] = np.ones_like(I[..., modality_i]) * (I[..., modality_i] > self.threshold)
        else:
            I[..., modality_i] = 1
        
    def get_dimensions(self, initial_dimensions):
        return initial_dimensions
    
    
# class AffineTransformer(ImageTransformer):
#     def __init__(self, randomness,
#                  translation_x_range=(-2,2),    #in voxels
#                  translation_y_range=(-2,2),    #in voxels
#                  translation_z_range=(-2,2),    #in voxels
#                  
#                  deformation_MAD_range = (0,4),
#                  smooth_sigma_range = (30,50),
#                  transform_probability=1.0,
#                  voxel_size = None,
#                  max_smooth_sigma_wo_interpolation = 10
#                  ):
#         '''
#             deformation_MAD:      expected mean abs of deformation field, can be a tuple
#             smooth_sigma:         spatial smoothing, can be a tuple 
#         '''
#         super(ElasticTransformer,self).__init__(randomness)
#         self.deformation_MAD_range = deformation_MAD_range
#         self.smooth_sigma_range = smooth_sigma_range
#         self.transform_probability = transform_probability
#         self.voxel_size = voxel_size
#         self.max_smooth_sigma_wo_interpolation = max_smooth_sigma_wo_interpolation
#         
#     def _randomize(self, I):
#         shape = I.shape[:3]
#         self.do_deformation = (random.uniform(0,1)<self.transform_probability)
#         if self.do_deformation:
#             self.deformation_MAD = random.uniform(*self.deformation_MAD_range)
#             self.smooth_sigma = random.uniform(*self.smooth_sigma_range)
#             self.deformation_field = randomdeformation.get_random_deformation(shape, 
#                  self.deformation_MAD, self.smooth_sigma,
#                  voxel_size=self.voxel_size, max_smooth_sigma_wo_interpolation=self.max_smooth_sigma_wo_interpolation)
# 
#     def _transform(self,I, modality_i):
#         if self.do_deformation:
#             I[:,:,:,modality_i] = randomdeformation.apply_deformation(I[:,:,:,modality_i], self.deformation_field)
# 
#     def get_dimensions(self,initial_dimensions):
#         return initial_dimensions
# 
# 
# class AffineGroundTruthTransformer(ImageTransformer):
#     def __init__(self,affine_transformer):
#         self.affine_transformer=affine_transformer
# 
#     def __getattr__(self,name):
#         if name=="affine_transformer":                 #necessary check to avoid endless loops when pickling
#             raise AttributeError(name)
#         return getattr(self.affine_transformer,name)
#            
# 
#     def __setattr__(self,name,value):
#         #There is an assymmetry in __getattr__ and __setattr__,
#         # the former is only called when an attribute does not exist, the other is always called
#         #Hence we explicitly check whether we want to store an attribute locally or in the elastic_transformer
#         if name=="affine_transformer":
#             self.__dict__[name] = value
#             return object.__setattr__(self, name, value)
#         return setattr(self.affine_transformer,name,value)
#             
#     def _transform(self,I, modality_i):
#         if self.do_deformation:
#             I[:,:,:,modality_i] = randomdeformation.apply_deformation(I[:,:,:,modality_i], self.elastic_transformer.deformation_field,order=0)