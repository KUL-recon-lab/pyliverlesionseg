import nibabel as nib
import numpy as np
from .randomness import Randomness
import os

class ImageLoader(object):
    '''
    Load, transforms and caches images. Returns a 4D image.
    '''
    
    def __init__(self, image_paths_per_subject, image_transformers=[]):
        '''
        @param image_paths_per_subject: a list that for each subject lists the path(s) of the image(s) of that subject
        '''
        assert isinstance(image_paths_per_subject, (list, tuple))
        if not isinstance(image_paths_per_subject[0], (list, tuple)):
            image_paths_per_subject = [[i_p_p_s] for i_p_p_s in image_paths_per_subject]
        self.image_paths_per_subject = image_paths_per_subject
        self.image_transformers = image_transformers
        self.randomness = max([image_transformer.randomness for image_transformer in image_transformers] + [Randomness.PerSubject])
        #Caching
        self.old_subject_id = None
        self.old_epoch_identifier = None
        self.old_img = None    
    
    def load(self, subject_id, epoch_identifier=None):
        if self.old_img is None \
                or (epoch_identifier is not None and self.randomness>=Randomness.PerEpoch and epoch_identifier!=self.old_epoch_identifier) \
                or (subject_id!=self.old_subject_id):
            paths = self.image_paths_per_subject[subject_id]
            I = []
            for path in paths:
                if isinstance(path, str):
                    img = nib.load(path)
                    I_ = img.get_data(caching='unchanged').astype(dtype='float32')
                    img = (I_, img.affine, img.header)
                else:
                    assert isinstance(path, np.ndarray)
                    I_ = path.astype('float32')
                    img = (I_, np.eye(4), None)
                I_ = I_[..., None] if I_.ndim == 2 else I_
                I_ = I_[..., None] if I_.ndim == 3 else I_
                I.append(I_)
            I = np.concatenate(I, axis=-1)
            for image_transformer in self.image_transformers:
                I = image_transformer.transform(I, subject_id, epoch_identifier)
            #Caching
            self.old_subject_id = subject_id
            self.old_epoch_identifier = epoch_identifier
            self.old_img = (I, img[1], img[2])
        return self.old_img
    
    def get_subject_ids(self):
        return list(range(len(self.image_paths_per_subject)))
    
    def get_number_of_subjects(self):
        return len(self.image_paths_per_subject)


class ClassLoader(object):
    '''
    Creates a 4D image (1x1x1x1) out of a list of classes.
    '''

    def __init__(self, image_paths_per_subject, image_transformers=[]):
        '''
        @param image_paths_per_subject: a list that for each subject is the class
        '''
        if not isinstance(image_paths_per_subject[0], (list, tuple)):
            image_paths_per_subject = [[paths] for paths in image_paths_per_subject]
        self.image_paths_per_subject = image_paths_per_subject
        self.image_transformers = image_transformers
        self.randomness = max([image_transformer.randomness for image_transformer in image_transformers] + [Randomness.PerSubject])
        # Caching
        self.old_subject_id = None
        self.old_epoch_identifier = None
        self.old_img = None

    def load(self, subject_id, epoch_identifier=None):
        if self.old_img is None \
                or (epoch_identifier is not None and self.randomness >= Randomness.PerEpoch and epoch_identifier != self.old_epoch_identifier) \
                or (subject_id != self.old_subject_id):
            paths = self.image_paths_per_subject[subject_id]
            I = np.stack([np.array([[[path]]], dtype='float32') for path in paths], axis=-1)
            for image_transformer in self.image_transformers:
                I = image_transformer.transform(I, subject_id, epoch_identifier)
            # Caching
            self.old_subject_id = subject_id
            self.old_epoch_identifier = epoch_identifier
            self.old_img = (I, np.eye(4), None)

        return self.old_img

    def get_subject_ids(self):
        return list(range(len(self.image_paths_per_subject)))

    def get_number_of_subjects(self):
        return len(self.image_paths_per_subject)

    
class ImageLoaderFromMemory(ImageLoader):
    
    def __init__(self,images,image_transformers=[]):
        '''
        @param images:    list of tuples containing: numpy array, affine matrix, header
        '''
        self.images = images
        self.image_transformers = image_transformers
        self.randomness = max([image_transformer.randomness for image_transformer in image_transformers] + [Randomness.PerSubject])
        #Caching
        self.old_subject_id = None
        self.old_epoch_identifier = None
        self.old_img = None  
        
    def load(self, subject_id, epoch_identifier=None):
        if self.old_img is None \
                or (epoch_identifier is not None and self.randomness>=Randomness.PerEpoch and epoch_identifier!=self.old_epoch_identifier) \
                or (subject_id!=self.old_subject_id):
            img = self.images[subject_id]
            I = img[0]
            if len(I.shape)==3:
                I = I[:,:,:, np.newaxis]
            for image_transformer in self.image_transformers:
                image_transformer.transform(I, subject_id, epoch_identifier)
            #Caching
            self.old_subject_id = subject_id
            self.old_epoch_identifier = epoch_identifier
            self.old_img = (I,img[1],img[2])
        
        return self.old_img
    
    def get_subject_ids(self):
        return range(self.images)
    
    def get_number_of_subjects(self):
        return len(self.images)