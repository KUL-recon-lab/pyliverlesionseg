'''
Created on 7 dec. 2016

Note, copy from mic.registration.deformation.random

@author: drobbe1

Some methods to do random deformations for images.

'''

import numpy as np
import scipy.ndimage.interpolation
from scipy.ndimage.filters import gaussian_filter
import math

def apply_deformation(I, deformation_per_dimension,
                       order=1, mode = "nearest", cval = 0):
    '''
    Apply a certain deformation to an image.
    '''
    D = len(I.shape)
    indices_per_dimension       = np.meshgrid( *[np.arange(I.shape[d]) for d in range(D)], indexing='ij')
    new_indices_per_dimension   = [np.reshape(indices_per_dimension[d]+deformation_per_dimension[d], (-1, 1)) for d in range(D)]
    return scipy.ndimage.interpolation.map_coordinates(I, new_indices_per_dimension, order=order, mode=mode, cval=cval).reshape(I.shape)





def get_random_deformation(shape, deformation_MAD, smooth_sigma,
                           voxel_size = None,
                           max_smooth_sigma_wo_interpolation = 10):
    """
    shape:                                shape of the deformation field
    deformation_MAD:                      expected mean abs of deformation field, can be a tuple, in physical units if voxel_size is supplied
    smooth_sigma:                         spatial smoothing, can be a tuple 
    voxel_size:                           voxel size 
    max_smooth_sigma_wo_interpolation:    if the smooth_sigma is very large, if so, generate a subsampled version and then upsample
    
    @return:    list with for every dimension the displacement in that dimension
    """
    result = []
    D = len(shape)
    # Parameter courtesy, allow non-tuples
    if not isinstance(deformation_MAD,(list,tuple,np.ndarray)):
        deformation_MAD = (deformation_MAD,) * D
    if not isinstance(smooth_sigma,(list,tuple,np.ndarray)):
        smooth_sigma = (smooth_sigma,) * D
    # Parameter courtesy, allow non numpy arrays
    deformation_MAD = np.array(deformation_MAD).astype(np.float)
    smooth_sigma = np.array(smooth_sigma).astype(np.float)
    shape = np.array(shape)
    # Parameter courtesy, voxel_size
    if voxel_size is not None:
        deformation_MAD = deformation_MAD/ voxel_size
        smooth_sigma    = smooth_sigma / voxel_size
    #check if the smooth_sigma is very large, if so, generate a subsampled version and then upsample
    if max(smooth_sigma)>max_smooth_sigma_wo_interpolation:
        subsample_factor = np.clip( np.floor(smooth_sigma/max_smooth_sigma_wo_interpolation), 1, float('inf'))
        subsampled_deformations = get_random_deformation(np.ceil(shape/subsample_factor).astype(np.int),
                                                    deformation_MAD,
                                                    smooth_sigma/subsample_factor,
                                                    max_smooth_sigma_wo_interpolation=float('inf'))
        result = [ scipy.ndimage.interpolation.zoom(deformation,subsample_factor,order=1,prefilter=False)[tuple(slice(0,s) for s in shape)]
                   for deformation in subsampled_deformations ]
        return result
        
        
    for d in range(D):
        border = 1+2*np.ceil(smooth_sigma).astype(np.int)
        random_deformation_with_border = np.random.normal(size=shape+2*border)
        random_deformation_with_border = gaussian_filter(random_deformation_with_border, smooth_sigma, mode="constant", cval=0)
        random_deformation = random_deformation_with_border[tuple(slice(b,-b) for b in border)]
        #scaling
        np.multiply(random_deformation, deformation_MAD[d] / np.mean(np.abs(random_deformation)), random_deformation)
        result.append( random_deformation )
    return result


def get_deformation_rotationxy(alpha, shape):
    sx,sy,sz = shape
    hsx,hsy,hsz = sx/2,sy/2,sz/2
    
    relxs = np.arange(sx)-hsx
    relys = np.arange(sy)-hsy
    
    # dx = (x-x0) (cos a - 1) + (y-y0)*-sin a
    # dy = (x-x0) sin a  + (y-y0) (cos a -1)
    dx_by_x = relxs*(math.cos(alpha)-1)
    dx_by_y = relys*(-math.sin(alpha))
    dy_by_x = relxs*(math.sin(alpha))
    dy_by_y = relys*(math.cos(alpha)-1)    
    
    deformation = [
        np.add.outer(dx_by_x,dx_by_y), #TODO, finish z direction
        np.add.outer(dy_by_x,dy_by_y),
        np.zeros(shape)
        ] 



if __name__=='__main__':
    import matplotlib.pyplot as plt
    import mic.io.image
    import cProfile as profile
    import time
    
    print("==Test on toy data==")
    I = 100*np.diag(np.ones((9,)))
    deformation_per_dimension = get_random_deformation(I.shape, 2, 2)
    Id = apply_deformation(I, deformation_per_dimension).astype(np.int)
    print("I:")
    print(I)
    print("deformation:")
    print(deformation_per_dimension)
    print("Id:")
    print(Id)
    
    print("==Test on a real image==")
    path_in = "/uz/data/avalok/mic/data/drobbe1/Data/AnkeFlairLesion/train/flair3m_1028.nii"
    path_in = "/uz/data/avalok/mic/tmp/drobbe1/AnkeFlairLesion/data/train/flair3m_1028.nii"
    path_GT = "/uz/data/avalok/mic/tmp/drobbe1/AnkeFlairLesion/data/train/flair3m_1028-AW-finallesion.nii"
    
    I,wm,o = mic.io.image.load(path_in)
    GT,GTwm,GTo = mic.io.image.load(path_GT)
    voxel_size = np.linalg.norm(wm[:3,:3], ord=2, axis=0)
    print("voxelsize:{}".format(voxel_size))
    deformation_per_dimension = get_random_deformation(I.shape, (0,4,4), 40, voxel_size)
    
    #profile.runctx("deformation_per_dimension = get_random_deformation(I.shape, 4/voxel_size, 40/voxel_size)",globals(),locals())
    Id1 = apply_deformation(I, deformation_per_dimension)
    GTd1 = apply_deformation(GT, deformation_per_dimension)

#    GTd1f = apply_deformation(GT.astype(np.float), deformation_per_dimension)
#     mic.io.image.save("/Users/drobbe1/Id1.nii",Id1,wm,o)
#     mic.io.image.save("/Users/drobbe1/GTd1.nii",GTd1,wm,o)
#     mic.io.image.save("/Users/drobbe1/GTd1f.nii",GTd1f,wm,o)

    t = time.time()
    deformation_per_dimension = get_random_deformation(I.shape, 4, 50, voxel_size)
    print("Deformation took {} s".format(time.time()-t))
    Id2 = apply_deformation(I, deformation_per_dimension)
    print("Deformation and application took {} s".format(time.time()-t))
    #plt.hist(np.ravel(deformation_per_dimension[1]),bins=100)
    #plt.show()    
    
    #Show
    plt.subplot(2,3,1)
    plt.imshow(I[10,:,:])
    plt.subplot(2,3,2)
    plt.imshow(Id1[10,:,:])
    plt.subplot(2,3,3)
    plt.imshow(Id2[10,:,:])
    plt.subplot(2,3,4)
    plt.imshow(I[:,100,:])
    plt.subplot(2,3,5)
    plt.imshow(Id1[:,100,:])
    plt.subplot(2,3,6)
    plt.imshow(Id2[:,100,:])
    plt.show()
    
    ## Speedtest
    s = time.time()
    I = np.zeros((250,250,80))
    deformation_per_dimension = get_random_deformation(I.shape, 10, 20)
    apply_deformation(I, deformation_per_dimension)
    print("Transformation took {} s".format(time.time()-s))