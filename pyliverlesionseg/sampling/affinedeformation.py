import numpy as np
import scipy.ndimage.interpolation
import math


def apply_deformation(
        I,
        scaling=None,
        rotation=None,
        translation=None,
        voxel_size=None,
        transformation_to_original=True,
        order=1,
        mode="nearest",
        cval=0
):
    """
    Apply a certain deformation to an image, using the center of the image as origin for the scaling and rotation, and afterwards an optional translation as well.
    ---->   Therefor voxel space has its origin at the corner of the image (x = y (= z) = 0) and the units of the axes are the pixels/voxels.
            In world space the origin is at the center of the image and the axes are scaled according to the voxel size. It is there that the transformation is defined.

    @param scaling:                     tuple, one value per axis
    @param rotation:                    tuple, one value per axis, number of radians to rotate around the z,y,x axis (assuming order of x,y,z axis in I)
    @param translation:                 tuple, one value per axis
    @param transformation_to_original:  are the operations defined from the deformed to the original image space?

          --------- p_v voxel space p_v' <--------
         |                                        |
         | W                                      | W-1
         v                                        |
    world space p_w       == T_w ==>         world space' p_w'

    Step 1:
        The image is centered (T2c = Translation to center) and then the axes are scaled according to the voxel size (S2w = Scaled to world (real size)) => W = S2w T2c
    Step 2:
        The image as such is "augmented": rotated (Rw = rotated in world), scaled (Sw = scaled in world) and translated (Tw = translated in ) in that order => T_w = Tw Sw Rw
    Step 3:
        The image is put back on its original place (T2c-1) with the original voxel size (S2w-1) => W-1 = T2c-1 S2w-1
    This means the transformation matrix = matrix-1 for the augmentation is:
        matrix-1 = W-1 T_w W = T2c-1 . S2w-1 . Tw . Sw . Rw . S2w . T2c
    Because the function takes the transformation matrix from deformed space to original space, we take the inverse tranformation matrix = matrix as input to the function.

    p_w         = W p_v
    p_w'        = T_w W p_v
    p_v'        = W-1 T_w W p_v
                = T2c-1 S2w-1 Tw Sw Rw S2w T2c p_v

    :param I:
    :param scaling:
    :param rotation:
    :param translation:
    :param voxel_size:
    :param transformation_to_original:
    :param order:
    :param mode:
    :param cval:
    :return:
    """
    Tw, Sw, Rw, S2w, T2c = np.eye(I.ndim + 1), np.eye(I.ndim + 1), np.eye(I.ndim + 1), np.eye(I.ndim + 1), np.eye(I.ndim + 1)
    T2c[:-1, -1] = - np.array(I.shape) / 2
    if voxel_size is not None:
        S2w[:-1, :-1] = np.diag(voxel_size)
    if rotation is not None:
        Rw[:-1, :-1] = euler2mat(*rotation)[:I.ndim, :I.ndim]
    if scaling is not None:
        Sw[:-1, :-1] = np.diag(scaling)
    if translation is not None:
        Tw[:-1, -1] = np.array(translation)
    matrix = 1
    for m in [np.linalg.inv(T2c), np.linalg.inv(S2w), Tw, Sw, Rw, S2w, T2c]:
        matrix = np.dot(matrix, m)
    if transformation_to_original:
        matrix = np.linalg.inv(matrix)
    return scipy.ndimage.interpolation.affine_transform(I, matrix=matrix[:3,:3], offset=matrix[:3,3], output_shape=I.shape, output=None, order=order, mode=mode, cval=cval, prefilter=True)


def euler2mat(
        z=0,
        y=0,
        x=0
):
    """
    Return matrix for rotations around z, y and x axes (assuming order of x,y,z)

    Uses the z, then y, then x convention above

    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Returns
    -------
    M : array shape (3,3)
       Rotation matrix giving same rotation as for given angles

    From https://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/eulerangles.py

    :param z:
    :param y:
    :param x:
    :return:
    """
    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]]))
    M = np.eye(3)
    if Ms:
        for m in Ms[::-1]:
            M = np.dot(M, m)
    return M
