import os

import numpy as np
import cv2
from numba import njit

class NaiveStereoCorrespondence(object):
    def __init__(self, img1, img2, color=False, k_size=5, ndisp=None, disp_aware=False):
        """
        :img1:   image np array or image path of left image
        :img2:   image np array or image path of left image
        :color:  boolean, true for three channel BGR images, false for grayscale
        :k_size: size of (k x k) kernel used in SSD template matching
                 for stereo correspondence
        :ndisp:  int specifying the disparity range (0,1,...,ndisp-1).
                 Does nothing when disp_aware is false.
        :disp_aware: Boolean to determine if the program examines the entire
                     epipolar line (disp_aware=False) or scans only the possible
                     disparities specified by ndisp (disp_aware=True)
        """

        # Assert correct format
        for im in [img1, img2]:
            assert isinstance(im, np.ndarray) or isinstance(im, str)

        # Handle paths
        flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
        if isinstance(img1, str):
            img1 = cv2.imread(img1, flag)
        if isinstance(img2, str):
            img2 = cv2.imread(img2, flag)

        # Assert correct dimensions
        for im in [img1, img2]:
            if color:
                assert im.ndim==3
            else:
                assert im.ndim==2

        assert img1.shape==img2.shape

        # Create local variables
        self.color = color
        self.img1 = img1
        self.img2 = img2
        self.k_size = k_size
        self.ndisp = ndisp
        self.disp_aware = disp_aware

    def calculate(self):
        b = int((self.k_size-1)/2) #buffer

        # Create temp images with reflected borders
        I1 = cv2.cv2.copyMakeBorder(self.img1, b, b, b, b, cv2.BORDER_REFLECT101)
        I2 = cv2.cv2.copyMakeBorder(self.img2, b, b, b, b, cv2.BORDER_REFLECT101)

        return numba_SSD(I1, I2, self.k_size, self.img1.shape[0], self.img1.shape[1],
                        self.ndisp, self.disp_aware)

@njit(parallel=False)
def numba_SSD(I1, I2, k_size, m, n, ndisp, disp_aware):
    """Calculates stereo correspondence via SSD"""

    # Find highest SSD per pixel
    stereo = np.zeros((m,n), dtype=np.int16)
    for i in range(m):
        for j in range(n):
            min_diff = np.inf
            snip1 = I1[i:i+k_size, j:j+k_size]
            scanline = range(max(j-ndisp+1,0),j+1) if disp_aware else range(n)
            for k in scanline:
                snip2 = I2[i:i+k_size, k:k+k_size]
                diff = ((snip2-snip1)**2).sum() #SSD
                if diff<min_diff:
                    min_diff=diff
                    dist = j-k
            stereo[i,j] = dist

    return stereo
