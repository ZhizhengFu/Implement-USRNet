import numpy as np
from scipy import fftpack
import torch

from math import cos, sin
from numpy import zeros, array, pi, log, min, arange, sum, mgrid, exp, pad, round
from numpy.random import randn, rand
from scipy.signal import convolve2d
import cv2
import random

'''
modified by Kai Zhang (github: https://github.com/cszn)
03/03/2019
'''

def rfft(t):
    return torch.rfft(t, 2, onesided=False)


def irfft(t):
    return torch.irfft(t, 2, onesided=False)


def fft(t):
    return torch.fft(t, 2)


def ifft(t):
    return torch.ifft(t, 2)


def p2o(psf, shape):
    '''
    # psf: NxCxhxw
    # shape: [H,W]
    # otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.rfft(otf, 2, onesided=False)
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    otf[...,1][torch.abs(otf[...,1])<n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf

def wrap_boundary_liu(img, img_size):

    """
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    if img.ndim == 2:
        ret = wrap_boundary(img, img_size)
    elif img.ndim == 3:
        ret = [wrap_boundary(img[:, :, i], img_size) for i in range(3)]
        ret = np.stack(ret, 2)
    return ret


def wrap_boundary(img, img_size):

    """
    python code from:
    https://github.com/ys-koshelev/nla_deblur/blob/90fe0ab98c26c791dcbdf231fe6f938fca80e2a0/boundaries.py
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    (H, W) = np.shape(img)
    H_w = int(img_size[0]) - H
    W_w = int(img_size[1]) - W

    # ret = np.zeros((img_size[0], img_size[1]));
    alpha = 1
    HG = img[:, :]

    r_A = np.zeros((alpha*2+H_w, W))
    r_A[:alpha, :] = HG[-alpha:, :]
    r_A[-alpha:, :] = HG[:alpha, :]
    a = np.arange(H_w)/(H_w-1)
    # r_A(alpha+1:end-alpha, 1) = (1-a)*r_A(alpha,1) + a*r_A(end-alpha+1,1)
    r_A[alpha:-alpha, 0] = (1-a)*r_A[alpha-1, 0] + a*r_A[-alpha, 0]
    # r_A(alpha+1:end-alpha, end) = (1-a)*r_A(alpha,end) + a*r_A(end-alpha+1,end)
    r_A[alpha:-alpha, -1] = (1-a)*r_A[alpha-1, -1] + a*r_A[-alpha, -1]

    r_B = np.zeros((H, alpha*2+W_w))
    r_B[:, :alpha] = HG[:, -alpha:]
    r_B[:, -alpha:] = HG[:, :alpha]
    a = np.arange(W_w)/(W_w-1)
    r_B[0, alpha:-alpha] = (1-a)*r_B[0, alpha-1] + a*r_B[0, -alpha]
    r_B[-1, alpha:-alpha] = (1-a)*r_B[-1, alpha-1] + a*r_B[-1, -alpha]

    if alpha == 1:
        A2 = solve_min_laplacian(r_A[alpha-1:, :])
        B2 = solve_min_laplacian(r_B[:, alpha-1:])
        r_A[alpha-1:, :] = A2
        r_B[:, alpha-1:] = B2
    else:
        A2 = solve_min_laplacian(r_A[alpha-1:-alpha+1, :])
        r_A[alpha-1:-alpha+1, :] = A2
        B2 = solve_min_laplacian(r_B[:, alpha-1:-alpha+1])
        r_B[:, alpha-1:-alpha+1] = B2
    A = r_A
    B = r_B

    r_C = np.zeros((alpha*2+H_w, alpha*2+W_w))
    r_C[:alpha, :] = B[-alpha:, :]
    r_C[-alpha:, :] = B[:alpha, :]
    r_C[:, :alpha] = A[:, -alpha:]
    r_C[:, -alpha:] = A[:, :alpha]

    if alpha == 1:
        C2 = C2 = solve_min_laplacian(r_C[alpha-1:, alpha-1:])
        r_C[alpha-1:, alpha-1:] = C2
    else:
        C2 = solve_min_laplacian(r_C[alpha-1:-alpha+1, alpha-1:-alpha+1])
        r_C[alpha-1:-alpha+1, alpha-1:-alpha+1] = C2
    C = r_C
    # return C
    A = A[alpha-1:-alpha-1, :]
    B = B[:, alpha:-alpha]
    C = C[alpha:-alpha, alpha:-alpha]
    ret = np.vstack((np.hstack((img, B)), np.hstack((A, C))))
    return ret


def solve_min_laplacian(boundary_image):
    (H, W) = np.shape(boundary_image)

    # Laplacian
    f = np.zeros((H, W))
    # boundary image contains image intensities at boundaries
    boundary_image[1:-1, 1:-1] = 0
    j = np.arange(2, H)-1
    k = np.arange(2, W)-1
    f_bp = np.zeros((H, W))
    f_bp[np.ix_(j, k)] = -4*boundary_image[np.ix_(j, k)] + boundary_image[np.ix_(j, k+1)] + boundary_image[np.ix_(j, k-1)] + boundary_image[np.ix_(j-1, k)] + boundary_image[np.ix_(j+1, k)]
    
    del(j, k)
    f1 = f - f_bp  # subtract boundary points contribution
    del(f_bp, f)

    # DST Sine Transform algo starts here
    f2 = f1[1:-1,1:-1]
    del(f1)

    # compute sine tranform
    if f2.shape[1] == 1:
        tt = fftpack.dst(f2, type=1, axis=0)/2
    else:
        tt = fftpack.dst(f2, type=1)/2

    if tt.shape[0] == 1:
        f2sin = np.transpose(fftpack.dst(np.transpose(tt), type=1, axis=0)/2)
    else:
        f2sin = np.transpose(fftpack.dst(np.transpose(tt), type=1)/2) 
    del(f2)

    # compute Eigen Values
    [x, y] = np.meshgrid(np.arange(1, W-1), np.arange(1, H-1))
    denom = (2*np.cos(np.pi*x/(W-1))-2) + (2*np.cos(np.pi*y/(H-1)) - 2)

    # divide
    f3 = f2sin/denom
    del(f2sin, x, y)

    # compute Inverse Sine Transform
    if f3.shape[0] == 1:
        tt = fftpack.idst(f3*2, type=1, axis=1)/(2*(f3.shape[1]+1))
    else:
        tt = fftpack.idst(f3*2, type=1, axis=0)/(2*(f3.shape[0]+1))
    del(f3)
    if tt.shape[1] == 1:
        img_tt = np.transpose(fftpack.idst(np.transpose(tt)*2, type=1)/(2*(tt.shape[0]+1)))
    else:
        img_tt = np.transpose(fftpack.idst(np.transpose(tt)*2, type=1, axis=0)/(2*(tt.shape[1]+1)))
    del(tt)

    # put solution in inner points; outer points obtained from boundary image
    img_direct = boundary_image
    img_direct[1:-1, 1:-1] = 0
    img_direct[1:-1, 1:-1] = img_tt
    return img_direct


"""
Created on Thu Jan 18 15:36:32 2018
@author: italo
https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
"""

"""
Syntax
h = fspecial(type)
h = fspecial('average',hsize)
h = fspecial('disk',radius)
h = fspecial('gaussian',hsize,sigma)
h = fspecial('laplacian',alpha)
h = fspecial('log',hsize,sigma)
h = fspecial('motion',len,theta)
h = fspecial('prewitt')
h = fspecial('sobel')
"""


def fspecial_average(hsize=3):
    """Smoothing filter"""
    return np.ones((hsize, hsize))/hsize**2


def fspecial_disk(radius):
    """Disk filter"""
    raise(NotImplemented)


def fspecial_gaussian(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    h = g / g.sum()
    h[h < np.finfo(float).eps * h.max()] = 0  # Use numpy.finfo instead of scipy.finfo
    return h


def fspecial_laplacian(alpha):
    alpha = max([0, min([alpha,1])])
    h1 = alpha/(alpha+1)
    h2 = (1-alpha)/(alpha+1)
    h = [[h1, h2, h1], [h2, -4/(alpha+1), h2], [h1, h2, h1]]
    h = np.array(h)
    return h


def fspecial_log(hsize, sigma):
    raise(NotImplemented)


def fspecial_motion(motion_len, theta):
    raise(NotImplemented)


def fspecial_prewitt():
    return np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])


def fspecial_sobel():
    return np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


def fspecial(filter_type, *args, **kwargs):
    '''
    python code from:
    https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
    '''
    if filter_type == 'average':
        return fspecial_average(*args, **kwargs)
    if filter_type == 'disk':
        return fspecial_disk(*args, **kwargs)
    if filter_type == 'gaussian':
        return fspecial_gaussian(*args, **kwargs)
    if filter_type == 'laplacian':
        return fspecial_laplacian(*args, **kwargs)
    if filter_type == 'log':
        return fspecial_log(*args, **kwargs)
    if filter_type == 'motion':
        return fspecial_motion(*args, **kwargs)
    if filter_type == 'prewitt':
        return fspecial_prewitt(*args, **kwargs)
    if filter_type == 'sobel':
        return fspecial_sobel(*args, **kwargs)


def fspecial_gauss(size, sigma):
    x, y = mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]
    g = exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def blurkernel_synthesis(h=37, w=None):
    # https://github.com/tkkcc/prior/blob/879a0b6c117c810776d8cc6b63720bf29f7d0cc4/util/gen_kernel.py
    w = h if w is None else w
    kdims = [h, w]
    x = randomTrajectory(250)
    k = None
    while k is None:
        k = kernelFromTrajectory(x)

    # center pad to kdims
    pad_width = ((kdims[0] - k.shape[0]) // 2, (kdims[1] - k.shape[1]) // 2)
    pad_width = [(pad_width[0],), (pad_width[1],)]
    
    if pad_width[0][0]<0 or pad_width[1][0]<0:
        k = k[0:h, 0:h]
    else:
        k = pad(k, pad_width, "constant")
    x1,x2 = k.shape
    if np.random.randint(0, 4) == 1:
        k = cv2.resize(k, (random.randint(x1, 5*x1), random.randint(x2, 5*x2)), interpolation=cv2.INTER_LINEAR)
        y1, y2 = k.shape
        k = k[(y1-x1)//2: (y1-x1)//2+x1, (y2-x2)//2: (y2-x2)//2+x2]
        
    if sum(k)<0.1:
        k = fspecial_gaussian(h, 0.1+6*np.random.rand(1))
    k = k / sum(k)
    return k


def kernelFromTrajectory(x):
    h = 5 - log(rand()) / 0.15
    h = round(min([h, 27])).astype(int)
    h = h + 1 - h % 2
    w = h
    k = zeros((h, w))

    xmin = min(x[0])
    xmax = max(x[0])
    ymin = min(x[1])
    ymax = max(x[1])
    xthr = arange(xmin, xmax, (xmax - xmin) / w)
    ythr = arange(ymin, ymax, (ymax - ymin) / h)

    for i in range(1, xthr.size):
        for j in range(1, ythr.size):
            idx = (
                (x[0, :] >= xthr[i - 1])
                & (x[0, :] < xthr[i])
                & (x[1, :] >= ythr[j - 1])
                & (x[1, :] < ythr[j])
            )
            k[i - 1, j - 1] = sum(idx)
    if sum(k) == 0:
        return
    k = k / sum(k)
    k = convolve2d(k, fspecial_gauss(3, 1), "same")
    k = k / sum(k)
    return k


def randomTrajectory(T):
    x = zeros((3, T))
    v = randn(3, T)
    r = zeros((3, T))
    trv = 1 / 1
    trr = 2 * pi / T
    for t in range(1, T):
        F_rot = randn(3) / (t + 1) + r[:, t - 1]
        F_trans = randn(3) / (t + 1)
        r[:, t] = r[:, t - 1] + trr * F_rot
        v[:, t] = v[:, t - 1] + trv * F_trans
        st = v[:, t]
        st = rot3D(st, r[:, t])
        x[:, t] = x[:, t - 1] + st
    return x


def rot3D(x, r):
    Rx = array([[1, 0, 0], [0, cos(r[0]), -sin(r[0])], [0, sin(r[0]), cos(r[0])]])
    Ry = array([[cos(r[1]), 0, sin(r[1])], [0, 1, 0], [-sin(r[1]), 0, cos(r[1])]])
    Rz = array([[cos(r[2]), -sin(r[2]), 0], [sin(r[2]), cos(r[2]), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    x = R @ x
    return x