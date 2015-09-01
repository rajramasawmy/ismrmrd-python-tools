# -*- coding: utf-8 -*-
"""
Utilities for coil sensivity maps, pre-whitening, etc
"""
import numpy as np
from scipy import ndimage

def calculate_prewhitening(noise, scale_factor=1.0):
    '''Calculates the noise prewhitening matrix

    :param noise: Input noise data (array or matrix), ``[coil, nsamples]``
    :scale_factor: Applied on the noise covariance matrix. Used to 
                   adjust for effective noise bandwith and difference in 
                   sampling rate between noise calibration and actual measurement: 
                   scale_factor = (T_acq_dwell/T_noise_dwell)*NoiseReceiverBandwidthRatio
                   
    :returns w: Prewhitening matrix, ``[coil, coil]``, w*data is prewhitened
    '''

    noise_int = noise.reshape((noise.shape[0],noise.size/noise.shape[0]))
    M = float(noise_int.shape[1])    
    dmtx = (1/(M-1))*np.asmatrix(noise_int)*np.asmatrix(noise_int).H    
    dmtx = np.linalg.inv(np.linalg.cholesky(dmtx));
    dmtx = dmtx*np.sqrt(2)*np.sqrt(scale_factor);
    return dmtx

def apply_prewhitening(data,dmtx):
    '''Apply the noise prewhitening matrix

    :param noise: Input noise data (array or matrix), ``[coil, ...]``
    :param dmtx: Input noise prewhitening matrix
    
    :returns w_data: Prewhitened data, ``[coil, ...]``,
    '''

    s = data.shape
    return np.asarray(np.asmatrix(dmtx)*np.asmatrix(data.reshape(data.shape[0],data.size/data.shape[0]))).reshape(s)
    

def apply_csm(img, csm):
    '''Apply coil sensitivity maps to combine images

    :param img: Input images, ``[coil, y, x]``, or ``[coil, z, y, x]``
    :param csm: Coil sensitivity maps, ``[coil, y, x]``, or ``[coil, z, y, x]``

    :returns comim: Combined image, ``[y, x]`` or ``[z, y, x]``
    '''
    
    assert (img.shape == csm.shape), "Images and coil sensitivities must have matching shape"
    comim = np.squeeze(np.sum(np.conj(csm)*img,axis=0))

    return comim


def calculate_csm_walsh(img, smoothing=5, niter=3):
    '''Calculates the coil sensitivities using an iterative version of the Walsh method

    :param img: Input images, ``[coil, y, x]``, or ``[coil, z, y, x]``
    :param smoothing: Smoothing block size (default ``5``)
    :parma niter: Number of iterations for the power method (default ``3``).

    :returns csm: Relative coil sensitivity maps, ``[coil, y, x]`` or ``[coil, z, y, x]``
    :returns rho: Total power in the estimated coils maps, ``[y, x]`` or ``[z, y, x]``
    '''

    assert (img.ndim == 3) or (img.ndim == 4) , "Images must have 3 or 4 dimensions"

    if (img.ndim == 3):
        (csm, rho, comim) = _calculate_csm_walsh_2D(img, smoothing, niter)
    else:
        (csm, rho, comim) = _calculate_csm_walsh_3D(img, smoothing, niter)

    return (csm, rho, comim)

def _calculate_csm_walsh_2D(img, smoothing=5, niter=3):
    '''2D version of the iterative Walsh method'''

    ncoils = img.shape[0]
    ny = img.shape[1]
    nx = img.shape[2]

    # Compute the sample covariance pointwise
    Rs = np.zeros((ncoils,ncoils,ny,nx),dtype=img.dtype)
    for p in range(ncoils):
        for q in range(ncoils):
            Rs[p,q,:,:] = 1.0/(smoothing**2)*img[p,:,:] * np.conj(img[q,:,:])

    # Smooth the covariance
    for p in range(ncoils):
        for q in range(ncoils):
            Rs[p,q,:,:] = smooth(Rs[p,q,:,:], smoothing)

    # At each point in the image, find the dominant eigenvector
    # and corresponding eigenvalue of the signal covariance
    # matrix using the power method
    comim = np.zeros((ny, nx),dtype=img.dtype)
    rho = np.zeros((ny, nx))
    csm = np.zeros((ncoils, ny, nx),dtype=img.dtype)
    for y in range(ny):
        for x in range(nx):
            R = Rs[:,:,y,x]
            v = np.sum(R,axis=0)
            lam = np.linalg.norm(v)
            v = v/lam
            
            for iter in range(niter):
                v = np.dot(R,v)
                lam = np.linalg.norm(v)
                v = v/lam

            comim[y,x] = np.sum(np.conj(v)*img[:,y,x])
            rho[y,x] = smoothing*np.sqrt(lam)
            csm[:,y,x] = v
    

    return (csm, rho, comim)


def _calculate_csm_walsh_3D(img, smoothing=5, niter=3):
    '''3D version of the iterative Walsh method'''
    
    ncoils = img.shape[0]
    nz = img.shape[1]
    ny = img.shape[2]
    nx = img.shape[3]

    # Compute the sample covariance pointwise
    Rs = np.zeros((ncoils,ncoils,nz,ny,nx),dtype=img.dtype)
    for p in range(ncoils):
        for q in range(ncoils):
            Rs[p,q,:,:,:] = 1.0/(smoothing**3)*img[p,:,:,:] * np.conj(img[q,:,:,:])

    # Smooth the covariance
    for p in range(ncoils):
        for q in range(ncoils):
            Rs[p,q,:,:,:] = smooth(Rs[p,q,:,:,:], smoothing)

    # At each point in the image, find the dominant eigenvector
    # and corresponding eigenvalue of the signal covariance
    # matrix using the power method
    comim = np.zeros((nz, ny, nx),dtype=img.dtype)
    rho = np.zeros((nz, ny, nx))
    csm = np.zeros((ncoils, nz, ny, nx),dtype=img.dtype)
    for y in range(nz):
        for y in range(ny):
            for x in range(nx):
                R = Rs[:,:,z,y,x]
                v = np.sum(R,axis=0)
                lam = np.linalg.norm(v)
                v = v/lam
            
                for iter in range(niter):
                    v = np.dot(R,v)
                    lam = np.linalg.norm(v)
                    v = v/lam

                comim[z,y,x] = np.sum(np.conj(v)*img[:,z,y,x])
                rho[z,y,x] = smoothing*np.sqrt(lam)
                csm[:,z,y,x] = v
        
    return (csm, rho, comim)

def calculate_csm_global(data):

    if not ((data.ndim == 3) or (data.ndim == 4)):
        raise ValueError('Data dimension error: data must be 3D or 4D')

    nc = data.shape[0]
    
    if data.ndim == 3:
        beta = np.sum(data,axis=(1,2))
        beta /= np.linalg.norm(beta[:])
        cs = beta.reshape((nc,1,1)) * np.ones(data.shape,data.dtype)
    else:
        beta = np.sum(data,axis=(1,2,3))
        beta /= np.linalg.norm(beta[:])
        cs = beta.reshape((nc,1,1,1)) * np.ones(data.shape,data.dtype)
    
    comim = np.squeeze(np.sum(np.conj(cs)*data,axis=0))

    if data.ndim == 3:
        rho = np.sqrt(np.mean(np.abs(comim)**2, axis=(0,1)))
    else:
        rho = np.sqrt(np.mean(np.abs(comim)**2, axis=(0,1,2)))

    return cs, rho, comim


def calculate_csm_rot(data, smoothing=5):
    
    if not ((data.ndim == 3) or (data.ndim == 4)):
        raise ValueError('Data dimension error: data must be 3D or 4D')

    if not smoothing>0:
        raise ValueError('Smoothing size error: box must be a positive integer')

    nc = data.shape[0]
                
    cs = np.zeros(data.shape,dtype=data.dtype)
    for c in range(nc):
        if data.ndim == 3:
            cs[c,:,:] = smooth(np.squeeze(data[c,:,:]), smoothing)
        else:
            cs[c,:,:,:] = smooth(np.squeeze(data[c,:,:,:]), smoothing)

    csnorm = np.sqrt(np.sum(np.abs(cs)**2,axis=0))
    cs /= csnorm

    comim = np.squeeze(np.sum(np.conj(cs)*data,axis=0))    
    rho = np.sqrt(smooth(np.abs(comim)**2, smoothing))

    return cs, rho, comim

def calculate_csm_inati(data,smoothing=5,niter=5):

    if not ((data.ndim == 3) or (data.ndim == 4)):
        raise ValueError('Data dimension error: data must be 3D or 4D')

    if not (smoothing>0):
        raise ValueError('Box size error: box must be a positive integer')

    # the number of coils
    nc = data.shape[0]
    
    # initialize
    (cs, rho, comim) = calculate_csm_global(data)

    # store the global combine weights
    # this can be done in a much more memory efficient way by just
    # storing the individual channel weights
    cs_global = np.copy(cs)
    
    for iter in range(niter):
        
        # comim is D*v, (i.e. u*s)
        # rho is s
        
        # (u^H*s)*D, i.e. s^2 * v^H
        for c in range(nc):
            if data.ndim == 3:
                cs[c,:,:] = smooth(np.squeeze(np.conj(comim)*data[c,:,:]), smoothing)
            else:
                cs[c,:,:,:] = smooth(np.squeeze(np.cong(comim)*data[c,:,:,:]), smothing)

        # combine s*s*v, using the global combiner
        comim_glob = apply_csm(cs, cs_global)
        # and remove the phase
        cs *= np.exp(-1j*np.angle(comim_glob))

        # normalize s*s*v, i.e. compute s*s
        csnorm = np.sqrt(np.sum(np.abs(cs)**2,axis=0))
        cs /= csnorm

        # D*v = u*s
        comim = apply_csm(data, cs)

    # compute s
    rho = np.sqrt(csnorm)

    return cs, rho, comim

def smooth(img, box=5):
    '''Smooth images with a uniform filter

    :param img: Input real or complex images, ``[y, x] or [z, y, x]``
    :param box: Smoothing block size (default ``5``)

    :returns simg: Smoothed real or complex image ``[y,x] or [z,y,x]``
    '''
    
    if not np.isrealobj(img):
        t_real = np.zeros(img.shape)
        t_imag = np.zeros(img.shape)
        ndimage.filters.uniform_filter(img.real,size=box,output=t_real)
        ndimage.filters.uniform_filter(img.imag,size=box,output=t_imag)
        simg = t_real + 1j*t_imag

    else:
        simg = np.zeros(img.shape)
        ndimage.filters.uniform_filter(img,size=box,output=simg)
        
    return simg
