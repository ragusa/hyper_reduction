# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:40:34 2020

@author: ragusa
"""

import numpy as np
import scipy.linalg


def addblock_svd_update(U, S, Vh, A, force_orth=False):
    # Given the SVD of
    #    X = U*S*Vh,   Vh=V'
    # update it to be the SVD of
    #   [X A] = Up*Sp*Vhp,   Vhp=Vp'

    current_rank = U.shape[1]
    # print('addblock_svd_update')
    # print(U.shape)
    # print('sv in = ',sv)

    # P is an orthogonal basis of the column-space
    # of (I-UU')a, which is the component of "a" that is
    # orthogonal to U.
    m = U.T @ A
    p = A - U @ m
    P = scipy.linalg.orth(p)
    # p may not have full rank.  If not, P will be too small.  Pad
    # with zeros.
    delta_size = p.shape[1] - P.shape[1]
    if delta_size > 0:
        extra_zeros = np.zeros((P.shape[0], delta_size))
        P = np.column_stack((P, extra_zeros))
    #
    Ra = P.T @ p
    #
    # Diagonalize K, maintaining rank
    #
    z = np.zeros(m.shape)
    K = np.vstack((np.hstack((np.diag(S), m)), np.hstack((z.T, Ra))))
    #
    tUp, Sp, tVhp = scipy.linalg.svd(K, full_matrices=False, compute_uv=True)
    # print('Sp in = ',Sp)

    # Now update our matrices!
    # this may not preserve orthogonality over many repetitions.  See below.
    Up = np.hstack((U, P)) @ tUp
    # Vh: k x n
    # tVhp: kp x n
    # because python's svd returns Vh and not V the following two operations
    # are performed using Vh
    Vhp = tVhp[:, :current_rank] @ Vh
    Vhp = np.hstack((Vhp, tVhp[:, current_rank : tVhp.shape[1]]))

    # The above rotations may not preserve orthogonality, so we explicitly
    # deal with that via a QR plus another SVD.  In a long loop, you may
    # want to force orthogonality every so often.
    if force_orth:
        UQ, UR = scipy.linalg.qr(Up, mode="economic")
        VQ, VR = scipy.linalg.qr(Vhp.T, mode="economic")
        # below, we do not do: VR.T because qr was performed on Vh and not V
        tUp, Sp, tVhp = scipy.linalg.svd(
            UR @ np.diag(Sp) @ VR.T, full_matrices=False, compute_uv=True
        )
        Up = UQ @ tUp
        Vp = VQ @ tVhp.T
        Vhp = Vp.T
    return Up, Sp, Vhp
