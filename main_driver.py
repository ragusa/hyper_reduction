# -*- coding: utf-8 -*-tasm

"""
Created on Sat Apr  4 22:54:04 2020

@author: ragusa
"""
import numpy as np
import scipy.sparse.linalg

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from addblock_svd_update import addblock_svd_update
from energy_content import energy_content
from sampler import sampler

plt.close("all")
##########################################################
#
# geometry and mesh
#
geo_id_ = 1

"""
# this part does not work on windows machine
from create_geometry import create_geometry

# create specific geometry
# id=1, vol=1e-2--> 151 elements
# id=2, vol=1e1 -->  62 elements
# id=3, vol=1e2 --> 130 elements

def rf(vertices, area):
    bary = np.sum(np.array(vertices), axis=0) / 3
    x,y = bary[0], bary[1]
    rad = np.sqrt(x*x+y*y)
    a=0.01/5
    if rad>0.8:
        max_area = a
    elif rad>0.6:
        max_area = a/2
    elif rad>0.4:
        max_area = a/4
    elif rad>0.2:
        max_area = a/8
    else:
        max_area = a/16
    # max_area = 0.1
    return bool(area > max_area)

el2pt, el2at, fa2pt, fa2ma, pt2xy, el2ne = create_geometry(geo_id=geo_id_,max_vol=1e-3,\
                                                           refinement_funct=rf,do_plot=False)
"""
# whether or not the geometry was created with a refinement function or not
with_ref = False
if with_ref:
    ref_txt = "_REF"
else:
    ref_txt = ""
# geo_id      nelems
#  1       | 151 , 1553 , 3085 , 15620
#  2       | 599 , 6266 , 62278
#  4       | 104
#  5       | 153 , 322  , 1544
#  5 REF   | 2258, 4413
#  6       | 316 , 1530, 3049, 15221
#  7       | 159 , 319 , 1538, 1557, 5118
#  8       | 1 , 80
#  9       | 154, 314, 1542, 3025, 5080, 15246

# create basename to reload data
basename = (
    "./geo-" + str(geo_id_) + "/geo_id" + str(geo_id_) + ref_txt + "_elems" + "3085"
)

el2pt = np.loadtxt(basename + "_el2pt.txt", dtype=int)
el2at = np.loadtxt(basename + "_el2at.txt", dtype=int)
fa2pt = np.loadtxt(basename + "_fa2pt.txt", dtype=int)
fa2ma = np.loadtxt(basename + "_fa2ma.txt", dtype=int)
pt2xy = np.loadtxt(basename + "_pt2xy.txt")
el2ne = np.loadtxt(basename + "_el2ne.txt", dtype=int)
# raise ValueError('stopping')

# finish mesh
from tri_mesh import TriMesh

mesh = TriMesh(el2pt, el2at, fa2pt, fa2ma, pt2xy, el2ne)
mesh.complete()

# to plot the mesh
# mesh.plot_mesh()

# if you want to pickle the data or not
# import pickle as pickle
# with open('mesh'+str(geo_id_)+'.pkl', 'wb') as outp:
#     pickle.dump(mesh, outp, pickle.HIGHEST_PROTOCOL)

# with open('mesh'+str(geo_id_)+'.pkl', 'rb') as inp:
#     mesh = pickle.load(inp)

##########################################################
#
# data for that problem
#
# Caveat: cdif, qext must be entered in the order the materials are listed
# in mesh.attr
if geo_id_ == 1:
    #              (2)
    #      5----------------4
    #      |                |
    # (3)  |                |
    #      6                |
    #      |     3---2      |  (1)
    # (4)  |    /    |      |
    #      7   0-----1      |
    # (3)  |                |
    #      8----------------9
    #            (2)
    bc_rob = {
        "markers": np.array([1, 3, 4], dtype=int),
        "values": np.array([0.0, 1.0, 1.0], dtype=float),
    }
    bc_neu = {
        "markers": np.array([2], dtype=int),
        "values": np.array([0.0], dtype=float),
    }
    bc = {"Robin": bc_rob, "Neumman": bc_neu}

    cdif = np.array([3, 3], dtype=float)
    qext = np.array([0, 0], dtype=float)
else:
    raise ValueError("unknown geo_id = ", geo_id_)
mesh.check_bc(bc)

##########################################################
#
# linear operators per attribute
#
from lin_op import LinOp

lin_op = LinOp(mesh)
lin_op.build_diffusion_op_per_attribute()
lin_op.build_diffusion_op_per_bc(bc)
"""
plt.close('all')
plt.figure(0)
plt.spy(lin_op.M[0],marker='.',ms=3.)
plt.spy(lin_op.M[1],marker='.',ms=1.,color='red')
plt.figure(1)
plt.spy(lin_op.K[0],marker='.',ms=3.)
plt.spy(lin_op.K[1],marker='.',ms=1.,color='red')"""

##########################################################
#
# Build system and solve
#
siga = cdif * 0

A, b = lin_op.build_diffusion_system(qext, cdif, siga, bc)
Phi = scipy.sparse.linalg.spsolve(A, b)

if mesh.npts < 10000:
    fig = plt.figure(99)
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection="3d")
    ax.plot_trisurf(
        mesh.pt2xy[:, 0], mesh.pt2xy[:, 1], Phi, linewidth=0.2, cmap=plt.cm.CMRmap
    )
    plt.show()
raise ValueError("stopping here")
##########################################################
#
# Prepare for data perturbations
#
def new_param_values_list(cdif, siga, qext, bc, limits, which_to_pert, Pts):
    # Pts: nsamples x effective_npert
    # print('Points');print(Pts)
    nsamples = Pts.shape[0]
    npert = Pts.shape[1]
    if npert != len(np.where(which_to_pert == True)[0]):
        raise ValueError(
            "Pts.shape[1] and len( np.where(which_to_pert==True)[0] ) mismatch"
        )
    # duplicate the nominal values
    aux = np.append(cdif, siga)
    aux = np.append(aux, qext)
    for key in bc.keys():
        aux = np.append(aux, bc[key]["values"])
    one = np.ones([nsamples, 1])
    tmp = np.kron(aux, one)  # nsamples x npert
    if len(which_to_pert) != limits.shape[0]:
        raise ValueError("len(which_to_pert) != limits.shape[0]")
    counter = 0
    for p, logical in enumerate(which_to_pert):
        if logical:
            xi = Pts[:, counter]
            # x = x_ave + (x2-x1)/2 * xi
            ave = (limits[p, 1] + limits[p, 0]) / 2
            slope = (limits[p, 1] - limits[p, 0]) / 2
            tmp[:, p] = ave + xi * slope
            counter += 1
    # place in array to be returned
    n1 = 0
    n2 = len(cdif)
    cdif_ = np.array(tmp[:, n1:n2])
    n1 = n2
    n2 = n1 + len(siga)
    siga_ = np.array(tmp[:, n1:n2])
    n1 = n2
    n2 = n1 + len(qext)
    qext_ = np.array(tmp[:, n1:n2])
    bc_val = {}
    for key in bc.keys():
        bc_val[key] = {}
        n1 = n2
        n2 = n1 + len(bc[key]["values"])
        bc_val[key]["values"] = np.array(tmp[:, n1:n2])
    return cdif_, siga_, qext_, bc_val


if geo_id_ == 1:
    which_to_pert = np.zeros(10, dtype=bool)
    which_to_pert[0:5] = True
    # which_to_pert[5] = True
    # limits = np.array([[0.1,3], [0.1,5],\
    #                   [5,75], [0.1,3],\
    #                   [2,200], [0.1,1],\
    #                   [0.1,1], [0.1,1], [0.1,1],[0.1,1],\
    #                   [0.1,1], [0.1,1], [0.1,1],[0.1,1]])

    ff = 0.92
    a0 = np.array([1 - ff, 1 + ff])
    limits = np.array(
        [
            cdif[0] * a0,
            cdif[1] * a0,
            siga[0] * a0,
            siga[1] * a0,
            qext[0] * a0,
            qext[1] * a0,
            0 * a0,
            0 * a0,
            0 * a0,
            bc["Robin"]["values"][-1] * a0,
        ]
    )

    # hardcoded for this problem
    if len(which_to_pert) != 10:
        raise ValueError(
            "for geo_id=1, we need 10 params: D(2), siga(2), Q(2), Jinc(4)"
        )
    Pts = 0.3 * np.ones((3, 5))
    cdif_, siga_, qext_, bc_vals = new_param_values_list(
        cdif, siga, qext, bc, limits, which_to_pert, Pts
    )
else:
    raise ValueError("which_to_pert stills need to be implemented for other geometries")
# number of input space dimensions
iNumDimensions = len(np.where(which_to_pert == True)[0])

##########################################################
#
# Classic POD
#
# get samples in [-1,+1]^dim
number_of_snapshots = 250
use_LHS = False
TrainPoints = sampler(number_of_snapshots, iNumDimensions, use_LHS)

# get new values of parameters
cdif_, siga_, qext_, bc_vals = new_param_values_list(
    cdif, siga, qext, bc, limits, which_to_pert, TrainPoints
)


def make_bc_dict(bc, bc_vals, ind):
    bc_ = {}
    for key in bc.keys():
        bc_[key] = {}
        bc_[key]["markers"] = np.copy(bc[key]["markers"])
        bc_[key]["values"] = bc_vals[key]["values"][ind, :]
    return bc_


# compute snapshot values
pod_snapshots = np.zeros((mesh.npts, number_of_snapshots))
for ipt in range(TrainPoints.shape[0]):
    # get system and solve
    bc_ = make_bc_dict(bc, bc_vals, ipt)
    A, b = lin_op.build_diffusion_system(
        qext_[ipt, :], cdif_[ipt, :], siga_[ipt, :], bc_
    )
    Phi = scipy.sparse.linalg.spsolve(A, b)
    pod_snapshots[:, ipt] = Phi
# perform SVD
u, sv, vh = np.linalg.svd(pod_snapshots, full_matrices=False, compute_uv=True)
# energy content
ec = energy_content(sv)
# Choose the number of basis functions to keep to preserve e
e_threshold = 0.999
rank = np.argwhere(ec - e_threshold >= 0.0)[0][0]
print("POD rank = ", rank)
plt.figure(1)
# plt.semilogy(sv, label='POD-'+str(number_of_snapshots))
plt.semilogy(sv, label="POD-")
plt.legend()
plt.show()
# Get reduced basis
ur = np.copy(u[:, 0:rank])
# compute affine decomposition
lin_op.compute_reduced_operators(ur, bc)

plt.figure(2)
nn = np.minimum(number_of_snapshots, 2 * rank)
plt.plot(ec[0:nn])
plt.show()


# testing
number_of_tests = 50
TestPoints = sampler(number_of_tests, iNumDimensions, use_LHS)
# get new values of parameters
cdif_, siga_, qext_, bc_ = new_param_values_list(
    cdif, siga, qext, bc, limits, which_to_pert, TestPoints
)
rela_err = np.zeros(TestPoints.shape[0])
for ipt in range(TestPoints.shape[0]):
    # get system and solve
    bc_ = make_bc_dict(bc, bc_vals, ipt)
    Ar, br = lin_op.build_reduced_system(
        qext_[ipt, :], cdif_[ipt, :], siga_[ipt, :], bc_
    )
    c = np.linalg.solve(Ar, br)
    Phi_r = ur @ c
    A, b = lin_op.build_diffusion_system(
        qext_[ipt, :], cdif_[ipt, :], siga_[ipt, :], bc_
    )
    Phi = scipy.sparse.linalg.spsolve(A, b)
    flx_err = 100 * np.linalg.norm(Phi - Phi_r) / np.linalg.norm(Phi)
    rela_err[ipt] = flx_err
    # print('ipts={0:>3d}, L2_err(%flx)={1:,>+.8f}'.format(ipt,flx_err))
plt.figure(3)
plt.plot(rela_err)
plt.title("Flux relative error in %")
plt.show()
sv_classic = np.copy(sv)


##########################################################
#
# greedy POD
#
MiddlePoint = np.zeros((1, iNumDimensions))
# get new values of parameters
cdif_, siga_, qext_, bc_vals = new_param_values_list(
    cdif, siga, qext, bc, limits, which_to_pert, MiddlePoint
)
# raise ValueError("a")

bc_ = make_bc_dict(bc, bc_vals, 0)
A, b = lin_op.build_diffusion_system(qext_[0, :], cdif_[0, :], siga_[0, :], bc_)
phi = scipy.sparse.linalg.spsolve(A, b)
new_snap = np.expand_dims(phi, axis=1)
ur, sv, vh = np.linalg.svd(new_snap, full_matrices=False, compute_uv=True)
plt.figure(9)
plt.semilogy(sv, marker="+", label="greedy-0")

nCandidates = 1000
nKeep = 50
iend = 0
not_done = True
igreedy = 0
total_snap = np.copy(new_snap)
while not_done:
    igreedy += 1
    print("Greedy iteration ", igreedy)
    # update ROM ops using latest ur bases
    lin_op.compute_reduced_operators(ur, bc)
    # select new set of training points
    TrainPoints = sampler(nCandidates, iNumDimensions, use_LHS)
    cdif_, siga_, qext_, bc_vals = new_param_values_list(
        cdif, siga, qext, bc, limits, which_to_pert, TrainPoints
    )
    # use ROM to evaluate residual error
    Candidate_Error = np.zeros(nCandidates)
    plt.figure(7)
    for ipt in range(nCandidates):
        bc_ = make_bc_dict(bc, bc_vals, ipt)
        Ar, br = lin_op.build_reduced_system(
            qext_[ipt, :], cdif_[ipt, :], siga_[ipt, :], bc_
        )
        c = np.linalg.solve(Ar, br)
        Candidate_Error[ipt] = lin_op.residual_indicator(
            qext_[ipt, :], cdif_[ipt, :], siga_[ipt, :], bc_, c
        )
        # debug
        """A, b = lin_op.build_diffusion_system( qext_[ipt,:], cdif_[ipt,:], siga_[ipt,:], \
                                          bc_)
        aux = lin_op.residual_indicator_brute_force(A,b,ur,c)
        print(ipt,Candidate_Error[ipt],aux,aux-Candidate_Error[ipt])"""
    plt.semilogy(Candidate_Error, label="greedy-" + str(igreedy))
    plt.legend()
    # sort by largest error
    worst_ind = np.argsort(Candidate_Error)
    ibeg = iend + 1
    iend = iend + nKeep
    if iend >= number_of_snapshots:
        iend = number_of_snapshots
        nKeep = iend - ibeg
        not_done = False
    # perform FOM solves for worst param values
    new_snap = np.zeros((mesh.npts, nKeep))
    for i, ipt in enumerate(worst_ind[:nKeep]):
        bc_ = make_bc_dict(bc, bc_vals, ipt)
        A, b = lin_op.build_diffusion_system(
            qext_[ipt, :], cdif_[ipt, :], siga_[ipt, :], bc_
        )
        new_snap[:, i] = scipy.sparse.linalg.spsolve(A, b)
    # update the SVD decomposition
    ur, sv, vh = addblock_svd_update(ur, sv, vh, new_snap, force_orth=True)
    print(ur.shape)
    plt.figure(9)
    plt.semilogy(sv, label="greedy-" + str(igreedy))
    total_snap = np.hstack((total_snap, new_snap))
ur, sv, vh = np.linalg.svd(total_snap, full_matrices=False, compute_uv=True)
plt.semilogy(sv, marker="o", label="FULL")
plt.legend()
plt.show()
plt.figure(6)
plt.semilogy(sv_classic, label="classic")
plt.semilogy(sv, label="greedy")
plt.legend()
plt.show()

# MiddlePoint = 0.5*np.ones(iNumDimensions)
# # get new values of parameters
# cdif_,siga_,qext_,Jinc_,Jneu_ = new_param_values_single(cdif,siga,qext,Jinc,Jneu,\
#                                                      limits,which_to_pert,MiddlePoint)
#
# qext_[:]=0
# A, b = lin_op.build_diffusion_system( qext_, cdif_, siga_, bc, Jinc_, Jneu_)
# ur = np.random.uniform(size=[mesh.npts,1])
# print(ur.shape)
#
# lin_op.compute_reduced_operators(ur)
# Ar, br = lin_op.build_reduced_system( qext_, cdif_, siga_, bc, Jinc_, Jneu_)
# c = np.linalg.solve(Ar,br)
# auxr = lin_op.residual_indicator(qext_, cdif_, siga_, bc, Jinc_, Jneu_, c)[0]
# aux = lin_op.residual_indicator_brute_force(A,b,ur,c)
# print(auxr,aux,aux/auxr)
# print(Ar, ur.T@A@ur)
# print(br, ur.T@b)
raise ValueError("stop")

"""
##########################################################
#
# Black box model
#
def model(
    Point, is_qoi, cdif, siga, qext, Jinc, Jneu, limits, which_to_pert, lin_op, QoI_op
):
    # print("Model Point = ",Point)
    # get new values of parameters
    cdif_, siga_, qext_, Jinc_, Jneu_ = new_param_values_single(
        cdif, siga, qext, Jinc, Jneu, limits, which_to_pert, Point
    )
    # get system and solve
    A, b = lin_op.build_diffusion_system(qext_, cdif_, siga_, bc, Jinc_, Jneu_)
    Phi = scipy.sparse.linalg.spsolve(A, b)
    if is_qoi:
        val_qoi = (QoI_op @ Phi)[0, 0]
        return np.array([val_qoi])
    else:
        return Phi


############################################
#
# Construct sparse grid
#
import Tasmanian

####
# incremental SVD stuff
do_isvd = True
shrink_rank = True
e_threshold = 0.9999
####

# grid params
iInitialLevel = 1
max_level = 3
iNumThreads = 1

# make grid
grid = Tasmanian.SparseGrid()
iNumOutputs = mesh.npts
grid.makeGlobalGrid(
    iNumDimensions, iNumOutputs, iInitialLevel, "iptotal", "clenshaw-curtis"
)
# grid.makeGlobalGrid(iNumDimensions, iNumOutputs, iInitialLevel, "level", "clenshaw-curtis")
# make grid for qoi
grid_qoi = Tasmanian.SparseGrid()
# grid_qoi.makeGlobalGrid(iNumDimensions, 1, iInitialLevel, "level", "clenshaw-curtis")
grid_qoi.makeGlobalGrid(iNumDimensions, 1, iInitialLevel, "iptotal", "clenshaw-curtis")
# QoI operator
QoI_op = lin_op.M[0].sum(axis=0)

iBudget = 50
bBelowBudget = True
iteration = 0
while bBelowBudget:
    print("\n------------------------------")
    print(
        "Iteration             = ",
        iteration,
        " nbr pts = ",
        grid.getNumLoaded(),
        " nbr pts = ",
        grid_qoi.getNumLoaded(),
    )
    is_qoi = False
    Tasmanian.loadNeededPoints(
        lambda x, tid: model(
            x,
            is_qoi,
            cdif,
            siga,
            qext,
            Jinc,
            Jneu,
            limits,
            which_to_pert,
            lin_op,
            QoI_op,
        ),
        grid,
        iNumThreads,
    )
    grid.setAnisotropicRefinement("iptotal", 10, 0)
    is_qoi = True
    Tasmanian.loadNeededPoints(
        lambda x, tid: model(
            x,
            is_qoi,
            cdif,
            siga,
            qext,
            Jinc,
            Jneu,
            limits,
            which_to_pert,
            lin_op,
            QoI_op,
        ),
        grid_qoi,
        iNumThreads,
    )
    grid_qoi.setAnisotropicRefinement("iptotal", 10, 0)

    #   shrink_rank=True ...
    bBelowBudget = grid.getNumLoaded() < iBudget and grid_qoi.getNumLoaded() < iBudget
    iteration += 1

    if iNumDimensions == 2:
        fig, axs = plt.subplots(1, 2)
        grid.plotPoints2D(axs[0])
        grid_qoi.plotPoints2D(axs[1])
        plt.show()
print(
    "nbr pts grid = ",
    grid.getNumLoaded(),
    " nbr pts QoI grid = ",
    grid_qoi.getNumLoaded(),
)

# ##########################################################
# #
# # Old SG contruction
# #
#
#
# grid = Tasmanian.SparseGrid()
# iNumDimensions = len( np.where(which_to_pert==True)[0] )
# iNumOutputs = mesh.npts
# iInitialLevel = 0
# max_level = 5
#
# grid.makeGlobalGrid(iNumDimensions, iNumOutputs, iInitialLevel, "iptotal", "clenshaw-curtis")
# # grid.makeGlobalGrid(iNumDimensions, iNumOutputs, iInitialLevel, "level", "clenshaw-curtis")
#
# # make grid for qoi
# grid_qoi = Tasmanian.SparseGrid()
# # grid_qoi.makeGlobalGrid(iNumDimensions, 1, iInitialLevel, "level", "clenshaw-curtis")
# grid_qoi.makeGlobalGrid(iNumDimensions, 1, iInitialLevel, "iptotal", "clenshaw-curtis")
# # QoI operator
# QoI_op=lin_op.M[0].sum(axis=0)
#
# for lev in range(iInitialLevel,max_level):
#     print("\n------------------------------")
#     print('Level             = ',lev)
#     Points = grid.getNeededPoints()
#
#     # get new values of parameters
#     cdif_,siga_,qext_,Jinc_,Jneu_ = new_param_values_single(cdif,siga,qext,Jinc,Jneu,\
#                                                      limits,which_to_pert,Points)
#     #
#     verbose = True
#     new_snap = np.zeros( (mesh.npts, Points.shape[0]) );
#     new_snap_qoi = np.zeros( (1,Points.shape[0]) );
#     for ipt in range(Points.shape[0]):
#         # get system and solve
#         A, b = lin_op.build_diffusion_system( qext_[ipt,:], cdif_[ipt,:], siga_[ipt,:], \
#                                              bc, Jinc_[ipt,:], Jneu_[ipt,:])
#         # interpolate on current grid if good enough
#         do_solve = True
#         if lev>0:
#             Phi_i = grid.evaluate(Points[ipt])
#             qoi_i = grid_qoi.evaluate(Points[ipt])
#             resi = np.linalg.norm( A @ Phi_i - b ) / np.linalg.norm(b)
#             print('Level '+str(lev)+', Point '+str(ipt)+', resi='+str(resi))
#             if verbose:
#                 print('Level '+str(lev)+', Point '+str(ipt)+'/'+str(Points.shape[0]-1)+', resi='+str(resi))
#             if resi>0.01:
#                 do_solve = True
#             #else:
#             #    do_solve = False
#             #    Phi = np.copy(Phi_i)
#         else:
#             if verbose:
#                 print('Level '+str(lev)+', Point '+str(ipt)+'/'+str(Points.shape[0]-1))
#         # solve if requested
#         if do_solve:
#             Phi = scipy.sparse.linalg.spsolve(A, b)
#         if lev>0 and verbose:
#             print('                   L2_err(flx) = ',str(np.linalg.norm(Phi-Phi_i)))
#         # add to new snapshot array
#         new_snap[:,ipt] = Phi[:]
#         val = (QoI_op@Phi)[0,0]
#         new_snap_qoi[0,ipt] = val
#         if lev>0 and verbose:
#             print('                   QoI_err(%)  = ',str(abs(val-qoi_i[0])/qoi_i[0]*100) )
#             # print(val,qoi_i[0])
#
#     # add new snapshots to grid
#     grid.loadNeededPoints(new_snap.T)
#     grid_qoi.loadNeededPoints(new_snap_qoi.T)
#     #print('H =',grid.getHierarchicalCoefficients())
#
#     #Result = grid.evaluate(PointOfInterest)
#     #Error = np.abs(Result[0] - ReferenceSolution[0])
#     print("\n------------------------------")
#     print('Level             = ',lev)
#     #print('Result            = ',Result[0])
#     #print('ReferenceSolution = ',ReferenceSolution[0])
#     #print('Error             = ',Error)
#     #grid.plotPoints2D(plt)
#     #plt.show()
#
#
#     shrink_rank=True
#     if do_isvd:
#         fig = plt.figure(11)
#         if lev==0:
#             RR = np.copy(new_snap)
#             u, sv, vh = np.linalg.svd(new_snap, full_matrices=False, compute_uv=True)
#         else:
#             u, sv, vh = addblock_svd_update(u, sv, vh, new_snap, force_orth=False)
#             #print('sv lev++++=',sv)
#             if shrink_rank:
#                 # if reducting is not large, keep all:
#                 if False: #sv[0]/sv[-1]<1e3:
#                     r = len(sv)
#                 else:
#                     # energy content
#                     ec = energy_content(sv)
#                     # Choose the number of basis functions to keep to preserve e
#                     r = np.argwhere(ec - e_threshold >= 0.0 )[0][0]
#                 print('rank=',r)
#                 # Get reduced basis
#                 u  = np.copy(u[:, 0:r])
#                 sv = np.copy(sv[0:r])
#                 vh = np.copy(vh[0:r, :])
#         mark=''
#         if lev==0:
#           mark='+'
#         plt.semilogy(sv, marker=mark, label='lev-'+str(lev))
#
#     if lev<max_level-1:
#         grid.updateGlobalGrid(lev+1, "level")
#         #grid_qoi.updateGlobalGrid(lev+1, "level")
#         #grid.setAnisotropicRefinement("iptotal", 10, 0);
#         grid_qoi.setAnisotropicRefinement("iptotal", 1, 0);
#         print('num points        = ',grid.getNumPoints())
#         print('num points loaded = ',grid.getNumLoaded())
#         print('num points needed = ',grid.getNumNeeded())
#         #print('num points needed = ',grid.getNeededPoints())
#
#     if iNumDimensions==2:
#         fig = plt.figure(100+lev)
#         grid.plotPoints2D(plt)
#         plt.show()
#
#
# fig = plt.figure(11)
# plt.grid()
# plt.legend()
# plt.show()
#


##########################################################
#
# Copy of SG while debugging anisotropic grids
#
############################################
# incremental SVD stuff
do_isvd = True
shrink_rank = True
e_threshold = 0.9999
############################################
import Tasmanian

grid = Tasmanian.SparseGrid()
iNumDimensions = len(np.where(which_to_pert == True)[0])
iNumOutputs = mesh.npts
iInitialLevel = 0
max_level = 5

grid.makeGlobalGrid(
    iNumDimensions, iNumOutputs, iInitialLevel, "iptotal", "clenshaw-curtis"
)
# grid.makeGlobalGrid(iNumDimensions, iNumOutputs, iInitialLevel, "level", "clenshaw-curtis")

# make grid for qoi
grid_qoi = Tasmanian.SparseGrid()
# grid_qoi.makeGlobalGrid(iNumDimensions, 1, iInitialLevel, "level", "clenshaw-curtis")
grid_qoi.makeGlobalGrid(iNumDimensions, 1, iInitialLevel, "iptotal", "clenshaw-curtis")
# QoI operator
QoI_op = lin_op.M[0].sum(axis=0)

for lev in range(iInitialLevel, max_level):
    print("\n------------------------------")
    print("Level             = ", lev)
    Points = grid.getNeededPoints()

    # get new values of parameters
    cdif_, siga_, qext_, Jinc_, Jneu_ = new_param_values(
        cdif, siga, qext, Jinc, Jneu, limits, which_to_pert, Points
    )
    #
    verbose = False
    new_snap = np.zeros((mesh.npts, Points.shape[0]))
    new_snap_qoi = np.zeros((1, Points.shape[0]))
    for ipt in range(Points.shape[0]):
        # get system and solve
        A, b = lin_op.build_diffusion_system(
            qext_[ipt, :],
            cdif_[ipt, :],
            siga_[ipt, :],
            bc,
            Jinc_[ipt, :],
            Jneu_[ipt, :],
        )
        # interpolate on current grid if good enough
        do_solve = True
        if lev > 0:
            Phi_i = grid.evaluate(Points[ipt])
            qoi_i = grid_qoi.evaluate(Points[ipt])
            resi = np.linalg.norm(A @ Phi_i - b) / np.linalg.norm(b)
            if verbose:
                print(
                    "Level "
                    + str(lev)
                    + ", Point "
                    + str(ipt)
                    + "/"
                    + str(Points.shape[0] - 1)
                    + ", resi="
                    + str(resi)
                )
            if resi > 0.01:
                do_solve = True
            # else:
            #    do_solve = False
            #    Phi = np.copy(Phi_i)
        else:
            if verbose:
                print(
                    "Level "
                    + str(lev)
                    + ", Point "
                    + str(ipt)
                    + "/"
                    + str(Points.shape[0] - 1)
                )
        # solve if requested
        if do_solve:
            Phi = scipy.sparse.linalg.spsolve(A, b)
        if lev > 0 and verbose:
            print("                   L2_err(flx) = ", str(np.linalg.norm(Phi - Phi_i)))
        # add to new snapshot array
        new_snap[:, ipt] = Phi[:]
        val = (QoI_op @ Phi)[0, 0]
        new_snap_qoi[0, ipt] = val
        if lev > 0 and verbose:
            print(
                "                   QoI_err(%)  = ",
                str(abs(val - qoi_i[0]) / qoi_i[0] * 100),
            )
            # print(val,qoi_i[0])
    # add new snapshots to grid
    grid.loadNeededPoints(new_snap.T)
    grid_qoi.loadNeededPoints(new_snap_qoi.T)
    # print('H =',grid.getHierarchicalCoefficients())

    # Result = grid.evaluate(PointOfInterest)
    # Error = np.abs(Result[0] - ReferenceSolution[0])
    # print('Result            = ',Result[0])
    # print('ReferenceSolution = ',ReferenceSolution[0])
    # print('Error             = ',Error)
    # grid.plotPoints2D(plt)
    # plt.show()

    shrink_rank = True
    if do_isvd:
        fig = plt.figure(11)
        if lev == 0:
            RR = np.copy(new_snap)
            u, sv, vh = np.linalg.svd(new_snap, full_matrices=False, compute_uv=True)
        else:
            u, sv, vh = addblock_svd_update(u, sv, vh, new_snap, force_orth=False)
            # print('sv lev++++=',sv)
            if shrink_rank:
                # if reducting is not large, keep all:
                if False:  # sv[0]/sv[-1]<1e3:
                    r = len(sv)
                else:
                    # energy content
                    ec = energy_content(sv)
                    # Choose the number of basis functions to keep to preserve e
                    r = np.argwhere(ec - e_threshold >= 0.0)[0][0]
                print("rank=", r)
                # Get reduced basis
                u = np.copy(u[:, 0:r])
                sv = np.copy(sv[0:r])
                vh = np.copy(vh[0:r, :])
        mark = ""
        if lev == 0:
            mark = "+"
        plt.semilogy(sv, marker=mark, label="lev-" + str(lev))
    if lev < max_level - 1:
        grid.updateGlobalGrid(lev + 1, "level")
        grid_qoi.updateGlobalGrid(lev + 1, "level")
        print("num points        = ", grid.getNumPoints())
        print("num points loaded = ", grid.getNumLoaded())
        print("num points needed = ", grid.getNumNeeded())
        # print('num points needed = ',grid.getNeededPoints())
    if iNumDimensions == 2:
        fig = plt.figure(100 + lev)
        grid.plotPoints2D(plt)
        plt.show()
plt.figure(11)
plt.grid()
plt.legend()
plt.show()

##########################################################
#
# Use SG generated with random points
#
ns = 200
xy_min = -np.ones(iNumDimensions)
xy_max = np.ones(iNumDimensions)
# xy_max[-1]=-0.5
TestPoints = np.random.uniform(low=xy_min, high=xy_max, size=(ns, len(xy_min)))
# get new values of parameters
cdif_, siga_, qext_, Jinc_, Jneu_ = new_param_values(
    cdif, siga, qext, Jinc, Jneu, limits, which_to_pert, TestPoints
)

# use SG interpolant
Phi_i = grid.evaluateBatch(TestPoints)
qoi_i = grid_qoi.evaluateBatch(TestPoints)

# compute exact values
ii = np.array([], dtype=int)
for ipt in range(TestPoints.shape[0]):
    # get system and solve
    A, b = lin_op.build_diffusion_system(
        qext_[ipt, :], cdif_[ipt, :], siga_[ipt, :], bc, Jinc_[ipt, :], Jneu_[ipt, :]
    )
    Phi = scipy.sparse.linalg.spsolve(A, b)
    qoi = (QoI_op @ Phi)[0, 0]
    qoi_rel_err = abs(qoi - qoi_i[ipt, 0]) / qoi * 100
    flx_err = np.linalg.norm(Phi - Phi_i[ipt, :])
    print(
        "ipts={0:>3d}/{1:>d}, L2_err(flx)={2:,>+.8f}, QoI_err(%)={3:,>+.8f}".format(
            ipt, TestPoints.shape[0] - 1, flx_err, qoi_rel_err
        )
    )
    if qoi_rel_err > 5:
        ii = np.append(ii, ipt)
        print("   Pts = ", cdif_[ipt, :], siga_[ipt, :], qext_[ipt, :])
        # print('ipts={0:>3d}/{1:>d}, L2_err(flx)={2:,>+.8f}, QoI_err(%)={3:,>+.8f}'.format(ipt,TestPoints.shape[0]-1,flx_err,qoi_rel_err))
    print("   qoi_i = ", qoi_i[ipt, 0], ", qoi   = ", qoi)
    # print('   Pts = ',TestPoints[ipt,:])

    # fig = plt.figure(figsize=(15, 5), dpi= 80, facecolor='w', edgecolor='k')
    # ax = fig.add_subplot(1, 3, 1, projection='3d')
    # ax.plot_trisurf(mesh.pt2xy[:,0], mesh.pt2xy[:,1], Phi, linewidth=0.2, cmap=plt.cm.CMRmap)
    # ax.set_title('exact')
    # ax = fig.add_subplot(1, 3, 2, projection='3d')
    # ax.plot_trisurf(mesh.pt2xy[:,0], mesh.pt2xy[:,1], Phi_i[ipt,:], linewidth=0.2, cmap=plt.cm.CMRmap)
    # ax.set_title('interp')
    # ax = fig.add_subplot(1, 3, 3, projection='3d')
    # ax.plot_trisurf(mesh.pt2xy[:,0], mesh.pt2xy[:,1], Phi-Phi_i[ipt,:], linewidth=0.2, cmap=plt.cm.CMRmap)
    # ax.set_title('error')
    # plt.show()
    # ax.set_title('error')
    # plt.show()
"""
# -*- coding: utf-8 -*-tasm

"""
Created on Sat Apr  4 22:54:04 2020

@author: ragusa
"""
import numpy as np
import scipy.sparse.linalg

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from addblock_svd_update import addblock_svd_update
from energy_content import energy_content
from sampler import sampler

plt.close("all")
##########################################################
#
# geometry and mesh
#
geo_id_ = 1

"""
# this part does not work on windows machine
from create_geometry import create_geometry

# create specific geometry
# id=1, vol=1e-2--> 151 elements
# id=2, vol=1e1 -->  62 elements
# id=3, vol=1e2 --> 130 elements

def rf(vertices, area):
    bary = np.sum(np.array(vertices), axis=0) / 3
    x,y = bary[0], bary[1]
    rad = np.sqrt(x*x+y*y)
    a=0.01/5
    if rad>0.8:
        max_area = a
    elif rad>0.6:
        max_area = a/2
    elif rad>0.4:
        max_area = a/4
    elif rad>0.2:
        max_area = a/8
    else:
        max_area = a/16
    # max_area = 0.1
    return bool(area > max_area)

el2pt, el2at, fa2pt, fa2ma, pt2xy, el2ne = create_geometry(geo_id=geo_id_,max_vol=1e-3,\
                                                           refinement_funct=rf,do_plot=False)
"""
# whether or not the geometry was created with a refinement function or not
with_ref = False
if with_ref:
    ref_txt = "_REF"
else:
    ref_txt = ""
# geo_id      nelems
#  1       | 151 , 1553 , 3085 , 15620
#  2       | 599 , 6266 , 62278
#  4       | 104
#  5       | 153 , 322  , 1544
#  5 REF   | 2258, 4413
#  6       | 316 , 1530, 3049, 15221
#  7       | 159 , 319 , 1538, 1557, 5118
#  8       | 1 , 80
#  9       | 154, 314, 1542, 3025, 5080, 15246

# create basename to reload data
basename = (
    "./geo-" + str(geo_id_) + "/geo_id" + str(geo_id_) + ref_txt + "_elems" + "151"
)

el2pt = np.loadtxt(basename + "_el2pt.txt", dtype=int)
el2at = np.loadtxt(basename + "_el2at.txt", dtype=int)
fa2pt = np.loadtxt(basename + "_fa2pt.txt", dtype=int)
fa2ma = np.loadtxt(basename + "_fa2ma.txt", dtype=int)
pt2xy = np.loadtxt(basename + "_pt2xy.txt")
el2ne = np.loadtxt(basename + "_el2ne.txt", dtype=int)
# raise ValueError('stopping')

# finish mesh
from tri_mesh import TriMesh

mesh = TriMesh(el2pt, el2at, fa2pt, fa2ma, pt2xy, el2ne)
mesh.complete()

# to plot the mesh
# mesh.plot_mesh()

# if you want to pickle the data or not
# import pickle as pickle
# with open('mesh'+str(geo_id_)+'.pkl', 'wb') as outp:
#     pickle.dump(mesh, outp, pickle.HIGHEST_PROTOCOL)

# with open('mesh'+str(geo_id_)+'.pkl', 'rb') as inp:
#     mesh = pickle.load(inp)

##########################################################
#
# data for that problem
#
# Caveat: cdif, qext must be entered in the order the materials are listed
# in mesh.attr
if geo_id_ == 1:
    bc_rob = {
        "markers": np.array([1, 2, 3, 4], dtype=int),
        "values": np.array([0, 0, 0, 1]),
    }
    bc = {"Robin": bc_rob}

    cdif = np.array([1, 3], dtype=float)
    qext = np.array([100, 0], dtype=float)
    Jinc = np.array([0, 0, 0, 1], dtype=float)
    limits = np.array([])
else:
    raise ValueError("unknown geo_id = ", geo_id_)
mesh.check_bc(bc)

##########################################################
#
# linear operators per attribute
#
from lin_op import LinOp

lin_op = LinOp(mesh)
lin_op.build_diffusion_op_per_attribute()
lin_op.build_diffusion_op_per_bc(bc)
"""
plt.close('all')
plt.figure(0)
plt.spy(lin_op.M[0],marker='.',ms=3.)
plt.spy(lin_op.M[1],marker='.',ms=1.,color='red')
plt.figure(1)
plt.spy(lin_op.K[0],marker='.',ms=3.)
plt.spy(lin_op.K[1],marker='.',ms=1.,color='red')"""

##########################################################
#
# Build system and solve
#
siga = cdif * 0
Jneu = Jinc * 0
A, b = lin_op.build_diffusion_system(qext, cdif, siga, bc)
Phi = scipy.sparse.linalg.spsolve(A, b)

if mesh.npts < 10000:
    fig = plt.figure(99)
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection="3d")
    ax.plot_trisurf(
        mesh.pt2xy[:, 0], mesh.pt2xy[:, 1], Phi, linewidth=0.2, cmap=plt.cm.CMRmap
    )
    plt.show()
##########################################################
#
# Prepare for data perturbations
#
def new_param_values_list(cdif, siga, qext, bc, limits, which_to_pert, Pts):
    # Pts: nsamples x effective_npert
    # print('Points');print(Pts)
    nsamples = Pts.shape[0]
    npert = Pts.shape[1]
    if npert != len(np.where(which_to_pert == True)[0]):
        raise ValueError(
            "Pts.shape[1] and len( np.where(which_to_pert==True)[0] ) mismatch"
        )
    # duplicate the nominal values
    aux = np.append(cdif, siga)
    aux = np.append(aux, qext)
    for key in bc.keys():
        aux = np.append(aux, bc[key]["values"])
    one = np.ones([nsamples, 1])
    tmp = np.kron(aux, one)  # nsamples x npert
    if len(which_to_pert) != limits.shape[0]:
        raise ValueError("len(which_to_pert) != limits.shape[0]")
    counter = 0
    for p, logical in enumerate(which_to_pert):
        if logical:
            xi = Pts[:, counter]
            # x = x_ave + (x2-x1)/2 * xi
            ave = (limits[p, 1] + limits[p, 0]) / 2
            slope = (limits[p, 1] - limits[p, 0]) / 2
            tmp[:, p] = ave + xi * slope
            counter += 1
    # place in array to be returned
    n1 = 0
    n2 = len(cdif)
    cdif_ = np.array(tmp[:, n1:n2])
    n1 = n2
    n2 = n1 + len(siga)
    siga_ = np.array(tmp[:, n1:n2])
    n1 = n2
    n2 = n1 + len(qext)
    qext_ = np.array(tmp[:, n1:n2])
    bc_val = {}
    for key in bc.keys():
        bc_val[key] = {}
        n1 = n2
        n2 = n1 + len(bc[key]["values"])
        bc_val[key]["values"] = np.array(tmp[:, n1:n2])
    return cdif_, siga_, qext_, bc_val


if geo_id_ == 1:
    which_to_pert = np.zeros(10, dtype=bool)
    which_to_pert[0:5] = True
    # which_to_pert[5] = True
    # limits = np.array([[0.1,3], [0.1,5],\
    #                   [5,75], [0.1,3],\
    #                   [2,200], [0.1,1],\
    #                   [0.1,1], [0.1,1], [0.1,1],[0.1,1],\
    #                   [0.1,1], [0.1,1], [0.1,1],[0.1,1]])

    ff = 0.92
    a0 = np.array([1 - ff, 1 + ff])
    limits = np.array(
        [
            cdif[0] * a0,
            cdif[1] * a0,
            siga[0] * a0,
            siga[1] * a0,
            qext[0] * a0,
            qext[1] * a0,
            0 * a0,
            0 * a0,
            0 * a0,
            Jinc[-1] * a0,
        ]
    )

    # hardcoded for this problem
    if len(which_to_pert) != 10:
        raise ValueError(
            "for geo_id=1, we need 10 params: D(2), siga(2), Q(2), Jinc(4)"
        )
    Pts = 0.3 * np.ones((3, 5))
    cdif_, siga_, qext_, bc_vals = new_param_values_list(
        cdif, siga, qext, bc, limits, which_to_pert, Pts
    )
else:
    raise ValueError("which_to_pert stills need to be implemented for other geometries")
# number of input space dimensions
iNumDimensions = len(np.where(which_to_pert == True)[0])

##########################################################
#
# Classic POD
#
# get samples in [-1,+1]^dim
number_of_snapshots = 250
use_LHS = False
TrainPoints = sampler(number_of_snapshots, iNumDimensions, use_LHS)

# get new values of parameters
cdif_, siga_, qext_, bc_vals = new_param_values_list(
    cdif, siga, qext, bc, limits, which_to_pert, TrainPoints
)


def make_bc_dict(bc, bc_vals, ind):
    bc_ = {}
    for key in bc.keys():
        bc_[key] = {}
        bc_[key]["markers"] = np.copy(bc[key]["markers"])
        bc_[key]["values"] = bc_vals[key]["values"][ind, :]
    return bc_


# compute snapshot values
pod_snapshots = np.zeros((mesh.npts, number_of_snapshots))
for ipt in range(TrainPoints.shape[0]):
    # get system and solve
    bc_ = make_bc_dict(bc, bc_vals, ipt)
    A, b = lin_op.build_diffusion_system(
        qext_[ipt, :], cdif_[ipt, :], siga_[ipt, :], bc_
    )
    Phi = scipy.sparse.linalg.spsolve(A, b)
    pod_snapshots[:, ipt] = Phi
# perform SVD
u, sv, vh = np.linalg.svd(pod_snapshots, full_matrices=False, compute_uv=True)
# energy content
ec = energy_content(sv)
# Choose the number of basis functions to keep to preserve e
e_threshold = 0.999
rank = np.argwhere(ec - e_threshold >= 0.0)[0][0]
print("POD rank = ", rank)
plt.figure(1)
# plt.semilogy(sv, label='POD-'+str(number_of_snapshots))
plt.semilogy(sv, label="POD-")
plt.legend()
plt.show()
# Get reduced basis
ur = np.copy(u[:, 0:rank])
# compute affine decomposition
lin_op.compute_reduced_operators(ur, bc)

plt.figure(2)
nn = np.minimum(number_of_snapshots, 2 * rank)
plt.plot(ec[0:nn])
plt.show()


# testing
number_of_tests = 50
TestPoints = sampler(number_of_tests, iNumDimensions, use_LHS)
# get new values of parameters
cdif_, siga_, qext_, bc_ = new_param_values_list(
    cdif, siga, qext, bc, limits, which_to_pert, TestPoints
)
rela_err = np.zeros(TestPoints.shape[0])
for ipt in range(TestPoints.shape[0]):
    # get system and solve
    bc_ = make_bc_dict(bc, bc_vals, ipt)
    Ar, br = lin_op.build_reduced_system(
        qext_[ipt, :], cdif_[ipt, :], siga_[ipt, :], bc_
    )
    c = np.linalg.solve(Ar, br)
    Phi_r = ur @ c
    A, b = lin_op.build_diffusion_system(
        qext_[ipt, :], cdif_[ipt, :], siga_[ipt, :], bc_
    )
    Phi = scipy.sparse.linalg.spsolve(A, b)
    flx_err = 100 * np.linalg.norm(Phi - Phi_r) / np.linalg.norm(Phi)
    rela_err[ipt] = flx_err
    # print('ipts={0:>3d}, L2_err(%flx)={1:,>+.8f}'.format(ipt,flx_err))
plt.figure(3)
plt.plot(rela_err)
plt.title("Flux relative error in %")
plt.show()
sv_classic = np.copy(sv)


##########################################################
#
# greedy POD
#
MiddlePoint = np.zeros((1, iNumDimensions))
# get new values of parameters
cdif_, siga_, qext_, bc_vals = new_param_values_list(
    cdif, siga, qext, bc, limits, which_to_pert, MiddlePoint
)
raise ValueError("a")

bc_ = make_bc_dict(bc, bc_vals, 0)
A, b = lin_op.build_diffusion_system(qext_, cdif_, siga_, bc_)
phi = scipy.sparse.linalg.spsolve(A, b)
new_snap = np.expand_dims(phi, axis=1)
ur, sv, vh = np.linalg.svd(new_snap, full_matrices=False, compute_uv=True)
plt.figure(9)
plt.semilogy(sv, marker="+", label="greedy-0")

nCandidates = 1000
nKeep = 50
iend = 0
not_done = True
igreedy = 0
total_snap = np.copy(new_snap)
while not_done:
    igreedy += 1
    print("Greedy iteration ", igreedy)
    # update ROM ops using latest ur bases
    lin_op.compute_reduced_operators(ur, bc)
    # select new set of training points
    TrainPoints = sampler(nCandidates, iNumDimensions, use_LHS)
    cdif_, siga_, qext_, bc_vals = new_param_values_list(
        cdif, siga, qext, bc, limits, which_to_pert, TrainPoints
    )
    # use ROM to evaluate residual error
    Candidate_Error = np.zeros(nCandidates)
    plt.figure(7)
    for ipt in range(nCandidates):
        bc_ = make_bc_dict(bc, bc_vals, ipt)
        Ar, br = lin_op.build_reduced_system(
            qext_[ipt, :], cdif_[ipt, :], siga_[ipt, :], bc_
        )
        c = np.linalg.solve(Ar, br)
        Candidate_Error[ipt] = lin_op.residual_indicator(
            qext_[ipt, :], cdif_[ipt, :], siga_[ipt, :], bc_, c
        )
        # debug
        """A, b = lin_op.build_diffusion_system( qext_[ipt,:], cdif_[ipt,:], siga_[ipt,:], \
                                          bc_)
        aux = lin_op.residual_indicator_brute_force(A,b,ur,c)
        print(ipt,Candidate_Error[ipt],aux,aux-Candidate_Error[ipt])"""
    plt.semilogy(Candidate_Error, label="greedy-" + str(igreedy))
    plt.legend()
    # sort by largest error
    worst_ind = np.argsort(Candidate_Error)
    ibeg = iend + 1
    iend = iend + nKeep
    if iend >= number_of_snapshots:
        iend = number_of_snapshots
        nKeep = iend - ibeg
        not_done = False
    # perform FOM solves for worst param values
    new_snap = np.zeros((mesh.npts, nKeep))
    for ipt in worst_ind[:nKeep]:
        bc_ = make_bc_dict(bc, bc_vals, ipt)
        A, b = lin_op.build_diffusion_system(
            qext_[ipt, :], cdif_[ipt, :], siga_[ipt, :], bc_
        )
        new_snap[:, ipt] = scipy.sparse.linalg.spsolve(A, b)
    # update the SVD decomposition
    ur, sv, vh = addblock_svd_update(ur, sv, vh, new_snap, force_orth=True)
    print(ur.shape)
    plt.figure(9)
    plt.semilogy(sv, label="greedy-" + str(igreedy))
    total_snap = np.hstack((total_snap, new_snap))
ur, sv, vh = np.linalg.svd(total_snap, full_matrices=False, compute_uv=True)
plt.semilogy(sv, marker="o", label="FULL")
plt.legend()
plt.show()
plt.figure(6)
plt.semilogy(sv_classic, label="classic")
plt.semilogy(sv, label="greedy")
plt.legend()
plt.show()

# MiddlePoint = 0.5*np.ones(iNumDimensions)
# # get new values of parameters
# cdif_,siga_,qext_,Jinc_,Jneu_ = new_param_values_single(cdif,siga,qext,Jinc,Jneu,\
#                                                      limits,which_to_pert,MiddlePoint)
#
# qext_[:]=0
# A, b = lin_op.build_diffusion_system( qext_, cdif_, siga_, bc, Jinc_, Jneu_)
# ur = np.random.uniform(size=[mesh.npts,1])
# print(ur.shape)
#
# lin_op.compute_reduced_operators(ur)
# Ar, br = lin_op.build_reduced_system( qext_, cdif_, siga_, bc, Jinc_, Jneu_)
# c = np.linalg.solve(Ar,br)
# auxr = lin_op.residual_indicator(qext_, cdif_, siga_, bc, Jinc_, Jneu_, c)[0]
# aux = lin_op.residual_indicator_brute_force(A,b,ur,c)
# print(auxr,aux,aux/auxr)
# print(Ar, ur.T@A@ur)
# print(br, ur.T@b)
raise ValueError("stop")

"""
##########################################################
#
# Black box model
#
def model(
    Point, is_qoi, cdif, siga, qext, Jinc, Jneu, limits, which_to_pert, lin_op, QoI_op
):
    # print("Model Point = ",Point)
    # get new values of parameters
    cdif_, siga_, qext_, Jinc_, Jneu_ = new_param_values_single(
        cdif, siga, qext, Jinc, Jneu, limits, which_to_pert, Point
    )
    # get system and solve
    A, b = lin_op.build_diffusion_system(qext_, cdif_, siga_, bc, Jinc_, Jneu_)
    Phi = scipy.sparse.linalg.spsolve(A, b)
    if is_qoi:
        val_qoi = (QoI_op @ Phi)[0, 0]
        return np.array([val_qoi])
    else:
        return Phi


############################################
#
# Construct sparse grid
#
import Tasmanian

####
# incremental SVD stuff
do_isvd = True
shrink_rank = True
e_threshold = 0.9999
####

# grid params
iInitialLevel = 1
max_level = 3
iNumThreads = 1

# make grid
grid = Tasmanian.SparseGrid()
iNumOutputs = mesh.npts
grid.makeGlobalGrid(
    iNumDimensions, iNumOutputs, iInitialLevel, "iptotal", "clenshaw-curtis"
)
# grid.makeGlobalGrid(iNumDimensions, iNumOutputs, iInitialLevel, "level", "clenshaw-curtis")
# make grid for qoi
grid_qoi = Tasmanian.SparseGrid()
# grid_qoi.makeGlobalGrid(iNumDimensions, 1, iInitialLevel, "level", "clenshaw-curtis")
grid_qoi.makeGlobalGrid(iNumDimensions, 1, iInitialLevel, "iptotal", "clenshaw-curtis")
# QoI operator
QoI_op = lin_op.M[0].sum(axis=0)

iBudget = 50
bBelowBudget = True
iteration = 0
while bBelowBudget:
    print("\n------------------------------")
    print(
        "Iteration             = ",
        iteration,
        " nbr pts = ",
        grid.getNumLoaded(),
        " nbr pts = ",
        grid_qoi.getNumLoaded(),
    )
    is_qoi = False
    Tasmanian.loadNeededPoints(
        lambda x, tid: model(
            x,
            is_qoi,
            cdif,
            siga,
            qext,
            Jinc,
            Jneu,
            limits,
            which_to_pert,
            lin_op,
            QoI_op,
        ),
        grid,
        iNumThreads,
    )
    grid.setAnisotropicRefinement("iptotal", 10, 0)
    is_qoi = True
    Tasmanian.loadNeededPoints(
        lambda x, tid: model(
            x,
            is_qoi,
            cdif,
            siga,
            qext,
            Jinc,
            Jneu,
            limits,
            which_to_pert,
            lin_op,
            QoI_op,
        ),
        grid_qoi,
        iNumThreads,
    )
    grid_qoi.setAnisotropicRefinement("iptotal", 10, 0)

    #   shrink_rank=True ...
    bBelowBudget = grid.getNumLoaded() < iBudget and grid_qoi.getNumLoaded() < iBudget
    iteration += 1

    if iNumDimensions == 2:
        fig, axs = plt.subplots(1, 2)
        grid.plotPoints2D(axs[0])
        grid_qoi.plotPoints2D(axs[1])
        plt.show()
print(
    "nbr pts grid = ",
    grid.getNumLoaded(),
    " nbr pts QoI grid = ",
    grid_qoi.getNumLoaded(),
)

# ##########################################################
# #
# # Old SG contruction
# #
#
#
# grid = Tasmanian.SparseGrid()
# iNumDimensions = len( np.where(which_to_pert==True)[0] )
# iNumOutputs = mesh.npts
# iInitialLevel = 0
# max_level = 5
#
# grid.makeGlobalGrid(iNumDimensions, iNumOutputs, iInitialLevel, "iptotal", "clenshaw-curtis")
# # grid.makeGlobalGrid(iNumDimensions, iNumOutputs, iInitialLevel, "level", "clenshaw-curtis")
#
# # make grid for qoi
# grid_qoi = Tasmanian.SparseGrid()
# # grid_qoi.makeGlobalGrid(iNumDimensions, 1, iInitialLevel, "level", "clenshaw-curtis")
# grid_qoi.makeGlobalGrid(iNumDimensions, 1, iInitialLevel, "iptotal", "clenshaw-curtis")
# # QoI operator
# QoI_op=lin_op.M[0].sum(axis=0)
#
# for lev in range(iInitialLevel,max_level):
#     print("\n------------------------------")
#     print('Level             = ',lev)
#     Points = grid.getNeededPoints()
#
#     # get new values of parameters
#     cdif_,siga_,qext_,Jinc_,Jneu_ = new_param_values_single(cdif,siga,qext,Jinc,Jneu,\
#                                                      limits,which_to_pert,Points)
#     #
#     verbose = True
#     new_snap = np.zeros( (mesh.npts, Points.shape[0]) );
#     new_snap_qoi = np.zeros( (1,Points.shape[0]) );
#     for ipt in range(Points.shape[0]):
#         # get system and solve
#         A, b = lin_op.build_diffusion_system( qext_[ipt,:], cdif_[ipt,:], siga_[ipt,:], \
#                                              bc, Jinc_[ipt,:], Jneu_[ipt,:])
#         # interpolate on current grid if good enough
#         do_solve = True
#         if lev>0:
#             Phi_i = grid.evaluate(Points[ipt])
#             qoi_i = grid_qoi.evaluate(Points[ipt])
#             resi = np.linalg.norm( A @ Phi_i - b ) / np.linalg.norm(b)
#             print('Level '+str(lev)+', Point '+str(ipt)+', resi='+str(resi))
#             if verbose:
#                 print('Level '+str(lev)+', Point '+str(ipt)+'/'+str(Points.shape[0]-1)+', resi='+str(resi))
#             if resi>0.01:
#                 do_solve = True
#             #else:
#             #    do_solve = False
#             #    Phi = np.copy(Phi_i)
#         else:
#             if verbose:
#                 print('Level '+str(lev)+', Point '+str(ipt)+'/'+str(Points.shape[0]-1))
#         # solve if requested
#         if do_solve:
#             Phi = scipy.sparse.linalg.spsolve(A, b)
#         if lev>0 and verbose:
#             print('                   L2_err(flx) = ',str(np.linalg.norm(Phi-Phi_i)))
#         # add to new snapshot array
#         new_snap[:,ipt] = Phi[:]
#         val = (QoI_op@Phi)[0,0]
#         new_snap_qoi[0,ipt] = val
#         if lev>0 and verbose:
#             print('                   QoI_err(%)  = ',str(abs(val-qoi_i[0])/qoi_i[0]*100) )
#             # print(val,qoi_i[0])
#
#     # add new snapshots to grid
#     grid.loadNeededPoints(new_snap.T)
#     grid_qoi.loadNeededPoints(new_snap_qoi.T)
#     #print('H =',grid.getHierarchicalCoefficients())
#
#     #Result = grid.evaluate(PointOfInterest)
#     #Error = np.abs(Result[0] - ReferenceSolution[0])
#     print("\n------------------------------")
#     print('Level             = ',lev)
#     #print('Result            = ',Result[0])
#     #print('ReferenceSolution = ',ReferenceSolution[0])
#     #print('Error             = ',Error)
#     #grid.plotPoints2D(plt)
#     #plt.show()
#
#
#     shrink_rank=True
#     if do_isvd:
#         fig = plt.figure(11)
#         if lev==0:
#             RR = np.copy(new_snap)
#             u, sv, vh = np.linalg.svd(new_snap, full_matrices=False, compute_uv=True)
#         else:
#             u, sv, vh = addblock_svd_update(u, sv, vh, new_snap, force_orth=False)
#             #print('sv lev++++=',sv)
#             if shrink_rank:
#                 # if reducting is not large, keep all:
#                 if False: #sv[0]/sv[-1]<1e3:
#                     r = len(sv)
#                 else:
#                     # energy content
#                     ec = energy_content(sv)
#                     # Choose the number of basis functions to keep to preserve e
#                     r = np.argwhere(ec - e_threshold >= 0.0 )[0][0]
#                 print('rank=',r)
#                 # Get reduced basis
#                 u  = np.copy(u[:, 0:r])
#                 sv = np.copy(sv[0:r])
#                 vh = np.copy(vh[0:r, :])
#         mark=''
#         if lev==0:
#           mark='+'
#         plt.semilogy(sv, marker=mark, label='lev-'+str(lev))
#
#     if lev<max_level-1:
#         grid.updateGlobalGrid(lev+1, "level")
#         #grid_qoi.updateGlobalGrid(lev+1, "level")
#         #grid.setAnisotropicRefinement("iptotal", 10, 0);
#         grid_qoi.setAnisotropicRefinement("iptotal", 1, 0);
#         print('num points        = ',grid.getNumPoints())
#         print('num points loaded = ',grid.getNumLoaded())
#         print('num points needed = ',grid.getNumNeeded())
#         #print('num points needed = ',grid.getNeededPoints())
#
#     if iNumDimensions==2:
#         fig = plt.figure(100+lev)
#         grid.plotPoints2D(plt)
#         plt.show()
#
#
# fig = plt.figure(11)
# plt.grid()
# plt.legend()
# plt.show()
#


##########################################################
#
# Copy of SG while debugging anisotropic grids
#
############################################
# incremental SVD stuff
do_isvd = True
shrink_rank = True
e_threshold = 0.9999
############################################
import Tasmanian

grid = Tasmanian.SparseGrid()
iNumDimensions = len(np.where(which_to_pert == True)[0])
iNumOutputs = mesh.npts
iInitialLevel = 0
max_level = 5

grid.makeGlobalGrid(
    iNumDimensions, iNumOutputs, iInitialLevel, "iptotal", "clenshaw-curtis"
)
# grid.makeGlobalGrid(iNumDimensions, iNumOutputs, iInitialLevel, "level", "clenshaw-curtis")

# make grid for qoi
grid_qoi = Tasmanian.SparseGrid()
# grid_qoi.makeGlobalGrid(iNumDimensions, 1, iInitialLevel, "level", "clenshaw-curtis")
grid_qoi.makeGlobalGrid(iNumDimensions, 1, iInitialLevel, "iptotal", "clenshaw-curtis")
# QoI operator
QoI_op = lin_op.M[0].sum(axis=0)

for lev in range(iInitialLevel, max_level):
    print("\n------------------------------")
    print("Level             = ", lev)
    Points = grid.getNeededPoints()

    # get new values of parameters
    cdif_, siga_, qext_, Jinc_, Jneu_ = new_param_values(
        cdif, siga, qext, Jinc, Jneu, limits, which_to_pert, Points
    )
    #
    verbose = False
    new_snap = np.zeros((mesh.npts, Points.shape[0]))
    new_snap_qoi = np.zeros((1, Points.shape[0]))
    for ipt in range(Points.shape[0]):
        # get system and solve
        A, b = lin_op.build_diffusion_system(
            qext_[ipt, :],
            cdif_[ipt, :],
            siga_[ipt, :],
            bc,
            Jinc_[ipt, :],
            Jneu_[ipt, :],
        )
        # interpolate on current grid if good enough
        do_solve = True
        if lev > 0:
            Phi_i = grid.evaluate(Points[ipt])
            qoi_i = grid_qoi.evaluate(Points[ipt])
            resi = np.linalg.norm(A @ Phi_i - b) / np.linalg.norm(b)
            if verbose:
                print(
                    "Level "
                    + str(lev)
                    + ", Point "
                    + str(ipt)
                    + "/"
                    + str(Points.shape[0] - 1)
                    + ", resi="
                    + str(resi)
                )
            if resi > 0.01:
                do_solve = True
            # else:
            #    do_solve = False
            #    Phi = np.copy(Phi_i)
        else:
            if verbose:
                print(
                    "Level "
                    + str(lev)
                    + ", Point "
                    + str(ipt)
                    + "/"
                    + str(Points.shape[0] - 1)
                )
        # solve if requested
        if do_solve:
            Phi = scipy.sparse.linalg.spsolve(A, b)
        if lev > 0 and verbose:
            print("                   L2_err(flx) = ", str(np.linalg.norm(Phi - Phi_i)))
        # add to new snapshot array
        new_snap[:, ipt] = Phi[:]
        val = (QoI_op @ Phi)[0, 0]
        new_snap_qoi[0, ipt] = val
        if lev > 0 and verbose:
            print(
                "                   QoI_err(%)  = ",
                str(abs(val - qoi_i[0]) / qoi_i[0] * 100),
            )
            # print(val,qoi_i[0])
    # add new snapshots to grid
    grid.loadNeededPoints(new_snap.T)
    grid_qoi.loadNeededPoints(new_snap_qoi.T)
    # print('H =',grid.getHierarchicalCoefficients())

    # Result = grid.evaluate(PointOfInterest)
    # Error = np.abs(Result[0] - ReferenceSolution[0])
    # print('Result            = ',Result[0])
    # print('ReferenceSolution = ',ReferenceSolution[0])
    # print('Error             = ',Error)
    # grid.plotPoints2D(plt)
    # plt.show()

    shrink_rank = True
    if do_isvd:
        fig = plt.figure(11)
        if lev == 0:
            RR = np.copy(new_snap)
            u, sv, vh = np.linalg.svd(new_snap, full_matrices=False, compute_uv=True)
        else:
            u, sv, vh = addblock_svd_update(u, sv, vh, new_snap, force_orth=False)
            # print('sv lev++++=',sv)
            if shrink_rank:
                # if reducting is not large, keep all:
                if False:  # sv[0]/sv[-1]<1e3:
                    r = len(sv)
                else:
                    # energy content
                    ec = energy_content(sv)
                    # Choose the number of basis functions to keep to preserve e
                    r = np.argwhere(ec - e_threshold >= 0.0)[0][0]
                print("rank=", r)
                # Get reduced basis
                u = np.copy(u[:, 0:r])
                sv = np.copy(sv[0:r])
                vh = np.copy(vh[0:r, :])
        mark = ""
        if lev == 0:
            mark = "+"
        plt.semilogy(sv, marker=mark, label="lev-" + str(lev))
    if lev < max_level - 1:
        grid.updateGlobalGrid(lev + 1, "level")
        grid_qoi.updateGlobalGrid(lev + 1, "level")
        print("num points        = ", grid.getNumPoints())
        print("num points loaded = ", grid.getNumLoaded())
        print("num points needed = ", grid.getNumNeeded())
        # print('num points needed = ',grid.getNeededPoints())
    if iNumDimensions == 2:
        fig = plt.figure(100 + lev)
        grid.plotPoints2D(plt)
        plt.show()
plt.figure(11)
plt.grid()
plt.legend()
plt.show()

##########################################################
#
# Use SG generated with random points
#
ns = 200
xy_min = -np.ones(iNumDimensions)
xy_max = np.ones(iNumDimensions)
# xy_max[-1]=-0.5
TestPoints = np.random.uniform(low=xy_min, high=xy_max, size=(ns, len(xy_min)))
# get new values of parameters
cdif_, siga_, qext_, Jinc_, Jneu_ = new_param_values(
    cdif, siga, qext, Jinc, Jneu, limits, which_to_pert, TestPoints
)

# use SG interpolant
Phi_i = grid.evaluateBatch(TestPoints)
qoi_i = grid_qoi.evaluateBatch(TestPoints)

# compute exact values
ii = np.array([], dtype=int)
for ipt in range(TestPoints.shape[0]):
    # get system and solve
    A, b = lin_op.build_diffusion_system(
        qext_[ipt, :], cdif_[ipt, :], siga_[ipt, :], bc, Jinc_[ipt, :], Jneu_[ipt, :]
    )
    Phi = scipy.sparse.linalg.spsolve(A, b)
    qoi = (QoI_op @ Phi)[0, 0]
    qoi_rel_err = abs(qoi - qoi_i[ipt, 0]) / qoi * 100
    flx_err = np.linalg.norm(Phi - Phi_i[ipt, :])
    print(
        "ipts={0:>3d}/{1:>d}, L2_err(flx)={2:,>+.8f}, QoI_err(%)={3:,>+.8f}".format(
            ipt, TestPoints.shape[0] - 1, flx_err, qoi_rel_err
        )
    )
    if qoi_rel_err > 5:
        ii = np.append(ii, ipt)
        print("   Pts = ", cdif_[ipt, :], siga_[ipt, :], qext_[ipt, :])
        # print('ipts={0:>3d}/{1:>d}, L2_err(flx)={2:,>+.8f}, QoI_err(%)={3:,>+.8f}'.format(ipt,TestPoints.shape[0]-1,flx_err,qoi_rel_err))
    print("   qoi_i = ", qoi_i[ipt, 0], ", qoi   = ", qoi)
    # print('   Pts = ',TestPoints[ipt,:])

    # fig = plt.figure(figsize=(15, 5), dpi= 80, facecolor='w', edgecolor='k')
    # ax = fig.add_subplot(1, 3, 1, projection='3d')
    # ax.plot_trisurf(mesh.pt2xy[:,0], mesh.pt2xy[:,1], Phi, linewidth=0.2, cmap=plt.cm.CMRmap)
    # ax.set_title('exact')
    # ax = fig.add_subplot(1, 3, 2, projection='3d')
    # ax.plot_trisurf(mesh.pt2xy[:,0], mesh.pt2xy[:,1], Phi_i[ipt,:], linewidth=0.2, cmap=plt.cm.CMRmap)
    # ax.set_title('interp')
    # ax = fig.add_subplot(1, 3, 3, projection='3d')
    # ax.plot_trisurf(mesh.pt2xy[:,0], mesh.pt2xy[:,1], Phi-Phi_i[ipt,:], linewidth=0.2, cmap=plt.cm.CMRmap)
    # ax.set_title('error')
    # plt.show()
    # ax.set_title('error')
    # plt.show()
"""
