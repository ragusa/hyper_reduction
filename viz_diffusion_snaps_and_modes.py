# -*- coding: utf-8 -*-

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import scipy.sparse.linalg


plt.close('all')
#######################################################
basename = 'geo-1-7812' 
extra = '_v2'

filename = basename + extra + '/' + basename + '_Mesh.npy'
pt2xy = np.load(filename)

filename = basename + extra + '/' + basename + '_Ur.npy'
ur = np.load(filename)
dofs = ur.shape[0]
rank = ur.shape[1]
                        
filename = basename + extra + '/' + basename + '_Snapshots.npy'
sol  = np.load(filename)


extra = extra + '/pix'

# for i in range(sol.shape[1]):
#     # fig = plt.figure(1)
#     Phi = sol[:,i]
#     # ax = fig.gca(projection='3d')
#     # ax.plot_trisurf(pt2xy[:,0], pt2xy[:,1], Phi, linewidth=0.2, cmap=plt.cm.CMRmap)
#     # plt.show()

#     fig = plt.figure(1)
#     plt.tricontourf(pt2xy[:,0], pt2xy[:,1], Phi, levels=19, cmap=plt.cm.coolwarm)
#     plt.show()
#     plt.savefig(basename + extra + '/' + basename + '_Sol_'+str(i)+'.png')

# Lx = np.max(pt2xy[:,0])
# Ly = np.max(pt2xy[:,1])
# pi = np.pi
# def f(x,y):
#     return np.sin(3*pi*x/Lx) * np.sin(4*pi*y/Ly)
# def f(x,y):
#     r = np.sqrt(x*x+y*y)
#     if r<1e-3: r=1e-3
#     return np.sin(3*pi*r/Lx) /r
# not_sol = np.zeros(dofs)
# for k in range(len(pt2xy[:,0])):
#     not_sol[k] = f(pt2xy[k,0],pt2xy[k,1])
# fig = plt.figure(2)
# plt.tricontourf(pt2xy[:,0], pt2xy[:,1], not_sol, levels=19, cmap=plt.cm.coolwarm)
# plt.show()
# plt.savefig(basename + extra + '/' + basename + 'npt_sol_2.png')

# for i in range(rank):
#     # fig = plt.figure(1)
#     # Phi = sol[:,i]
#     # ax = fig.gca(projection='3d')
#     # ax.plot_trisurf(pt2xy[:,0], pt2xy[:,1], Phi, linewidth=0.2, cmap=plt.cm.CMRmap)
#     # plt.show()

#     fig = plt.figure(2)
#     plt.tricontourf(pt2xy[:,0], pt2xy[:,1], ur[:,i], levels=19, cmap=plt.cm.coolwarm)
#     plt.show()
#     plt.savefig(basename + extra + '/' + basename + '_Pod_'+str(i)+'.png')


u, sv, vh = np.linalg.svd(sol, full_matrices=False, compute_uv=True)
plt.figure(1)
#plt.semilogy(sv, label='POD-'+str(number_of_snapshots))
plt.semilogy(sv, label='POD-')
plt.legend()
plt.title('Singular Value Decay')
plt.grid('on')
plt.show()