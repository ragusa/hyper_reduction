#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 23:12:09 2023

@author: jean.ragusa
"""

# %% import python module
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# import matplotlib.tri as mtri
from six.moves import range

import meshpy.triangle as triangle

plt.close("all")

# %% small utility function to create facets given a list of points


def round_trip_connect(start, end):
    result = []
    for i in range(start, end):
        result.append((i, i + 1))
    result.append((end, start))
    return result


# %% definition of the 2D geometry using planar straight line segments
# to be then meshed with the Triangle mesh generator, via MeshPy

max_vol = 1e-2

#              (2)
#      5------------------4
#      |                  |
# (3)  |                  |
#      6                  |
#      |      3---2       |  (1)
# (4)  |     /    |       |
#      7    0-----1       |
#      |                  |
# (3)  |                  |
#      |                  |
#      8------------------9
#            (2)
# create inner points
points = [(0.3, 0.3), (0.6, 0.3), (0.6, 0.4), (0.4, 0.4)]
facets = round_trip_connect(0, len(points) - 1)
markers = [0, 0, 0, 0]

# create outer points
outter_start = len(points)
points.extend([(1, 1), (0, 1), (0, 0.5), (0, 0.25), (0, 0), (1, 0)])
facets.extend(round_trip_connect(outter_start, len(points) - 1))
markers.extend([2, 3, 4, 3, 2, 1])

info = triangle.MeshInfo()
# populate information
info.set_points(points)
info.set_facets(facets, facet_markers=markers)
# add region attributes
info.regions.resize(2)
info.regions[0] = (
    # point in region
    [0.5, 0.35]
    + [
        # region number
        1,
        # max volume in region
        0.001,
    ]
)
info.regions[1] = (
    # point in region
    [0.1, 0.1]
    + [
        # region number
        2,
        # max volume in region
        0.001,
    ]
)

# generate triangulation
mesh = triangle.build(
    info,
    verbose=False,
    refinement_func=None,
    max_volume=max_vol,
    min_angle=25,
    attributes=True,
    generate_faces=True,
)
# save it
# triangle.write_gnuplot_mesh("triangles.dat", mesh)
print(("%d elements" % len(mesh.elements)))
print(("%d vertices" % len(mesh.points)))

# %% extract meshing data into standard numpy ndarrays

# el2pt = np.array(mesh.elements, dtype=int)
print(mesh.elements[0])
el2pt = np.zeros((len(mesh.elements), 3), dtype=int)
for i in range(len(mesh.elements)):
    el2pt[i, :] = mesh.elements[i]

# el2at = np.array(mesh.element_attributes, dtype=int)
print(mesh.element_attributes[0])
el2at = np.zeros(len(mesh.element_attributes), dtype=int)
for i in range(len(mesh.element_attributes)):
    el2at[i] = mesh.element_attributes[i]

# el2ne = np.array(mesh.neighbors, dtype=int)
print(mesh.neighbors[0])
el2ne = np.zeros((len(mesh.neighbors), 3), dtype=int)
for i in range(len(mesh.neighbors)):
    el2ne[i, :] = mesh.neighbors[i]

# fa2pt = np.array(mesh.faces, dtype=int)
print(mesh.faces[0])
fa2pt = np.zeros((len(mesh.faces), 2), dtype=int)
for i in range(len(mesh.faces)):
    fa2pt[i, :] = mesh.faces[i]

# fa2ma = np.array(mesh.face_markers, dtype=int)
print(mesh.face_markers[0])
fa2ma = np.zeros(len(mesh.face_markers), dtype=int)
for i in range(len(mesh.face_markers)):
    fa2ma[i] = mesh.face_markers[i]

# pt2xy = np.array(mesh.points)
print(mesh.points[0])
pt2xy = np.zeros((len(mesh.points), 2))
for i in range(len(mesh.points)):
    pt2xy[i, :] = mesh.points[i]

# %% plot mesh if requested
do_plot = True
axis_range = [-0.15, 1.15, -0.15, 1.15]

if do_plot:

    plt.figure(figsize=(9, 8), dpi=160, facecolor="w", edgecolor="k")
    # plot trianlges (x-coord, y-coord, triangles)
    plt.triplot(pt2xy[:, 0], pt2xy[:, 1], el2pt)
    plt.xlabel("x")
    plt.ylabel("y")
    #
    # inner_nodes = [i for i in range(n) if mesh_attr[i]==0]
    # outer_nodes = [i for i in range(n) if mesh_attr[i]==1]
    # plt.plot(pt2xy[inner_nodes, 0], pt2xy[inner_nodes, 1], 'ro')
    # plt.plot(pt2xy[outer_nodes, 0], pt2xy[outer_nodes, 1], 'go')
    plt.plot(pt2xy[:, 0], pt2xy[:, 1], "ro", ms=3)
    plt.axis(axis_range)

    for iel in range(el2pt.shape[0]):
        vert_list = el2pt[iel, :]
        x = np.sum(pt2xy[vert_list, 0]) / 3
        y = np.sum(pt2xy[vert_list, 1]) / 3
        attr = el2at[iel]
        text_ = str(attr) + " (" + str(iel) + ")"
        plt.text(x, y, text_, color="red", fontsize=6)
    for ied in range(fa2pt.shape[0]):
        vert_list = fa2pt[ied, :]
        x = np.sum(pt2xy[vert_list, 0]) / 2
        y = np.sum(pt2xy[vert_list, 1]) / 2
        attr = fa2ma[ied]
        if attr != 0:
            plt.text(
                x, y, str(attr), color="black", backgroundcolor="yellow", fontsize=4
            )
    plt.show()

# %% save meshing data in txt files, if requested
save_meshing_data_to_txt = False

if save_meshing_data_to_txt:
    basename = 'Triangular_mesh_nelems' + str(len(el2pt[:, 0]))
    np.savetxt(basename+'_el2pt.txt', el2pt, fmt="%i")
    np.savetxt(basename+'_el2at.txt', el2at, fmt="%i")
    np.savetxt(basename+'_fa2pt.txt', fa2pt, fmt="%i")
    np.savetxt(basename+'_fa2ma.txt', fa2ma, fmt="%i")
    np.savetxt(basename+'_pt2xy.txt', pt2xy)
    np.savetxt(basename+'_el2ne.txt', el2ne, fmt="%i")

# %% convert to wavefront obj
dest_file = 'tri_2mat_bc.obj'

with open(dest_file, 'w') as f:
    f.write('# triangular mesh generated with Triangle thru MeshPy\n')
    f.write('# number of triangles: {}\n'.format(el2pt.shape[0]))
    f.write('# number of material attributes: {}\n'.format(
        len(np.unique(el2at))))
    f.write('# number of boundary markers: {}\n'.format(
        len(np.unique(fa2ma))))
    f.write('mtllib untitled.mtl \n')
    f.write('o main_object \n')

    # write mesh vertices
    for pt in pt2xy:
        f.write('v {} {} {}\n'.format(pt[0], pt[1], 0.))

    # write normal
    f.write('vn -0.0000 -0.0000 1.0000 \n')
    # write shading
    f.write('s 1 \n')

    # write triangles per material attribute
    for m in np.unique(el2at):
        f.write('g mat{} \n'.format(m))
        f.write('usemtl Mat{} \n'.format(m))
        for att, tri in zip(el2at, el2pt):
            if att == m:
                # converting to 1-indexing
                f.write(
                    'f {}//1 {}//1 {}//1 \n'.format(tri[0]+1, tri[1]+1, tri[2]+1))

    # find external edges (face marker>0)
    ind = np.where(fa2ma != 0)[0]
    fa2ma = fa2ma[ind]
    fa2pt = fa2pt[ind, :]

    # number of vertices in the main object
    nvert = pt2xy.shape[0]

    # write edges per bc attribute
    for bdID in np.unique(fa2ma):
        # get bd edge vertices
        vert_list = []
        f.write('o bd_object{} \n'.format(bdID))
        for bdatt, face in zip(fa2ma, fa2pt):
            if bdatt == bdID:
                vert_list.append(face)
        vertID = np.reshape(np.asarray(vert_list), (-1,))
        # make vertex ID unique
        vertID_uniq, inv_ind = np.unique(vertID, return_inverse=True)
        # write vertex coordinates
        for v in vertID_uniq:
            pt = pt2xy[v, :]
            f.write('v {} {} {}\n'.format(pt[0], pt[1], 0.))
        # write lines
        inv_ind = np.reshape(inv_ind, (-1, 2))
        for edg in inv_ind:
            # converting to 1-indexing
            # adding nvert
            f.write('l {} {} \n'.format(edg[0]+1+nvert, edg[1]+1+nvert))
        # increment number of nvert used after each boundary object
        nvert += len(vertID_uniq)
