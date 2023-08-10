# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 22:35:16 2020

@author: ragusa
"""

from __future__ import absolute_import
from __future__ import print_function
from six.moves import range
import numpy as np

import meshpy.triangle as triangle
import matplotlib.pyplot as plt

def round_trip_connect(start, end):
  result = []
  for i in range(start, end):
    result.append((i, i+1))
  result.append((end, start))
  return result

def create_geometry(geo_id=1, max_vol=1e-2, refinement_funct=None, do_plot=False):
    if geo_id==1:
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
        # create inner points
        points = [ (0.3,0.3),(0.6,0.3),(0.6,0.4),(0.4,0.4)]
        facets = round_trip_connect(0, len(points)-1)
        markers = [0,0,0,0]

        # create outer points
        outter_start = len(points)
        points.extend([(1,1),(0,1),(0,0.5),(0,0.25),(0,0),(1,0)])
        facets.extend(round_trip_connect(outter_start, len(points) - 1))
        markers.extend([2,3,4,3,2,1])

        info = triangle.MeshInfo()
        # populate information
        info.set_points(points)
        info.set_facets(facets, facet_markers=markers)
        # add region attributes
        info.regions.resize(2)
        info.regions[0] = (
                    # point in region
                    [0.5, 0.35] + [
                        # region number
                        1,
                        # max volume in region
                        0.001])
        info.regions[1] = (
                    # point in region
                    [0.1, 0.1] + [
                        # region number
                        2,
                        # max volume in region
                        0.001])
        axis_range = [-0.15, 1.15, -0.15, 1.15]
    elif geo_id==2:
    #  11-----------------------10
    #   |                       |
    #   |                       |
    #   |                       |
    #   |     7-----------6     |
    #   |     |           |     |
    #   |     |   3---2   |     |
    #   |     |   | . |   |     |
    #   |     |   0---1   |     |
    #   |     |           |     |
    #   |     4-----------5     |
    #   |                       |
    #   |                       |
    #   |                       |
    #   8-----------------------9
    #
        # create inner square points
        points = [ (-2,-2),(2,-2),(2,2),(-2,2)]
        facets = round_trip_connect(0, len(points)-1)
        markers = [0,0,0,0]

        # create middle square points
        outter_start = len(points)
        points.extend([ (-4,-4),(4,-4),(4,4),(-4,4)])
        facets.extend(round_trip_connect(outter_start, len(points) - 1))
        markers.extend([0,0,0,0])

        # create outer square points
        outter_start = len(points)
        points.extend([ (-10,-10),(10,-10),(10,10),(-10,10)])
        facets.extend(round_trip_connect(outter_start, len(points) - 1))
        markers.extend([1,1,1,1])

        info = triangle.MeshInfo()
        # populate information
        info.set_points(points)
        info.set_facets(facets, facet_markers=markers)
        # add region attributes
        info.regions.resize(3)
        info.regions[0] = (
                    # point in region
                    [0, 0] + [
                        # region number
                        1,
                        # max volume in region
                        0.001])
        info.regions[1] = (
                    # point in region
                    [0, 3] + [
                        # region number
                        2,
                        # max volume in region
                        0.001])
        info.regions[2] = (
                    # point in region
                    [0, 5] + [
                        # region number
                        3,
                        # max volume in region
                        0.001])
        axis_range = [-10.15, 10.15, -10.15, 10.15]
    elif geo_id==3:
    #  12-----------------------11
    #   |                       |
    #   |                       |
    #   |                       |
    #   |     8-----3-----2     |
    #   |     |     |     |     |
    #   |     |   4 |   3 |     |
    #   |     7-----0-----1     |
    #   |     |     |     |     |
    #   |     |   1 |  2  |     |
    #   |     6-----4-----5     |
    #   |                       |
    #   |               5       |
    #   |                       |
    #   9-----------------------10
    #
        # create rectangle-3 points
        points = [ (0,0),(30,0),(30,25),(0,25) ]
        facets = round_trip_connect(0, len(points)-1)
        markers = [0,0,0,0]

        # create rectangle-2 points
        outter_start = len(points)
        points.extend([ (0,-25),(30,-25),(30,0),(0,0) ])
        facets.extend(round_trip_connect(outter_start, len(points) - 1))
        markers.extend([0,0,0,0])

        # create rectangle-1 points
        outter_start = len(points)
        points.extend([ (-30,-25),(0,-25),(0,0),(-30,0) ])
        facets.extend(round_trip_connect(outter_start, len(points) - 1))
        markers.extend([0,0,0,0])

        # create rectangle-4 points
        outter_start = len(points)
        points.extend([ (-30,0),(0,0),(0,25),(-30,25) ])
        facets.extend(round_trip_connect(outter_start, len(points) - 1))
        markers.extend([0,0,0,0])

        # create outer rectangle points
        outter_start = len(points)
        points.extend([ (-48,-43),(48,-43),(48,43),(-48,43)])
        facets.extend(round_trip_connect(outter_start, len(points) - 1))
        markers.extend([1,1,1,1])

        info = triangle.MeshInfo()
        # populate information
        info.set_points(points)
        info.set_facets(facets, facet_markers=markers)
        # add region attributes
        info.regions.resize(5)
        info.regions[0] = (
                    # point in region
                    [-1, -1] + [
                        # region number
                        1,
                        # max volume in region
                        0.001])
        info.regions[1] = (
                    # point in region
                    [1, -1] + [
                        # region number
                        2,
                        # max volume in region
                        0.001])
        info.regions[2] = (
                    # point in region
                    [1, 1] + [
                        # region number
                        3,
                        # max volume in region
                        0.001])
        info.regions[3] = (
                    # point in region
                    [-1, 1] + [
                        # region number
                        4,
                        # max volume in region
                        0.001])
        info.regions[4] = (
                    # point in region
                    [40, 0] + [
                        # region number
                        5,
                        # max volume in region
                        0.001])
        axis_range = [-50,50,-50,50]
    elif geo_id==4:
    #  15--------------------------14
    #   |                          |
    #   | 11------------------10   |
    #   |  |                   |   |
    #   |  |   7-----------6   |   |
    #   |  |   |           |   |   |
    #   |  |   |   3---2   |   |   |
    #   |  |   |   | . |   |   |   |
    #   |  |   |   0---1   |   |   |
    #   |  |   |           |   |   |
    #   |  |   4-----------5   |   |
    #   |  |                   |   |
    #   |  8-------------------9   |
    #   |                          |
    #  12--------------------------13
    #
        # create inner square points
        a=12.0
        points = [ (-a,-a),(a,-a),(a,a),(-a,a)]
        facets = round_trip_connect(0, len(points)-1)
        markers = [0,0,0,0]

        # create middle square points
        outter_start = len(points)
        a=15.0
        points.extend([ (-a,-a),(a,-a),(a,a),(-a,a)])
        facets.extend(round_trip_connect(outter_start, len(points) - 1))
        markers.extend([0,0,0,0])

        # create outer square points
        outter_start = len(points)
        a=21.0
        points.extend([ (-a,-a),(a,-a),(a,a),(-a,a)])
        facets.extend(round_trip_connect(outter_start, len(points) - 1))
        markers.extend([0,0,0,0])

        # create outer square points
        outter_start = len(points)
        a=30.0
        points.extend([ (-a,-a),(a,-a),(a,a),(-a,a)])
        facets.extend(round_trip_connect(outter_start, len(points) - 1))
        markers.extend([1,1,1,1])

        info = triangle.MeshInfo()
        # populate information
        info.set_points(points)
        info.set_facets(facets, facet_markers=markers)
        # add region attributes
        info.regions.resize(4)
        info.regions[0] = (
                    # point in region
                    [0, 0] + [
                        # region number
                        1,
                        # max volume in region
                        0.001])
        info.regions[1] = (
                    # point in region
                    [0, 13] + [
                        # region number
                        2,
                        # max volume in region
                        0.001])
        info.regions[2] = (
                    # point in region
                    [0, 16] + [
                        # region number
                        3,
                        # max volume in region
                        0.001])
        info.regions[3] = (
                    # point in region
                    [0, 25] + [
                        # region number
                        3,
                        # max volume in region
                        0.001])
        axis_range = [-30.15, 30.15, -30.15, 30.15]
    elif geo_id==5:
    #   3-----------------------2
    #   |                       |
    #   |                       |
    #   |                       |
    #   |                       |
    #   |                       |
    #   |                       |
    #   |                       |
    #   |                       |
    #   |                       |
    #   0-----------------------1
    #
        # create inner square points
        points = [ (0,0),(1,0),(1,1),(0,1)]
        facets = round_trip_connect(0, len(points)-1)
        markers = [1,2,3,4]

        info = triangle.MeshInfo()
        # populate information
        info.set_points(points)
        info.set_facets(facets, facet_markers=markers)
        # add region attributes
        info.regions.resize(1)
        info.regions[0] = (
                    # point in region
                    [0.5, 0.5] + [
                        # region number
                        1,
                        # max volume in region
                        0.001])
        axis_range = [-0.15, 1.15, -0.15, 1.15]
    elif geo_id==6:
    #   3-----------------------2
    #   |                       |
    #   |                       |
    #   |                       |
    #   |                       |
    #   |                       |
    #   4-----5                 |
    #         |                 |
    #         |                 |
    #         0-----------------1
    #
        # create inner square points
        a=0.15
        points = [ (a,0),(1,0),(1,1),(0,1),(0,a),(a,a)]
        facets = round_trip_connect(0, len(points)-1)
        markers = [1,1,1,1,99,99]

        info = triangle.MeshInfo()
        # populate information
        info.set_points(points)
        info.set_facets(facets, facet_markers=markers)
        # add region attributes
        info.regions.resize(1)
        info.regions[0] = (
                    # point in region
                    [0.5, 0.5] + [
                        # region number
                        1,
                        # max volume in region
                        0.001])
        axis_range = [-0.15, 1.15, -0.15, 1.15]
    elif geo_id==7:
    #   3-----------------------2
    #   |                       |
    #   |                       |
    #   |                       |
    #   |                       |
    #   |                       |
    #   4.                      |
    #      .                    |
    #        .                  |
    #         0-----------------1
    #
        # create inner square points
        a = 0.15
        points = [ (a,0),(1,0),(1,1),(0,1),(0,a)]
        facets = [(0,1), (1,2), (2,3), (3,4)]
        markers = [1,1,1,1]

        n_circumferential_points = 20
        radius = a
        # Create a list of points for the circle (centered on 0)
        for i in range(1,n_circumferential_points):
            angle = np.pi/2 * (1. - 1.*i / n_circumferential_points)
            points.append((radius * np.cos(angle), radius * np.sin(angle)))
        Nbeg = len(markers)
        # Define the segments for the circle
        for i in range(Nbeg,n_circumferential_points+Nbeg):
            if i+1 == n_circumferential_points+Nbeg:
                ip1 = 0
            else:
                ip1 = i+1
            facets.extend([(i,ip1)])
            markers.extend([99])
        
        # print(markers)
        # print(facets)
        # for p in points:
        #  print(p)
        # raise Exception('cc')

        info = triangle.MeshInfo()
        # populate information
        info.set_points(points)
        info.set_facets(facets, facet_markers=markers)
        # add region attributes
        info.regions.resize(1)
        info.regions[0] = (
                    # point in region
                    [0.5, 0.5] + [
                        # region number
                        1,
                        # max volume in region
                        0.001])
        axis_range = [-0.15, 1.15, -0.15, 1.15]
    elif geo_id==8:
    #               2
    #             /   \
    #            /     \
    #           /       \
    #          /         \
    #         /           \
    #        /             \
    #       /               \
    #      /                 \
    #     /                   \
    #    /                     \
    #   /                       \
    #   0-----------+-----------1
    #
        # create inner square points
        points = [ (-0.5,0),(0.5,0),(0,1)]
        facets = round_trip_connect(0, len(points)-1)
        markers = [99,1,2]

        info = triangle.MeshInfo()
        # populate information
        info.set_points(points)
        info.set_facets(facets, facet_markers=markers)
        # add region attributes
        info.regions.resize(1)
        info.regions[0] = (
                    # point in region
                    [0., 0.25] + [
                        # region number
                        1,
                        # max volume in region
                        0.001])
        axis_range = [-0.65, .65, -0.15, 1.15]
    elif geo_id==9:
    #   3-------------------------2
    #   |                         |
    #   |                         |
    #   |                         |
    #   |          9---8          |
    #   |          |   |          |
    #   |          6---7          |
    #   |                         |
    #   4-----5                   |
    #         |                   |
    #         |                   |
    #         0-------------------1
    #
        # create inner square points
        a=0.15
        points = [ (a,0),(1,0),(1,1),(0,1),(0,a),(a,a)]
        facets = round_trip_connect(0, len(points)-1)
        markers = [1,1,1,1,99,99]

        b=0.1
        outter_start = len(points)
        points.extend([ (0.5-b,0.5-b),(0.5+b,0.5-b),(0.5+b,0.5+b),(0.5-b,0.5+b)])
        facets.extend(round_trip_connect(outter_start, len(points) - 1))
        markers.extend([0,0,0,0])

        info = triangle.MeshInfo()
        # populate information
        info.set_points(points)
        info.set_facets(facets, facet_markers=markers)
        # add region attributes
        info.regions.resize(2)
        info.regions[0] = (
                    # point in region
                    [a/2, 0.5] + [
                        # region number
                        1,
                        # max volume in region
                        0.001])
        info.regions[1] = (
                    # point in region
                    [0.5, 0.5] + [
                        # region number
                        2,
                        # max volume in region
                        0.001])
        axis_range = [-0.15, 1.15, -0.15, 1.15]
    elif geo_id==10:
    #   3-------------------------2
    #   |                         |
    #   |                         |
    #   |          circle         |
    #   |            -            |
    #   |          |   |          |
    #   |            -            |
    #   |                         |
    #   |                         |
    #   |                         |
    #   0-------------------------1

    #
        # create inner square points
        points = [ (0,0),(1,0),(1,1),(0,1)]
        facets = [(0,1), (1,2), (2,3), (3,0)]
        markers = [1,1,1,1]

        n_circumferential_points = 4
        radius = 0.15
        # Create a list of points for the circle (centered on 0)
        for i in range(0,n_circumferential_points):
            angle = 2*np.pi * i / n_circumferential_points
            points.append((0.5 + radius * np.cos(angle), 0.5 + radius * np.sin(angle)))
        Nbeg = len(markers)
        # Define the segments for the circle
        for i in range(Nbeg,n_circumferential_points+Nbeg):
            if i+1 == n_circumferential_points+Nbeg:
                ip1 = 4
            else:
                ip1 = i+1
            facets.extend([(i,ip1)])
            markers.extend([99])

        # print(markers)
        # print(facets)
        # for p in points:
        #     print(p)
        # raise Exception('cc')

        info = triangle.MeshInfo()
        # populate information
        info.set_points(points)
        info.set_facets(facets, facet_markers=markers)
        # add region attributes
        info.regions.resize(1)
        info.regions[0] = (
                    # point in region
                    [0.5, 0.5] + [
                        # region number
                        1,
                        # max volume in region
                        0.001])

        info.set_holes([(0.5, 0.5)])
        axis_range = [-0.15, 1.15, -0.15, 1.15]
    else:
        raise ValueError('unknown geo id: ',geo_id)

    #print(info.dump())

    # generate triangulation
    mesh = triangle.build(info, verbose=False, refinement_func=refinement_funct,\
                          max_volume=max_vol, min_angle=25, attributes=True, generate_faces=True)
    # save it
    # triangle.write_gnuplot_mesh("triangles.dat", mesh)

    # do some more work
    print(("%d elements" % len(mesh.elements)))
    print(("%d vertices" % len(mesh.points)))
    #print("elem attributes")
    #print(np.array(mesh.element_attributes))
    #print("pt")
    #print(np.array(np.array(mesh.points)))
    #print("pt markers")
    #print(np.array(mesh.point_markers))
    #print("elements")
    #print(np.array(mesh.elements))
    #print("faces")
    #print(np.array(mesh.faces))
    #print("face markers")
    #print(np.array(mesh.face_markers))
    #print("regions")
    #print(np.array(mesh.regions))
    #print("neighbors")
    #print(np.array(mesh.neighbors))
    el2pt = np.array(mesh.elements,dtype=int)
    el2at = np.array(mesh.element_attributes,dtype=int)
    el2ne = np.array(mesh.neighbors,dtype=int)
    fa2pt = np.array(mesh.faces,dtype=int)
    fa2ma = np.array(mesh.face_markers,dtype=int)
    pt2xy = np.array(mesh.points)

    # plot it
    if do_plot:
        mesh_points = np.array(mesh.points)
        mesh_tris = np.array(mesh.elements)

        plt.figure(figsize=(9, 8), dpi= 160, facecolor='w', edgecolor='k')
        plt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_tris)
        plt.xlabel('x')
        plt.ylabel('y')
        #
        #inner_nodes = [i for i in range(n) if mesh_attr[i]==2]
        #outer_nodes = [i for i in range(n) if mesh_attr[i]==3]
        #plt.plot(mesh_points[inner_nodes, 0], mesh_points[inner_nodes, 1], 'ro')
        #plt.plot(mesh_points[outer_nodes, 0], mesh_points[outer_nodes, 1], 'go')
        plt.plot(mesh_points[:, 0], mesh_points[:, 1], 'ro',ms=3)
        plt.axis(axis_range)

        for iel in range(len(mesh.elements)):
            vert_list = el2pt[iel,:]
            x = np.sum(pt2xy[vert_list,0])/3
            y = np.sum(pt2xy[vert_list,1])/3
            attr = el2at[iel]
            text_ = str(attr) + ' (' + str(iel) + ')'
            plt.text(x,y,text_,color='red',fontsize=6)
        for ied in range(len(mesh.faces)):
            vert_list = fa2pt[ied,:]
            x = np.sum(pt2xy[vert_list,0])/2
            y = np.sum(pt2xy[vert_list,1])/2
            attr = fa2ma[ied]
            if attr != 0:
                plt.text(x,y,str(attr),color='black',backgroundcolor='yellow',fontsize=4)
        plt.show()

    return el2pt, el2at, fa2pt, fa2ma, pt2xy, el2ne
