#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 23:17:29 2020

@author: ragusa
"""
import numpy as np
import matplotlib.pyplot as plt

###----------------------------------------------------------------------
class TriMesh:
###----------------------------------------------------------------------
    
###----------------------------------------------------------------------
    def __init__( self, el2pt, el2at, fa2pt, fa2ma, pt2xy, el2ne):

        # special case of single element mesh
        if el2at.size ==1:
            el2at = np.expand_dims(el2at,axis=0)
        if el2pt.ndim ==1:
            el2pt = np.expand_dims(el2pt,axis=0)
        if el2ne.ndim ==1:
            el2ne = np.expand_dims(el2ne,axis=0)


        self.el2pt = el2pt
        self.el2at = el2at
        self.fa2pt = fa2pt
        self.fa2ma = fa2ma
        self.pt2xy = pt2xy
        self.el2ne = el2ne
        self.nelem = np.shape(el2pt)[0]
        self.npts  = np.shape(pt2xy)[0]
        self.nface = np.shape(fa2pt)[0]
        self.attr  = np.unique(el2at, axis=0)
        self.nattr = len(np.unique(el2at, axis=0))

###----------------------------------------------------------------------
    def complete(self):
        
        # compute and store det(Jac) and inv(Jac.T)
        Jac_matrix = np.zeros([2,2,self.nelem])
        detJ = np.zeros(self.nelem)
        
        for iel in range(self.nelem):
            vertID = self.el2pt[iel,:];
            x = self.pt2xy[vertID,0]
            y = self.pt2xy[vertID,1]

            Jac_matrix[:,:,iel] = np.array([ [ x[1]-x[0], x[2]-x[0] ], [ y[1]-y[0], y[2]-y[0] ] ])
            detJ[iel] = np.linalg.det(Jac_matrix[:,:,iel])
            
            # we want to store inv(J^T)
            Jac_matrix[:,:,iel] = np.transpose( np.linalg.inv(Jac_matrix[:,:,iel]) )
            
        self.detJ = detJ
        self.iT_Jac_matrix = Jac_matrix
        
        # compute fa2el array
        self.fa2el = -np.ones((self.nface,2),dtype=int) # -1 means no triangle
        for ifa in range(self.nface):
            # get vertex ID for current edge
            vertID = self.fa2pt[ifa,:]
            # find row and col indices in el2pt for each vertex ID
            # we only need the row = list of triangle IDs where vert is found
            row0,col0 = np.where(self.el2pt == vertID[0] )
            row1,col1 = np.where(self.el2pt == vertID[1] )
            # find common triangles
            common_els = np.intersect1d(row0,row1)
            # update
            self.fa2el[ifa,0:len(common_els)] = common_els
            
        # compute el2fa array
        self.el2fa = np.zeros((self.nelem,3),dtype=int)
        for iel in range(self.nelem):
            # find edges that touch a given elem
            faces, col = np.where(self.fa2el == iel )
            # self.el2fa[iel,0:len(row)] = faces (inconsistency between el2pt and fa2pt: not same ordering)
            # at this stage, row contains the faces that are connected to element iel
            # vertex list
            vertID = self.el2pt[iel,:]
            # easier for looping
            vertID = np.append(vertID, vertID[0]) 
            # loop over the edges stored in the element
            for ied in range(3):
                # pick an edge
                edg_vertID = vertID[ied:ied+2]
                found_face = False
                for iface,face in enumerate(faces):
                    fapt = self.fa2pt[face]
                    if np.all( np.sort(edg_vertID) == np.sort(fapt) ):
                        self.el2fa[iel,ied] = face
                        found_face = True
                if not found_face:
                    print('Candidate faces ',faces,', Triangle:',iel,', edge loop ',iel)
                    print('Vertices from el2pt:',self.el2pt[iel,:])
                    print('Vertices from fa2pt:',self.fa2pt[faces])
                    raise Exception('face not found')

        # compute edge normals and edge lengths
        ## self.normals = np.zeros((self.nelem, 3, 3)) 
        self.normals = np.zeros((self.nelem, 3, 2)) # 2d normal for fatser dot prods
        self.edge_len = np.zeros((self.nelem, 3))
        for iel in range(self.nelem):
            # edges = self.el2fa[iel,:]
            vertID = self.el2pt[iel,:]
            coords = self.pt2xy[vertID,:]
            # easier for looping
            vertID = np.append(vertID, vertID[0]) 
            for iedg in range(3):
                # edgvert = self.fa2pt[edges[iedg],:] 
                edgvert = vertID[iedg:iedg+2]
                self.normals[iel,iedg,:] = self.compute_normal(edgvert, vertID, coords)
                self.edge_len[iel,iedg] = np.linalg.norm(self.pt2xy[edgvert[0],:] - self.pt2xy[edgvert[1],:] )
    
        print('Mesh completed!')

###----------------------------------------------------------------------
    # plot it
    def plot_mesh(self, axis_range=[]):
        mesh_points = np.array(self.pt2xy)
        mesh_tris = np.array(self.el2pt)

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
        if len(axis_range) == 0:
            xmin, ymin =np.min(self.pt2xy,axis=0)
            xmax, ymax =np.max(self.pt2xy,axis=0)
            dx = xmax - xmin
            dy = ymax - ymin
            xmin -= 0.05*dx
            ymin -= 0.05*dy
            xmax += 0.05*dx
            ymax += 0.05*dy
            axis_range = [xmin, xmax, ymin, ymax]
        plt.axis(axis_range)

        for iel in range(self.nelem):
            vertID = self.el2pt[iel,:]
            x = np.sum(self.pt2xy[vertID,0])/3
            y = np.sum(self.pt2xy[vertID,1])/3
            attr = self.el2at[iel]
            text_ = str(attr) + ' (' + str(iel) + ')'
            plt.text(x,y,text_,color='red',fontsize=6)
        for ied in range(self.nface):
            vertID = self.fa2pt[ied,:]
            x = np.sum(self.pt2xy[vertID,0])/2
            y = np.sum(self.pt2xy[vertID,1])/2
            attr = self.fa2ma[ied]
            if attr != 0: # marker 0 is reserved for interior edges
                plt.text(x,y,str(attr),color='black',backgroundcolor='yellow',fontsize=4)
        plt.show()

###----------------------------------------------------------------------
    def compute_normal(self, edgvertID, trivertID, coord):
        # A, B: vertices of given edge
        indexA = np.where( trivertID == edgvertID[0])[0][0]
        indexB = np.where( trivertID == edgvertID[1])[0][0]
        # 3rd vertex, not on edge
        other_vert = np.setdiff1d(np.union1d(trivertID, edgvertID),\
                                  np.intersect1d(trivertID, edgvertID))[0]
        indexC = np.where(trivertID == other_vert)[0][0]
        # assign coordinates. A,B = edge nodes
        A = coord[indexA,:]
        B = coord[indexB,:]
        C = coord[indexC,:]
        # create vectors
        AB = B-A
        AC = C-A
        BC = C-B
        # compute normal of AB
        n_AB = np.array([ AB[1], -AB[0] ])
        # determine the proper sign
        if np.dot(n_AB,BC)<0 and np.dot(n_AB,AC)<0:
            sign_AB =  1
        else:
            sign_AB = -1
        n_AB *= sign_AB
        # make a unit vector
        n_AB /= np.linalg.norm(n_AB)
        # add dummy z-component
        ## n_AB = np.append(n_AB, 0)
        return n_AB
            
###----------------------------------------------------------------------
    def check_bc(self, bc, verbose=False):
               
        markers_found = np.empty(0,dtype=int)

        key_set = bc.keys()
        
        for key in key_set:
            # check that we actually enter a non-empty array for markers
            if (bc.get(key)).get('markers').size>0:
                # check that marker array and value array have same length
                if (bc.get(key)).get('markers').size != (bc.get(key)).get('values').size:
                    raise ValueError('marker and value arrays of different length for bc type = ' + key)
                # check that the marker value in bc is actually found in the mesh data structure
                for mark in (bc.get(key)).get('markers'):
                    arr = np.where(self.fa2ma == mark)[0]
                    if arr.size==0:
                        raise ValueError('face marker '+str(mark)+' of type '+key+' was not found in mesh.fa2ma')
                # add markers to array of found markers
                markers_found = np.append(markers_found, bc.get(key).get('markers') )
        
        # check that markers found in bc dict match the ones from fa2ma
        markers_found = np.sort(np.unique(markers_found))
        fa2ma = np.sort(np.unique(self.fa2ma))
        ind = np.where( fa2ma == 0)[0]
        fa2ma = np.delete(fa2ma, ind)
        if len(fa2ma) != len(markers_found):
            raise Exception('bc dictionary:: len(fa2ma) != len(markers_found)')
        if np.all( markers_found == fa2ma ):
            if verbose:
                print('bc dictionary is consistent with fa2ma')
        else:
            raise Exception('bc dictionary is NOT consistent with fa2ma')

###----------------------------------------------------------------------           
    def sweep_order(self, omega, do_plot=False, verbose=False):
        
        # list of incoming edges
        inc_edg = np.array([],dtype=int)
        # get list of faces with any nonzero face markers
        bc_faces = np.where( self.fa2ma > 0 )[0]
        
        # establish list of incoming edges for a given omega direction
        for ibc, ifa in enumerate(bc_faces):
            # vertices for that edge
            edgvertID = self.fa2pt[ifa,:] 
            # find triangle. first entry because edge is on bd
            elem   = self.fa2el[ifa,0]
            vertID = self.el2pt[elem,:]
            coords = self.pt2xy[vertID,:]
            # compute normal
            normal = self.compute_normal(edgvertID, vertID, coords)
            # is edge incoming?
            if np.dot(normal, omega)<0:
                inc_edg = np.append(inc_edg, ifa)
                
        
        # make a copy because we are going to modify that structure 
        fa2el = np.copy(self.fa2el)
        
        # generate the sweep order for current omega
        sweep_order = np.array([],dtype=int)

        while (len(inc_edg)>0):
            # pick an edge from the list, call it edge-1
            edg1 = inc_edg[0]
            # find its triangle (the one with >=0 ID)
            elem = fa2el[edg1,:]
            iel = elem[elem>=0][0]
            # triangle vertices and coordinates
            vertID = self.el2pt[iel,:]
            coords = self.pt2xy[vertID,:]
            # find the other two edges, e2 and e3
            ind = np.where( self.el2fa[iel,:] != edg1)[0]
            edg2 = self.el2fa[iel,ind[0]]
            edg3 = self.el2fa[iel,ind[1]]
            # find the points that make up edges 2 and 3
            edg2_vertID = self.fa2pt[edg2,:]
            edg3_vertID = self.fa2pt[edg3,:]
            # compute normals
            n2 = self.compute_normal(edg2_vertID, vertID, coords)
            n3 = self.compute_normal(edg3_vertID, vertID, coords)
            # booleans
            is_e2_outgoing = np.dot(n2,omega) >=0 # whether e2 = outgoing edge
            is_e3_outgoing = np.dot(n3,omega) >=0 # whether e3 = outgoing edge
            res = np.where( inc_edg == edg2 )
            is_e2_incoming = len(res[0])>0 # whether e2 is ready to be an incoming edge
            res = np.where( inc_edg == edg3 )
            is_e3_incoming = len(res[0])>0 # whether e3 is ready to be an incoming edge
            
            if (   ( is_e2_outgoing and is_e3_outgoing )  # both e2 and e3 are outgoing edges
                or ( is_e2_outgoing and is_e3_incoming )  # e2=outgoing, e3=ready as an incoming
                or ( is_e3_outgoing and is_e2_incoming )  # e3=outgoing, e2=ready as an incoming
                ):
                # make e2 and e3 forget about current triangle
                ind = np.where( fa2el[edg2,:] == iel)[0]
                fa2el[edg2,ind] = -1
                ind = np.where( fa2el[edg3,:] == iel)[0]
                fa2el[edg3,ind] = -1

                # check if e2 and e3 are incoming for another triangle
                res = np.where( fa2el[edg2,:] >=0 )
                if len(res[0])>0:
                    inc_edg = np.append(inc_edg, edg2)
                else: # remove from stack
                    ind = np.where( inc_edg == edg2)[0]
                    inc_edg = np.delete(inc_edg, ind)
                res = np.where( fa2el[edg3,:] >=0 )
                if len(res[0])>0:
                    inc_edg = np.append(inc_edg, edg3)
                else: # remove from stack
                    ind = np.where( inc_edg == edg3)[0]
                    inc_edg = np.delete(inc_edg, ind)
                
                # remove edge-1
                inc_edg = np.delete(inc_edg,0)
                # add triangle to sweep order
                sweep_order = np.append(sweep_order, iel)
                if verbose:
                    print('done with triangle ',iel)
                
            else:
                # cannot yet solve transport eqn for this triangle,
                # thus place this edge at the end of the list
                inc_edg = np.append(inc_edg, edg1)
                inc_edg = np.delete(inc_edg,0)

        # plot sweep order 
        if do_plot:
            plt.figure(figsize=(9, 8), dpi= 160, facecolor='w', edgecolor='k')
            plt.triplot(self.pt2xy[:, 0], self.pt2xy[:, 1], self.el2pt)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.plot(self.pt2xy[:, 0], self.pt2xy[:, 1], 'ro',ms=3)
            xmin, ymin =np.min(self.pt2xy,axis=0)
            xmax, ymax =np.max(self.pt2xy,axis=0)
            dx = xmax - xmin
            dy = ymax - ymin
            xmin -= 0.05*dx
            ymin -= 0.05*dy
            xmax += 0.05*dx
            ymax += 0.05*dy
            axis_range = [xmin, xmax, ymin, ymax]
            plt.axis(axis_range)

            for iel in range(self.nelem):
                vert_list = self.el2pt[iel,:]
                x = np.sum(self.pt2xy[vert_list,0])/3
                y = np.sum(self.pt2xy[vert_list,1])/3
                ind = np.where(sweep_order==iel)[0]
                plt.text(x,y,str(ind),color='red',fontsize=4)
            plt.show()    
            
        return sweep_order
            
###----------------------------------------------------------------------
    def upwind_neighbor(self, sweep_task, list_omega, verbose=False):
        
        n_dir = list_omega.shape[1]
        upwind_neigh = -np.ones((n_dir, self.nelem, 3, 3),dtype=int)
        mat_ind = np.zeros(self.nelem,dtype=int)
        
        # loop over directions
        for dir_ in range(n_dir):
            if verbose:
                print('*** upwind neighbors dir {0}/{1}'.format(dir_,n_dir-1))
            # sweep order
            swp = sweep_task[dir_,:]
            # omega
            omega = list_omega[:,dir_]
            
            # loop over triangles in sweeping order
            for tri in swp:
                if dir_==0:
                    # material
                    local_mat = self.el2at[tri]
                    ind = np.where(self.attr == local_mat)[0][0]
                    mat_ind[tri] = ind
                # vertex list
                vertID = self.el2pt[tri,:]
                # easier for looping
                vertID = np.append(vertID, vertID[0]) 
                # determine if edges of triangle are incoming or outgoing edges
                for ied in range(3):
                    # pick an edge
                    edg_vertID = vertID[ied:ied+2]
                    # compute normal
                    normal = self.normals[tri,ied,:]
                    omega_dot_n = np.dot(omega, normal)
                    if not(omega_dot_n>0):
                        # edge is incoming -> find upwind triangle
                        # first find upwind triangles that are not on boundary
                        ind = np.where( self.el2ne[tri,:] != -1)[0]
                        # find which triangle shares current edge with tri
                        found_edge = False
                        for k in range(len(ind)):
                            neigh = self.el2ne[tri,ind[k]]
                            indexA = np.where( self.el2pt[neigh,:] == edg_vertID[0])[0]
                            indexB = np.where( self.el2pt[neigh,:] == edg_vertID[1])[0]
                            if len(indexA)>0 and len(indexB)>0:
                                found_edge = True
                                upwind_neigh[dir_,tri,ied,0] = indexA[0]
                                upwind_neigh[dir_,tri,ied,1] = indexB[0]
                                upwind_neigh[dir_,tri,ied,2] = neigh
                                break # once found, we know there was only one
                        if found_edge == False:
                            # the edge must be on the boundary then
                            # find row and col indices in fa2pt for each vertex
                            # we only need the row = list of face IDs where vert is found
                            row0,col0 = np.where( self.fa2pt == edg_vertID[0] )
                            row1,col1 = np.where( self.fa2pt == edg_vertID[1] )
                            # find common faces
                            common_fa = np.intersect1d(row0,row1)
                            marker = self.fa2ma[common_fa]
                            upwind_neigh[dir_,tri,ied,0:2] = -9
                            upwind_neigh[dir_,tri,ied,2] = marker
                            # print('triangle {} has marker {}'.format(tri,marker))

        # done with direction dir
        return upwind_neigh, mat_ind
        
###----------------------------------------------------------------------           
    def sweep_order_unc(self, src_pt_in=[], near_src_region=False, do_plot=False, verbose=False):
        
        if not near_src_region:
            # src_pt: we pick the lower left corner, unless pt_src is given
            if len(src_pt_in) == 0:
                src_pt = np.min(self.pt2xy,axis=0)
            else:
                src_pt = np.copy(np.asarray(src_pt_in))
            
            # find cell with source origin in it:
            iel_src = []
            for iel in range(self.nelem):
                # vertices
                vertID = self.el2pt[iel,:]
                for iv in range(3):
                    dist = np.linalg.norm( self.pt2xy[vertID[iv],:] - src_pt)
                    # found src_pt ?
                    if dist < 1e-8:
                        iel_src.append(iel)
            if len(iel_src) == 0:
                raise Exception('source cell not found', src_pt)
            print('Found the following source cells:', iel_src,'\n')
        
            # list of incoming edges
            inc_edg = np.array([],dtype=int)
            
            # get list of faces with any nonzero face markers
            bc_faces = np.where( self.fa2ma > 0 )[0]
            
            # establish list of incoming edges 
            # (boundary edges of elements containing the source pt)
            for ibc, ifa in enumerate(bc_faces):
                # find triangle. first entry because edge is on bd
                elem  = self.fa2el[ifa,0]
                # is edge incoming?
                if elem in iel_src:
                    inc_edg = np.append(inc_edg, ifa)
            if verbose:
                print("inc_edg associated with source cell(s) = ",inc_edg)
            
        else: # if near-source reagion:
            if len(src_pt_in) == 0:
                raise Exception('provide the location of the source point')
            else:
                src_pt = np.copy(np.asarray(src_pt_in))
            # list of incoming edges
            inc_edg = np.array([],dtype=int)
            # get list of faces with any nonzero face markers
            bc_faces = np.where( self.fa2ma == 99 )[0]
            inc_edg = np.copy(bc_faces)

        # make a copy because we are going to modify that structure 
        fa2el = np.copy(self.fa2el)
        
        # generate the sweep order for current omega
        sweep_order = np.array([],dtype=int)

        # count = 0 
        while (len(inc_edg)>0):
            # pick an edge from the list, call it edge-1
            edg1 = inc_edg[0]
            # find its triangle (the one with >=0 ID)
            elem = fa2el[edg1,:]
            iel = elem[elem>=0][0]
            # triangle vertices and coordinates
            vertID  = self.el2pt[iel,:]
            coords = self.pt2xy[vertID,:]
            # find the other two edges, e2 and e3
            ind = np.where( self.el2fa[iel,:] != edg1)[0]
            edg2 = self.el2fa[iel,ind[0]]
            edg3 = self.el2fa[iel,ind[1]]
            # find the points that make up edges 2 and 3
            edg2_vertID = self.fa2pt[edg2,:]
            edg3_vertID = self.fa2pt[edg3,:]
            # compute normals
            n2 = self.compute_normal(edg2_vertID, vertID, coords)
            n3 = self.compute_normal(edg3_vertID, vertID, coords)
            # compute omegas from source to mid-edge
            # edge2:
            ptA = self.pt2xy[edg2_vertID[0],:]
            ptB = self.pt2xy[edg2_vertID[1],:]
            ptM = 0.5*(ptA+ptB)
            omega2 = (ptM - src_pt) / np.linalg.norm(ptM - src_pt)
            # print(ptA,ptB)
            # print('edg2 mid pt=',edg2,edg2_vert,ptM,omega2)
            ptA = self.pt2xy[edg3_vertID[0],:]
            ptB = self.pt2xy[edg3_vertID[1],:]
            ptM = 0.5*(ptA+ptB)
            omega3 = (ptM - src_pt) / np.linalg.norm(ptM - src_pt)
            # print(ptA,ptB)
            # print('edg3 mid pt=',edg3,edg3_vert,ptM,omega3)
            # booleans
            is_e2_outgoing = np.dot(n2,omega2) >=0 # whether e2 = outgoing edge
            is_e3_outgoing = np.dot(n3,omega3) >=0 # whether e3 = outgoing edge
            res = np.where( inc_edg == edg2 )
            is_e2_incoming = len(res[0])>0 # whether e2 is ready to be an incoming edge
            res = np.where( inc_edg == edg3 )
            is_e3_incoming = len(res[0])>0 # whether e3 is ready to be an incoming edge
            
            # print('edg2=',edg2,is_e2_incoming,is_e2_outgoing,'\t edg3=',edg3,is_e3_incoming,is_e3_outgoing)
            # print(n2,omega2,np.dot(n2,omega2))
            # print(n3,omega3,np.dot(n3,omega3))
            if (   ( is_e2_outgoing and is_e3_outgoing )  # both e2 and e3 are outgoing edges
                or ( is_e2_outgoing and is_e3_incoming )  # e2=outgoing, e3=ready as an incoming
                or ( is_e3_outgoing and is_e2_incoming )  # e3=outgoing, e2=ready as an incoming
                ):
                # make e2 and e3 forget about current triangle
                ind = np.where( fa2el[edg2,:] == iel)[0]
                fa2el[edg2,ind] = -1
                ind = np.where( fa2el[edg3,:] == iel)[0]
                fa2el[edg3,ind] = -1

                # check if e2 and e3 are incoming for another triangle
                res = np.where( fa2el[edg2,:] >=0 )
                if len(res[0])>0:
                    inc_edg = np.append(inc_edg, edg2)
                else: # remove from stack
                    ind = np.where( inc_edg == edg2)[0]
                    inc_edg = np.delete(inc_edg, ind)
                res = np.where( fa2el[edg3,:] >=0 )
                if len(res[0])>0:
                    inc_edg = np.append(inc_edg, edg3)
                else: # remove from stack
                    ind = np.where( inc_edg == edg3)[0]
                    inc_edg = np.delete(inc_edg, ind)
                
                # remove edge-1
                inc_edg = np.delete(inc_edg,0)
                # add triangle to sweep order
                sweep_order = np.append(sweep_order, iel)
                if verbose:
                    print('done with triangle ',iel)
                
            else:
                # cannot yet solve transport eqn for this triangle,
                # thus place this edge at the end of the list
                inc_edg = np.append(inc_edg, edg1)
                inc_edg = np.delete(inc_edg,0)
                # print("**** incoming edges = ",inc_edg)
            # print("swp=",sweep_order)
            # count +=1
            # if count==7:
            #     raise Exception('a')

        # save
        self.src_pt = np.copy(src_pt)
        
        # plot sweep order 
        if do_plot:
            plt.figure(figsize=(9, 8), dpi= 160, facecolor='w', edgecolor='k')
            plt.triplot(self.pt2xy[:, 0], self.pt2xy[:, 1], self.el2pt)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.plot(self.pt2xy[:, 0], self.pt2xy[:, 1], 'ro',ms=3)
            xmin, ymin =np.min(self.pt2xy,axis=0)
            xmax, ymax =np.max(self.pt2xy,axis=0)
            dx = xmax - xmin
            dy = ymax - ymin
            xmin -= 0.05*dx
            ymin -= 0.05*dy
            xmax += 0.05*dx
            ymax += 0.05*dy
            axis_range = [xmin, xmax, ymin, ymax]
            plt.axis(axis_range)

            for iel in range(self.nelem):
                vert_list = self.el2pt[iel,:]
                x = np.sum(self.pt2xy[vert_list,0])/3
                y = np.sum(self.pt2xy[vert_list,1])/3
                ind = np.where(sweep_order==iel)[0]
                plt.text(x,y,str(ind),color='red',fontsize=4)
            plt.show()    
            
        return sweep_order
            
###----------------------------------------------------------------------
    def upwind_neighbor_unc(self, swp, near_src_region=False, verbose=False):
        
        upwind_neigh = -np.ones((self.nelem, 3, 3),dtype=int)
        mat_ind = np.zeros(self.nelem,dtype=int)

        # loop over triangles in sweeping order
        for tri in swp:
            # material
            local_mat = self.el2at[tri]
            ind = np.where(self.attr == local_mat)[0][0]
            mat_ind[tri] = ind
            # vertex list
            vertID = self.el2pt[tri,:]
            # easier for looping
            vertID = np.append(vertID, vertID[0]) 
            # determine if edges of triangle are incoming or outgoing edges
            for ied in range(3):
                # pick an edge
                edg_vertID = vertID[ied:ied+2]
                # compute normal
                normal = self.normals[tri,ied,:]
                # compute omegas from source to mid-edge
                ptA = self.pt2xy[edg_vertID[0],:]
                ptB = self.pt2xy[edg_vertID[1],:]
                ptM = 0.5*(ptA+ptB)
                omega = (ptM-self.src_pt) / np.linalg.norm(ptM-self.src_pt)
                omega_dot_n = np.dot(omega, normal)
                if not(omega_dot_n>0):
                    # edge is incoming -> find upwind triangle
                    # first find upwind triangles that are not on boundary
                    ind = np.where( self.el2ne[tri,:] != -1)[0]
                    # find which triangle shares current edge with tri
                    found_edge = False
                    for k in range(len(ind)):
                        neigh = self.el2ne[tri,ind[k]]
                        indexA = np.where( self.el2pt[neigh,:] == edg_vertID[0])[0]
                        indexB = np.where( self.el2pt[neigh,:] == edg_vertID[1])[0]
                        if len(indexA)>0 and len(indexB)>0:
                            upwind_neigh[tri,ied,0] = indexA[0]
                            upwind_neigh[tri,ied,1] = indexB[0]
                            upwind_neigh[tri,ied,2] = neigh
                            found_edge = True
                            break # once found, we know there was only one
                    if found_edge == False:
                        # the edge must be on the boundary then
                        # find row and col indices in fa2pt for each vertex
                        # we only need the row = list of face IDs where vert is found
                        row0,col0 = np.where( self.fa2pt == edg_vertID[0] )
                        row1,col1 = np.where( self.fa2pt == edg_vertID[1] )
                        # find common faces
                        common_fa = np.intersect1d(row0,row1)
                        marker = self.fa2ma[common_fa]
                        upwind_neigh[tri,ied,0:2] = -9
                        upwind_neigh[tri,ied,2] = marker
                        # print('triangle {} has marker {}'.format(tri,marker))
                        
        return upwind_neigh, mat_ind
        