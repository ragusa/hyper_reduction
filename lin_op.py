#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 23:17:29 2020

@author: ragusa
"""
import numpy as np
import scipy.sparse

class LinOp:

    def __init__( self, mesh):
        self.mesh = mesh
        # elemental matrices
        m = np.ones((3,3)) + np.eye(3)
        m /= 24.0
        g = np.zeros((2,3))
        g[:,0] = np.array([-1,-1])
        g[:,1] = np.array([ 1, 0])
        g[:,2] = np.array([ 0, 1])
        self.m = m
        self.g = g


    def build_diffusion_op_per_attribute(self):

        self.M = []
        self.K = []
        self.q = []

        for imat in range(self.mesh.nattr):

            rows = []
            cols = []
            entries_m = []
            entries_k = []
            q = np.zeros(self.mesh.npts)

            mat = self.mesh.attr[imat]

            for iel in range(self.mesh.nelem):

                local_mat = self.mesh.el2at[iel]
                if local_mat == mat:
                    # list of vertices
                    v = self.mesh.el2pt[iel,:]
                    # connectivity
                    rows += [ v[0], v[0], v[0], v[1], v[1], v[1], v[2], v[2], v[2] ]
                    cols += [ v[0], v[1], v[2], v[0], v[1], v[2], v[0], v[1], v[2] ]
                    # mass matrix
                    """print('m=',self.m)
                    print('m=',self.m.flatten())
                    print('J=',self.mesh.detJ[iel])
                    print(self.m.flatten() * self.mesh.detJ[iel])"""
                    entries_m += list( self.m.flatten() * self.mesh.detJ[iel] )
                    # stiffness matrix
                    k = np.zeros((3,3))
                    iJacT = self.mesh.Jac_matrix[:,:,iel]
                    for i in range(3):
                        for j in range(3):
                            k[i,j] += np.dot( iJacT @ self.g[:,i] , iJacT @ self.g[:,j] )
                    entries_k += list( k.flatten() * self.mesh.detJ[iel]/2 )
                    # rhs vector (divide by 2-> area, divide by 3 -> 3 vertices)
                    q[v] += self.mesh.detJ[iel]/2/3

            # create sparse matrix per attribute
            M = scipy.sparse.csr_matrix( (entries_m, (rows, cols)), \
                                        shape=(self.mesh.npts, self.mesh.npts) )
            K = scipy.sparse.csr_matrix( (entries_k, (rows, cols)), \
                                        shape=(self.mesh.npts, self.mesh.npts) )
            self.M.append(M)
            self.K.append(K)
            self.q.append(q)


    def build_diffusion_op_per_bc(self, bc):
        # apply Robin bc conditions
        # phi/4 + D/2 dphi/dn = Jinc
        # {bd} on the LHS: int{bd} (phi/2-2.Jinc).bi
        # Keep on the LHS: int{bd} phi/2.bi
        # Put on the RHS: +int{bd} 2.Jinc.bi
        # 1d mass matrix, reference element
        m1d = np.array([ [2, 1],[1, 2] ]) / 6 # /6 = /3/2, the 2 is from Jacobian)
        # deal with Robin part of the matrix
        key = "Robin"
        if self.mesh.is_bc.get(key):
            # get list of face with any nonzero Robin face markers
            faces_rob = np.where( np.in1d( self.mesh.fa2ma , bc.get(key)['markers'] ) )
            # list of such vertices
            vert_list = self.mesh.fa2pt[faces_rob]
            # fill in matrix entries
            rows = []
            cols = []
            entries = []
            for i in range(np.shape(vert_list)[0]):
                # pick vertives for a given Robin edge
                v = vert_list[i,:]
                ptA = self.mesh.pt2xy[v[0],:]
                ptB = self.mesh.pt2xy[v[1],:]
                # compute the edge length
                AB=ptB-ptA;
                edge_len = np.linalg.norm(AB)
                # connectivity
                rows += [ v[0], v[0], v[1], v[1] ]
                cols += [ v[0], v[1], v[0], v[1] ]
                # the 1/2 comes from phi/2 in the formula
                # the row-sum of m1d already gives dx/2
                entries += list( m1d.flatten() * edge_len/2)
            # build CSR matrix
            self.Arob = scipy.sparse.csr_matrix( (entries, (rows, cols)), \
                                        shape=(self.mesh.npts, self.mesh.npts) )
        # deal with Robin rhs
        self.qrob = []
        if self.mesh.is_bc.get(key):
            for irob in range( bc.get(key)['markers'].shape[0] ):
                q = np.zeros(self.mesh.npts)
                # get list of face with a given Robin face marker
                faces_rob = np.where( self.mesh.fa2ma == bc.get(key)['markers'][irob] )
                # list of such vertices
                vert_list = self.mesh.fa2pt[faces_rob]
                for i in range(np.shape(vert_list)[0]):
                    # pick vertives for a given Robin edge
                    v = vert_list[i,:]
                    ptA = self.mesh.pt2xy[v[0],:]
                    ptB = self.mesh.pt2xy[v[1],:]
                    # compute the edge length
                    AB=ptB-ptA;
                    edge_len = np.linalg.norm(AB)
                    # rhs vector: 2*Jinc*edge_length/2;
                    q[v] += 2*edge_len/2
                self.qrob.append(q)

        # deal with Robin rhs
        self.qneu = []
        key = "Neumann"
        if self.mesh.is_bc.get(key):
            for irob in range( bc.get(key)['markers'].shape[0] ):
                q = np.zeros(self.mesh.npts)
                # get list of face with a given Robin face marker
                faces_rob = np.where( self.mesh.fa2ma == bc.get(key)['markers'][irob] )
                # list of such vertices
                vert_list = self.mesh.fa2pt[faces_rob]
                for i in range(np.shape(vert_list)[0]):
                    # pick vertives for a given Robin edge
                    v = vert_list[i,:]
                    ptA = self.mesh.pt2xy[v[0],:]
                    ptB = self.mesh.pt2xy[v[1],:]
                    # compute the edge length
                    AB=ptB-ptA;
                    edge_len = np.linalg.norm(AB)
                    # rhs vector: val*edge_length/2;
                    q[v] += edge_len/2
                self.qneu.append(q)


    def build_diffusion_system(self, qext, cdif, siga, bc, Jinc, Jneu):
        A = self.build_diffusion_matrix(cdif, siga)
        b = self.build_diffusion_rhs(qext)
        A,b = self.apply_bc(A, b, bc, Jinc, Jneu)
        return A, b


    def build_diffusion_matrix(self, cdif, siga):
        A = scipy.sparse.csr_matrix( (self.mesh.npts,self.mesh.npts) )
        for imat in range(self.mesh.nattr):
            A += cdif[imat] * self.K[imat] + siga[imat] * self.M[imat]
        return A


    def build_diffusion_rhs(self, qext):
        rhs = np.zeros(self.mesh.npts)
        for imat in range(self.mesh.nattr):
            rhs += qext[imat] * self.q[imat]
        return rhs


    def apply_bc(self, A, b, bc, Jinc, Jneu ):
        key = "Robin"
        if self.mesh.is_bc.get(key):
            len_ = bc.get(key)['markers'].shape[0]
            if  len_ != len(Jinc):
                raise ValueError('Jinc length = '+str(len(Jinc))+' but it should be = '+str(len_) )
            for irob in range(len_):
                b += self.qrob[irob]*Jinc[irob]
            A += self.Arob

        key = "Neumann"
        if self.mesh.is_bc.get(key):
            len_ = bc.get(key)['markers'].shape[0]
            if  len_ != len(Jneu):
                raise ValueError('Jneu length = '+str(len(Jneu))+' but it should be = '+str(len_) )
            for ineu in range(len_):
                b += self.qneu[ineu]*Jneu[ineu]

        key = "Dirichlet"
        if self.mesh.is_bc.get(key):
            raise ValueError('Apply bc of type'+key+' not implemented yet')

        return A, b

                
    def compute_reduced_operators(self, ur, bc):
        self.Mr = []
        self.Kr = []
        self.qr = []
        self.qrobr = []
        self.qneur = []

        for imat in range(self.mesh.nattr):
            Kr = ur.T @ self.K[imat] @ ur
            Mr = ur.T @ self.M[imat] @ ur
            qr = ur.T @ self.q[imat]
            self.Mr.append(Mr)
            self.Kr.append(Kr)
            self.qr.append(qr)

        self.Arobr = ur.T @ self.Arob @ ur
        for i in range(len(self.qrob)):
          self.qrobr.append( ur.T @ self.qrob[i] )
        for i in range(len(self.qneu)):
          self.qneur.append( ur.T @ self.qneu[i] )
    
        # below, compute reduced operators need for fast estimation of the ROM residual
        # matrix-matrix
        self.KK= []; self.MM= []; self.MK= []; self.Kq= []; self.Mq= []; self.qq= []
        for i in range(self.mesh.nattr):
            for j in range(i,self.mesh.nattr): # sum on j starts at i
                if i == j:
                    factor = 1
                else:
                    factor = 2 # because we only compute the triangular part (symmetry)
                self.KK.append( (ur.T @ self.K[i] @ self.K[j] @ ur) *factor )
                self.MM.append( (ur.T @ self.M[i] @ self.M[j] @ ur) *factor )
        for i in range(self.mesh.nattr):
            for j in range(self.mesh.nattr): # sum on j starts at 0
                self.MK.append( (ur.T @ self.M[i] @ self.K[j] @ ur) *2 ) # here, the factor 2 is b/c (a+b)^2 = a^2+2ab+b^2
        # matrix-source vector
        for i in range(self.mesh.nattr):
            for j in range(self.mesh.nattr): # sum on j starts at 0
                self.Kq.append( self.q[i] @ self.K[j] @ ur )
                self.Mq.append( self.q[i] @ self.M[j] @ ur )
        # source vector-source vector
        for i in range(self.mesh.nattr):
            for j in range(i,self.mesh.nattr): # sum on j starts at i
                if i == j:
                    factor = 1
                else:
                    factor = 2 # because we only compute the triangular part (symmetry)
                self.qq.append( (self.q[i] @ self.q[j]) *factor )

        # Robin BC:
        key = "Robin"
        if self.mesh.is_bc.get(key):
            self.KArob = []; self.MArob = []; self.Kqrob = []; self.Mqrob = [];
            self.Arobq = []; self.Arobqrob = []; self.qrobqrob = []; self.qqrob = [];
            n_rob = bc.get(key)['markers'].shape[0]
            # Arob to Arob
            self.ArobArob = (ur.T @ self.Arob @ self.Arob @ ur)
            # M/K to Arob
            for i in range(self.mesh.nattr):
                self.KArob.append( (ur.T @ self.K[i] @ self.Arob @ ur) *2 ) # here, the factor 2 is b/c (a+b)^2 = a^2+2ab+b^2
                self.MArob.append( (ur.T @ self.M[i] @ self.Arob @ ur) *2 ) # here, the factor 2 is b/c (a+b)^2 = a^2+2ab+b^2
            # M/K to qrob
            for i in range(n_rob):
                for j in range(self.mesh.nattr):
                    self.Kqrob.append( self.qrob[i] @ self.K[j] @ ur )
                    self.Mqrob.append( self.qrob[i] @ self.M[j] @ ur )
            # Arob to q
            for i in range(self.mesh.nattr):
                self.Arobq.append( self.q[i] @ self.Arob @ ur )
            # Arob to qrob
            for i in range(n_rob):
                self.Arobqrob.append( self.qrob[i] @ self.Arob @ ur )
            # q to qrob
            for i in range(self.mesh.nattr):
                for j in range(n_rob):
                    self.qqrob.append( (self.q[i] @ self.qrob[j]) *2 ) # here, the factor 2 is b/c (a+b)^2 = a^2+2ab+b^2
            # qrob to qrob
            for i in range(n_rob):
                for j in range(i,n_rob): # sum on j starts at i
                    if i == j:
                        factor = 1
                    else:
                        factor = 2 # because we only compute the triangular part (symmetry)
                    self.qrobqrob.append( (self.qrob[i] @ self.qrob[j]) *factor )

        # Neumann BC:
        key = "Neumann"
        if self.mesh.is_bc.get(key):
            self.Kqneu = []; self.Mqneu = [];
            self.Arobqneu = []; self.qneuqneu = []; self.qqneu = [];
            n_neu = bc.get(key)['markers'].shape[0]
            # M/K to qneu
            for i in range(n_neu):
                for j in range(self.mesh.nattr):
                    self.Kqneu.append( self.qneu[i] @ self.K[j] @ ur )
                    self.Mqneu.append( self.qneu[i] @ self.M[j] @ ur )
            # Arob to qneu
            for i in range(n_neu):
                self.Arobqneu.append( self.qneu[i] @ self.Arob @ ur )
            # q to qneu
            for i in range(self.mesh.nattr):
                for j in range(n_neu):
                    self.qqneu.append( (self.q[i]  @ self.qneu[j]) *2 ) # here, the factor 2 is b/c (a+b)^2 = a^2+2ab+b^2
            # qneu to qneu
            for i in range(n_neu):
                for j in range(i,n_neu): # sum on j starts at i
                    if i == j:
                        factor = 1
                    else:
                        factor = 2 # because we only compute the triangular part (symmetry)
                    self.qneuqneu.append( (self.qneu[i]  @ self.qneu[j]) *factor )
  

    def save_reduced_operators_and_POD_basis(self, ur, basename):
        # stifness, mass, and vol src arrays
        for imat in range(len(self.Kr)):
            filename = basename + str("_Kr_") +str(imat)
            np.save(filename,self.Kr[imat])
            filename = basename + str("_Mr_") +str(imat)
            np.save(filename,self.Mr[imat])
            filename = basename + str("_qr_") +str(imat)
            np.save(filename,self.qr[imat])
        # Rob matrix
        filename = basename + str("_Arobr")
        np.save(filename,self.Arobr)    
        # bc arrays
        for i in range(len(self.qrobr)):
            filename = basename + str("_qrobr_") +str(i)
            np.save(filename,self.qrobr[i])
        for i in range(len(self.qneur)):
            filename = basename + str("_qneur_") +str(i)
            np.save(filename,self.qneur[i])
        # POD basis
        filename = basename + str("_Ur")
        np.save(filename,ur)  


    def build_reduced_system(self, qext, cdif, siga, bc, Jinc, Jneu):
        r = self.Kr[1].shape[0]
        Ar = np.zeros((r,r))
        br = np.zeros(r)
    
        for imat in range(self.mesh.nattr):
            Ar += cdif[imat] * self.Kr[imat] + siga[imat] * self.Mr[imat]
            br += qext[imat] * self.qr[imat]
    
        key = "Robin"
        if self.mesh.is_bc.get(key):
            len_ = bc.get(key)['markers'].shape[0]
            for irob in range(len_):
                br += self.qrobr[irob]*Jinc[irob]
            Ar += self.Arobr
    
        key = "Neumann"
        if self.mesh.is_bc.get(key):
            len_ = bc.get(key)['markers'].shape[0]
            for ineu in range(len_):
                br += self.qneur[ineu]*Jneu[ineu]
    
        return Ar, br
    
    
    def residual_indicator_brute_force(self, A,b,ur,c):
        z = A @ ur @ c - b
        return np.linalg.norm(z)
    
    
    def residual_indicator(self, qext, cdif, siga, bc, Jinc, Jneu, c):
        # current reduced system size
        r = len(c)
        Arr = np.zeros((r,r))
        A1r = np.zeros((1,r))
        A11 = 0.
        # matrix-matrix
        counter = 0
        for i in range(self.mesh.nattr):
            cdif_i = cdif[i]; siga_i = siga[i]
            for j in range(i,self.mesh.nattr): # sum on j starts at i
                cdif_j = cdif[j]; siga_j = siga[j]
                Arr += cdif_i*cdif_j*self.KK[counter] + siga_i*siga_j*self.MM[counter]
                counter += 1
        counter = 0
        for i in range(self.mesh.nattr):
            siga_i = siga[i]
            for j in range(self.mesh.nattr): # sum on j starts at 0
                cdif_j = cdif[j]
                Arr += siga_i*cdif_j*self.MK[counter]
                counter += 1
        # matrix-source vector
        counter = 0
        for i in range(self.mesh.nattr):
            qext_i = qext[i]
            for j in range(self.mesh.nattr): # sum on j starts at 0
                cdif_j = cdif[j]; siga_j = siga[j]
                A1r += qext_i*( cdif_j*self.Kq[counter] + siga_j*self.Mq[counter] )
                counter += 1
        # source vector-source vector
        counter = 0
        for i in range(self.mesh.nattr):
            qext_i = qext[i]
            for j in range(i,self.mesh.nattr): # sum on j starts at i
                qext_j = qext[j]
                A11 += qext_i*qext_j*self.qq[counter]
                counter += 1
    
        # Robin BC:
        key = "Robin"
        if self.mesh.is_bc.get(key):
            n_rob = bc.get(key)['markers'].shape[0]
            # Arob to Arob
            Arr += self.ArobArob
            # M/K to Arob
            counter = 0
            for i in range(self.mesh.nattr):
                cdif_i = cdif[i]; siga_i = siga[i]
                Arr += cdif_i*self.KArob[counter] + siga_i*self.MArob[counter]
                counter += 1
            # M/K to qrob
            counter = 0
            for i in range(n_rob):
                Jinc_i = Jinc[i]
                for j in range(self.mesh.nattr):
                    cdif_j = cdif[j]; siga_j = siga[j]
                    A1r += Jinc_i *( cdif_j*self.Kqrob[counter] + siga_j*self.Mqrob[counter] )
                    counter += 1
            # Arob to q
            counter = 0
            for i in range(self.mesh.nattr):
                qext_i = qext[i]
                A1r += qext_i*self.Arobq[counter]
                counter += 1
            # Arob to qrob
            counter = 0
            for i in range(n_rob):
                Jinc_i = Jinc[i]
                A1r += Jinc_i*self.Arobqrob[counter]
                counter += 1
            # q to qrob
            counter = 0
            for i in range(self.mesh.nattr):
                qext_i = qext[i]
                for j in range(n_rob):
                    Jinc_j = Jinc[j]
                    A11 += qext_i*Jinc_j*self.qqrob[counter]
                    counter += 1
            # qrob to qrob
            counter = 0
            for i in range(n_rob):
                Jinc_i = Jinc[i]
                for j in range(i,n_rob): # sum on j starts at i
                    Jinc_j = Jinc[j]
                    A11 += Jinc_i*Jinc_j*self.qrobqrob[counter]
                    counter += 1
    
        # Neumann BC:
        key = "Neumann"
        if self.mesh.is_bc.get(key):
            n_neu = bc.get(key)['markers'].shape[0]
            # M/K to qneu
            counter = 0
            for i in range(n_neu):
                Jneu_i = Jneu[i]
                for j in range(self.mesh.nattr):
                    cdif_j = cdif[j]; siga_j = siga[j]
                    A1r += Jneu_i *( cdif_j*self.Kqneu[counter] + siga_j*self.Mqneu[counter] )
                    counter += 1
            # Arob to qneu
            counter = 0
            for i in range(n_neu):
                Jneu_i = Jneu[i]
                A1r += Jneu_i *self.Arobqneu[counter]
                counter += 1
            # q to qneu
            counter = 0
            for i in range(self.mesh.nattr):
                qext_i = qext[i]
                for j in range(n_neu):
                    Jneu_j = Jneu[j]
                    A11 += qext_i*Jneu_j*self.qqneu[counter]
                    counter += 1
            # qneu to qneu
            counter = 0
            for i in range(n_neu):
                Jneu_i = Jneu[i]
                for j in range(i,n_neu): # sum on j starts at i
                    Jneu_j = Jneu[j]
                    A11 += Jneu_i*Jneu_j*self.qneuqneu[counter]
                    counter += 1
    
        # compute residual:
        errorIndicator = c.T @ Arr @ c - 2* A1r @ c + A11
        if errorIndicator<0:
            if abs(errorIndicator)> 1E-10:
                raise ValueError('==> error indicator can only be <0 within roundoff :',errorIndicator)
            else:
              errorIndicator = abs(errorIndicator)
        return np.sqrt(errorIndicator)
