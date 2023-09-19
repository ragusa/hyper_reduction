#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 23:12:09 2023

@author: jean.ragusa
"""

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from create_geometry import create_geometry

plt.close("all")

rf=None

geo_id_ = 1 
el2pt, el2at, fa2pt, fa2ma, pt2xy, el2ne = create_geometry(geo_id=geo_id_,max_vol=1e-3,\
                                                           refinement_funct=rf,do_plot=True)
if rf==None:
    basename = 'geo_id' + str(geo_id_) + '_elems' + str(len(el2pt[:,0]))
else:
    basename = 'geo_id' + str(geo_id_) + '_REF_elems' + str(len(el2pt[:,0]))
np.savetxt(basename+'_el2pt.txt',el2pt, fmt="%i")
np.savetxt(basename+'_el2at.txt',el2at, fmt="%i")
np.savetxt(basename+'_fa2pt.txt',fa2pt, fmt="%i")
np.savetxt(basename+'_fa2ma.txt',fa2ma, fmt="%i")
np.savetxt(basename+'_pt2xy.txt',pt2xy)
np.savetxt(basename+'_el2ne.txt',el2ne, fmt="%i")

