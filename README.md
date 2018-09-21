# bsann
Bootstrapping Swarm Neural Network 

#   Bootstrapping Swarm Artificial Neural Network Program
#
#                        VERSION 1.0
#
# Written by Hiqmet Kamberaj.
# Copyright (C) 2018 Hiqmet Kamberaj.
# Check out h.kamberaj@gmail.com for more information.
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the Free Software Foundation; 
# GPL-3.0
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with this program; 
# if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, 
# Boston, MA 02111-1307 USA
#  Usage:
#  python driver_main.py inputfile          nframes Ndim_in Ndim_out nnets nepochs nruns   ntrains
#                        "data/data.txt"    30      5       1        5     50      10000   5
# where the input file is such:
#  x[t1,1],       x[t1,2], ....,       x[t1,Ndim_in],       y[t1,1] y[t1,2], ....,             y[t1,Ndim_out]
#  x[t2,1],       x[t2,2], ....,       x[t2,Ndim_in],       y[t2,1] y[t2,2], ....,             y[t2,Ndim_out]
#  ............................................................................
#  x[tNframes,1], x[tNframes,2], ...., x[tNframes,Ndim_in], y[tNframes,1] y[tNframes,2], ...., y[tNframes,Ndim_out]
#

"""
