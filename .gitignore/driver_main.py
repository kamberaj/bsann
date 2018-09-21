__doc__="""
# Id: driver_main.py,v 1.0 21-09-2018, IBU
#
#                This source code is part of
#
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
"""
from nnclass import *
from trainData import *
import numpy as np
from cmd import *

#----------------------------------------------------------------------------------------------------
def run_test(fname, nframes, Ndim_in, Ndim_out, nnets, nepochs, nruns, ntrains):
    T = nframes
    N = Ndim_in
    d = Ndim_out
    Alow  = -1.0
    Ahigh =  1.0

    [X, Y] = read_entries(fname, T, N, d)
	
    [Y, mn, mx] = scaleOutputs(Y,d,T,Alow,Ahigh)

### Prepare validation and training data
    t = ntrains
    marked = np.zeros( (T,1), int )
	
    Nnets = nnets
    epochs= nepochs
    Nruns = nruns
    Nets=[]
    #               0   -1   -2  -3   -4   -5  -6
    myLayer      = -3
    LayerNeurons = [N,  15,  30,  d]
    dimensions   = [N,   d,   d,  d] 
    
    train_data_size = t * Nnets
    xtrain = np.zeros((train_data_size, N), float)
    ytrain = np.zeros((train_data_size, d), float)
    valid_data_size = T - train_data_size
    xvalid = np.zeros((valid_data_size, N), float)
    yvalid = np.zeros((valid_data_size, d), float)
    ii = 0
    for inet in range(Nnets):
        x, y, marked = prepareData(T, t, N, d, X, Y, marked)
        Nets.append( NeuralNetwork(x, y, dimensions, LayerNeurons, 3) )
        for i in range(t):
            xtrain[ii] = x[i]
            ytrain[ii] = y[i]
            ii += 1
			
#-- validation data
    jj = 0
    for i in range(T):
        if marked[i] == 0:
		   xvalid[jj] = X[i]
		   yvalid[jj] = Y[i]
		   jj += 1
		   
############# Run Independently the Ensemble of Neural Networks
    global_err_best = 1000.0
    for irun in range(Nruns):
        inet = 0
        err=[]
        for net in Nets:
            net.NNrun(epochs)
            err.append( net.err_best )
            if global_err_best > net.err_best:
               global_err_best = net.err_best
               best_network = inet

            inet += 1
        print "Runs so far  ", irun+1, "  out of ", Nruns, global_err_best, err

## Swap params
        r = np.random.rand(1,1)
        if r > 0.5:
           inet0=0
        else:
           inet0=1
        for inet in xrange(inet0, Nnets-1, 2):
            for k in range(Nets[inet].nlayers):
                w1, b1 = Nets[inet].layer[k].getCurrentParams()
                w2, b2 = Nets[inet+1].layer[k].getCurrentParams()
                Nets[inet].layer[k].setCurrentParams(w2, b2)
                Nets[inet+1].layer[k].setCurrentParams(w1, b1)
                w1, b1 = Nets[inet].layer[k].getOptimalParams()
                w2, b2 = Nets[inet+1].layer[k].getOptimalParams()
                Nets[inet].layer[k].setOptimalParams(w2, b2)
                Nets[inet+1].layer[k].setOptimalParams(w1, b1)
                     
        for net in Nets:
            for k in range(net.nlayers):
                w, b = Nets[best_network].layer[k].getOptimalParams()
                net.layer[k].setGlobalBestParams(w,b)
        

    gNet = NeuralNetwork(xtrain, ytrain, dimensions, LayerNeurons, 3)
    for k in range(gNet.nlayers):
        w, b = Nets[best_network].layer[k].getGlobalBestParams()
        gNet.layer[k].setOptimalParams(w,b)
 
    train_output_mean   = gNet.predict(xtrain, 1)
    valid_output_mean   = gNet.predict(xvalid, 1)

    y                   = rescaleOutputs(ytrain, train_data_size, mx-mn, mn, Alow, Ahigh)
    y1                  = rescaleOutputs(yvalid, valid_data_size, mx-mn, mn, Alow, Ahigh)
    train_output_mean   = rescaleOutputs(train_output_mean, train_data_size, mx-mn, mn, Alow, Ahigh)
    valid_output_mean   = rescaleOutputs(valid_output_mean, valid_data_size, mx-mn, mn, Alow, Ahigh)

    fid = open("data/training.csv", 'w')
    for i in range(train_data_size):
        print >> fid, \
                 y[i][0], ",", \
                 train_output_mean[i][0]

    fid.close()

    fid = open("data/valid.csv", 'w')
    for i in range(valid_data_size):
        print >> fid, \
                 y1[i][0], ",", \
                 valid_output_mean[i][0]
    fid.close()
#---------------------------------------------------------------------------------------------------------  
if __name__ == '__main__':
    inputfile, nframes, Ndim_in, Ndim_out, nnets, nepochs, nruns, ntrains = callcmd()
    run_test(inputfile, nframes, Ndim_in, Ndim_out, nnets, nepochs, nruns, ntrains)