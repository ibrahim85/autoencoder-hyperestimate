# docker run -i -t --rm --volume=$PWD:/theapp --workdir=/theapp AUTHOR/NAME_OF_DOCKER_REPOSITORY ipython -i VisualizeReconstructionOfLayer.py snapshots/reconstructing_full_extra_unfrozen_iter_19600.caffemodel reconstructFullUnFrozen.prototxt

from __future__ import division
import os
import numpy as np
import sys
import scipy

import caffe
# from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2 as v2


import pickle
from collections import OrderedDict


mainPath="./"
print("Usage: VisualizeReconstructionOfLayer.py <path_to_caffemodel> <path_to_prototxt_architecture_definition_file>")
print("")
pathToModel = sys.argv[1]
pathToPrototxt = sys.argv[2]
quantity = 10


# caffe.set_phase_test()
net = caffe.Net(pathToPrototxt, pathToModel)
# net = caffe.Net(pathToPrototxt, pathToModel, v2.TEST)
# net = caffe.Net(pathToPrototxt, pathToModel, caffe.TEST)
# net = caffe.Net(pathToPrototxt, pathToModel, 'test')
# net.set_mode_gpu()
fwd = net.forward()

meanface=np.load(os.path.join("/dataset", "cifar100_lmdb_lab", "mean.npy"))

im = net.blobs['reconstruct2'].data.copy()
im_correct = net.blobs['data'].data.copy() + meanface

im_plus_mean = (im.copy() + meanface).clip(0,255)
im /= np.max(np.abs(im))
im = (im+1)/2*255
idx = 0
X=np.zeros((3,32*2,0))
for idx in range(quantity):
     catWith = np.hstack((im_correct[idx], im_plus_mean[idx]))
     X=np.concatenate((X,catWith),2)


[basePath, modelName] = os.path.split(pathToModel)
[baseName, extension] = os.path.splitext(modelName)
[basePathProto, prototxtName] = os.path.split(pathToPrototxt)
[baseNameProto, extensionProto] = os.path.splitext(prototxtName)
# output = scipy.misc.imrotate(X.swapaxes(0, 2), -90)
output = np.rot90(X.swapaxes(0, 2))
output = np.rot90(output)
output = np.rot90(output)
outputName = "model--" + baseName + "--prototxt--" + baseNameProto +  '.png'
scipy.misc.imsave(outputName, output)
print("Result saved to " + outputName)
# sys.exit()


X=np.zeros((3,32*2,0))
for idx in [0, 1, 4, 5, 9]:
     catWith = np.hstack((im_correct[idx], im_plus_mean[idx]))
     X=np.concatenate((X,catWith),2)
output = np.rot90(X.swapaxes(0, 2))
output = np.rot90(output)
output = np.rot90(output)
outputName = "model--" + baseName + "--prototxt--" + baseNameProto +  '_crop.png'
scipy.misc.imsave(outputName, output)
print("Result saved to " + outputName)


os._exit(0)
