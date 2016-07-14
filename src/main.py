from __future__ import print_function
# import utils
import unittest
# from datetime import date
import datetime

import caffe
# from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2 as v2
import traceback

import os
from prototype import *

USE_GPU = False


def main():
    global USE_GPU
    if os.environ.get('NUT_enable_nvidia_devices', '') == "true":
        USE_GPU = True

    # if USE_GPU:
    #     caffe.set_mode_gpu()
    # else:
    #     caffe.set_mode_cpu()

    createDir("snapshots/to_store_snapshots")
    createDir("prototxt/to_store_net_and_solvers")

    objectives = Objective(0.4, 500000)
    params = {
        "featuresPerLayer": Param("", slice(4, 64, 10), 64),
        "convLayersPerBlock": Param("", slice(1, 5, 1), 2),
        "blocks": Param("", slice(1, 5, 1), 3),
        "kernelSize": Param("", slice(1, 5, 1), 3),
        "kernelSizeLocal": Param("", slice(1, 5, 1), 1),
        "strideConv": Param("", slice(1, 1, 1), 1),
        "stridePool": Param("", slice(1, 5, 1), 3),
        "inputSize": Param("", slice(32, 32, 1), 32)
        }
    archDef = ArchDef(objectives, params)

    # default settings
    # settings = {
    #     "featuresPerLayer": 64,
    #     "convLayersPerBlock": 2,
    #     "blocks": 3,
    #     "kernelSize": 3,
    #     "kernelSizeLocal": 1,
    #     "strideConv": 1,
    #     "stridePool": 2,
    #     "inputSize": 32
    # }

    # fewer params settings: only 16
    # settings = {
    #     "featuresPerLayer": 16,
    #     "convLayersPerBlock": 2,
    #     "blocks": 3,
    #     "kernelSize": 3,
    #     "kernelSizeLocal": 1,
    #     "strideConv": 1,
    #     "stridePool": 2,
    #     "inputSize": 32
    #     }

    # # fewer params settings: only 8
    # settings = {
    #     "featuresPerLayer": 8,
    #     "convLayersPerBlock": 2,
    #     "blocks": 3,
    #     "kernelSize": 3,
    #     "kernelSizeLocal": 1,
    #     "strideConv": 1,
    #     "stridePool": 2,
    #     "inputSize": 32
    #     }

    # # more params settings: only 8
    # settings = {
    #     "featuresPerLayer": 128,
    #     "convLayersPerBlock": 2,
    #     "blocks": 3,
    #     "kernelSize": 3,
    #     "kernelSizeLocal": 1,
    #     "strideConv": 1,
    #     "stridePool": 2,
    #     "inputSize": 32
    #     }

    # fewer params settings: only 32
    # settings = {
    #     "featuresPerLayer": 32,
    #     "convLayersPerBlock": 2,
    #     "blocks": 3,
    #     "kernelSize": 3,
    #     "kernelSizeLocal": 1,
    #     "strideConv": 1,
    #     "stridePool": 2,
    #     "inputSize": 32
    #     }

    # # more conv per block
    # settings = {
    #     "featuresPerLayer": 64,
    #     "convLayersPerBlock": 3,
    #     "blocks": 3,
    #     "kernelSize": 3,
    #     "kernelSizeLocal": 1,
    #     "strideConv": 1,
    #     "stridePool": 2,
    #     "inputSize": 32
    # }

    # # more blocks
    # settings = {
    #     "featuresPerLayer": 64,
    #     "convLayersPerBlock": 2,
    #     "blocks": 4,
    #     "kernelSize": 3,
    #     "kernelSizeLocal": 1,
    #     "strideConv": 1,
    #     "stridePool": 2,
    #     "inputSize": 32
    # }

    # more convs, smaller kernel
    # settings = {
    #     "featuresPerLayer": 64,
    #     "convLayersPerBlock": 3,
    #     "blocks": 3,
    #     "kernelSize": 2,
    #     "kernelSizeLocal": 1,
    #     "strideConv": 1,
    #     "stridePool": 2,
    #     "inputSize": 32
    # }

    # # more convs, smaller kernel, less filters
    # settings = {
    #     "featuresPerLayer": 16,
    #     "convLayersPerBlock": 4,
    #     "blocks": 3,
    #     "kernelSize": 2,
    #     "kernelSizeLocal": 1,
    #     "strideConv": 1,
    #     "stridePool": 2,
    #     "inputSize": 32
    # }

    # less blocks
    settings = {
        "featuresPerLayer": 32,
        "convLayersPerBlock": 2,
        "blocks": 2,
        "kernelSize": 3,
        "kernelSizeLocal": 1,
        "strideConv": 1,
        "stridePool": 2,
        "inputSize": 32
    }

    d = datetime.datetime.now()
    trainArchitecture(d.strftime("%y-%m-%d_%Hh%Mm%Ss_") + "0", archDef, settings)


def trainArchitecture(ID, archDef, settings):
    steps = []

    stepID = ID + "_00"
    solverFileName, netFileName, weightsFilePath = pretrainingConvCifar10(stepID, archDef, settings, False)
    print("pretraining done: ", solverFileName, netFileName, weightsFilePath)
    print("pretraining done: caffe train --solver=" + solverFileName, " > " + stepID + "_logs.txt 2>&1")
    steps.append("time caffe train --solver=" + solverFileName + " > " + stepID + "_logs.txt 2>&1")

    stepID = ID + "_10"
    previousWeightsFilePath = weightsFilePath
    solverFileName, netFileName, weightsFilePath = pretrainClassificationFrozen(stepID, archDef, settings, False)
    print("pretraining done: ", solverFileName, netFileName, weightsFilePath)
    print("pretraining done: caffe train --solver=" + solverFileName, "--weights=" + previousWeightsFilePath, " > " + stepID + "_logs.txt 2>&1")
    steps.append("time caffe train --solver=" + solverFileName + " --weights=" + previousWeightsFilePath + " > " + stepID + "_logs.txt 2>&1")

    stepID = ID + "_11"
    previousWeightsFilePath = weightsFilePath
    solverFileName, netFileName, weightsFilePath = pretrainClassification(stepID, archDef, settings, False)
    print("pretraining done: ", solverFileName, netFileName, weightsFilePath)
    print("pretraining done: caffe train --solver=" + solverFileName, "--weights=" + previousWeightsFilePath, " > " + stepID + "_logs.txt 2>&1")
    steps.append("time caffe train --solver=" + solverFileName + " --weights=" + previousWeightsFilePath + " > " + stepID + "_logs.txt 2>&1")
    # weightsFilePath = "snapshots/16-06-04_04h42m15s_0_11_pretrainClassification_iter_20000.caffemodel"

    # if settings["blocks"] > 2:
    stepID = ID + "_20"
    previousWeightsFilePath = weightsFilePath
    solverFileName, netFileName, weightsFilePath = reconstructIncremental1(stepID, archDef, settings, False)
    print("pretraining done: ", solverFileName, netFileName, weightsFilePath)
    print("pretraining done: caffe train --solver=" + solverFileName, "--weights=" + previousWeightsFilePath, " > " + stepID + "_logs.txt 2>&1")
    steps.append("time caffe train --solver=" + solverFileName + " --weights=" + previousWeightsFilePath + " > " + stepID + "_logs.txt 2>&1")
    # weightsFilePath = "snapshots/16-06-04_04h42m15s_0_20_reconstructIncremental1_iter_2000.caffemodel"

    if settings["blocks"] > 2:
        stepID = ID + "_30"
        previousWeightsFilePath = weightsFilePath
        solverFileName, netFileName, weightsFilePath = reconstructIncremental2(stepID, archDef, settings, False)
        print("pretraining done: ", solverFileName, netFileName, weightsFilePath)
        print("pretraining done: caffe train --solver=" + solverFileName, "--weights=" + previousWeightsFilePath, " > " + stepID + "_logs.txt 2>&1")
        steps.append("time caffe train --solver=" + solverFileName + " --weights=" + previousWeightsFilePath + " > " + stepID + "_logs.txt 2>&1")

    stepID = ID + "_40"
    previousWeightsFilePath = weightsFilePath
    solverFileName, netFileName, weightsFilePath = reconstructFull(stepID, archDef, settings, False)
    print("pretraining done: ", solverFileName, netFileName, weightsFilePath)
    print("pretraining done: caffe train --solver=" + solverFileName, "--weights=" + previousWeightsFilePath, " > " + stepID + "_logs.txt 2>&1")
    steps.append("time caffe train --solver=" + solverFileName + " --weights=" + previousWeightsFilePath + " > " + stepID + "_logs.txt 2>&1")
    steps.append("time caffe test --weights=" + weightsFilePath + " --model=" + netFileName + " --gpu=0" + " > " + stepID + "_logs_tests.txt 2>&1")
    steps.append("time python2 VisualizeReconstructionOfLayer.py " + weightsFilePath + " " + netFileName)

    stepID = ID + "_50"
    previousWeightsFilePath = weightsFilePath
    solverFileName, netFileName, weightsFilePath = reconstructFullFC0(stepID, archDef, settings, False)
    print("pretraining done: ", solverFileName, netFileName, weightsFilePath)
    print("pretraining done: caffe train --solver=" + solverFileName, "--weights=" + previousWeightsFilePath, " > " + stepID + "_logs.txt 2>&1")
    steps.append("time caffe train --solver=" + solverFileName + " --weights=" + previousWeightsFilePath + " > " + stepID + "_logs.txt 2>&1")
    steps.append("time caffe test --weights=" + weightsFilePath + " --model=" + netFileName + " --gpu=0" + " > " + stepID + "_logs_tests.txt 2>&1")
    steps.append("time python2 VisualizeReconstructionOfLayer.py " + weightsFilePath + " " + netFileName)
    # weightsFilePath = "snapshots/16-06-07_15h12m56s_0_50_reconstructFullFC0_iter_30000.caffemodel"

    stepID = ID + "_60"
    previousWeightsFilePath = weightsFilePath
    solverFileName, netFileName, weightsFilePath = reconstructFullFC0unfrozen(stepID, archDef, settings, False)
    print("pretraining done: ", solverFileName, netFileName, weightsFilePath)
    print("pretraining done: caffe train --solver=" + solverFileName, "--weights=" + previousWeightsFilePath, " > " + stepID + "_logs.txt 2>&1")
    steps.append("time caffe train --solver=" + solverFileName + " --weights=" + previousWeightsFilePath + " > " + stepID + "_logs.txt 2>&1")
    steps.append("time caffe test --weights=" + weightsFilePath + " --model=" + netFileName + " --gpu=0" + " > " + stepID + "_logs_tests.txt 2>&1")
    steps.append("python2 VisualizeReconstructionOfLayer.py " + weightsFilePath + " " + netFileName)

    print("")
    print("Full command:")
    fullCommand = "nut --exec='cd /src; " + "; ".join(steps) + "'"
    print(fullCommand)
    open(ID + "_run.sh", "w+").write(fullCommand)


def get_function_name():
    return traceback.extract_stack(None, 2)[0][2]


def pretrainingConvCifar10(ID, archDef, settings, performTraining=True):
    phaseName = get_function_name()

    # create solver
    solver_param = v2.SolverParameter()
    solver_param.base_lr = 1E-4
    solver_param.display = 10
    solver_param.test_iter.append(5)
    solver_param.test_interval = 500
    solver_param.max_iter = 2000
    solver_param.lr_policy = "fixed"
    solver_param.momentum = 0.9
    solver_param.weight_decay = 0.004
    solver_param.snapshot = 500
    solver_param.snapshot_prefix = "snapshots/" + ID + "_" + phaseName
    if USE_GPU:
        solver_param.solver_mode = solver_param.GPU
    else:
        solver_param.solver_mode = solver_param.CPU

    # create network
    net_param = caffe.proto.caffe_pb2.NetParameter()

    (dataTrain, dataTest) = dataLayers(net_param, 128, dataset="cifar10")

    # create the conv pool blocks
    blocks = []
    for i in range(settings["blocks"]):
        block = archDef.createEncoderBlock(net_param, i, settings, outputMask=False)
        blocks.append(block)

    # create the middle layer
    middleKernelSize = settings["inputSize"] / (2**settings["blocks"])
    middleConv = plug(blocks[-1][-1], conv(net_param.layers.add(),
                                  name="middle_conv",
                                  ks=middleKernelSize,
                                  nout=50,
                                  stride=1
                                  ))
    relu(middleConv, net_param.layers.add())

    # create additional fully connected layer to classify
    top = plug(middleConv, fullyConnected(net_param.layers.add(), name="fc1", nout=512))
    top = trainPhase(dropout(top, net_param.layers.add(), ratio=0.5))
    top = relu(top, net_param.layers.add())

    top = plug(top, fullyConnected(net_param.layers.add(), name="fc2_10", nout=10))

    plug(dataTrain, plug(top, softmax(net_param.layers.add())))

    plug(dataTest, plug(top, testPhase(accuracy(net_param.layers.add(), 1))))
    plug(dataTest, plug(top, testPhase(accuracy(net_param.layers.add(), 5))))

    (solverFileName, netFileName) = saveToFiles(ID + "_" + phaseName, solver_param, net_param)

    # train
    if performTraining:
        iterations = solver_param.max_iter
        solver.step(iterations)
        [solver, net] = getSolverNet(solver_param, net_param)
        net.save(weightsFilePath)

    # save
    weightsFilePath = solver_param.snapshot_prefix + "_iter_" + str(solver_param.max_iter) + ".caffemodel"
    return solverFileName, netFileName, weightsFilePath


def pretrainClassificationFrozen(ID, archDef, settings, performTraining=True):
    phaseName = get_function_name()

    # create solver
    solver_param = v2.SolverParameter()
    solver_param.base_lr = 1E-4
    solver_param.display = 10
    solver_param.test_iter.append(5)
    solver_param.test_interval = 1000
    solver_param.max_iter = 3000
    solver_param.lr_policy = "fixed"
    solver_param.momentum = 0.9
    solver_param.weight_decay = 0.004
    solver_param.snapshot = 1000
    solver_param.snapshot_prefix = "snapshots/" + ID + "_" + phaseName
    if USE_GPU:
        solver_param.solver_mode = solver_param.GPU
    else:
        solver_param.solver_mode = solver_param.CPU

    # create network
    net_param = caffe.proto.caffe_pb2.NetParameter()

    (dataTrain, dataTest) = dataLayers(net_param, 128)

    # create the conv pool blocks
    blocks = []
    for i in range(settings["blocks"]):
        block = archDef.createEncoderBlock(net_param, i, settings, outputMask=False, freezeBlock=True)
        blocks.append(block)

    # create the middle layer
    middleKernelSize = settings["inputSize"] / (2**settings["blocks"])
    middleConv = plug(blocks[-1][-1], freeze(conv(net_param.layers.add(),
                                  name="middle_conv",
                                  ks=middleKernelSize,
                                  nout=50,
                                  stride=1
                                  )))
    relu(middleConv, net_param.layers.add())

    # create additional fully connected layer to classify
    top = plug(middleConv, freeze(fullyConnected(net_param.layers.add(), name="fc1", nout=512)))
    top = trainPhase(dropout(top, net_param.layers.add(), ratio=0.5))
    top = relu(top, net_param.layers.add())

    top = plug(top, fullyConnected(net_param.layers.add(), name="fc2", nout=100))

    plug(dataTrain, plug(top, softmax(net_param.layers.add())))

    plug(dataTest, plug(top, testPhase(accuracy(net_param.layers.add(), 1))))
    plug(dataTest, plug(top, testPhase(accuracy(net_param.layers.add(), 5))))

    (solverFileName, netFileName) = saveToFiles(ID + "_" + phaseName, solver_param, net_param)

    # train
    if performTraining:
        iterations = solver_param.max_iter
        solver.step(iterations)
        [solver, net] = getSolverNet(solver_param, net_param)
        net.save(weightsFilePath)

    # save
    weightsFilePath = solver_param.snapshot_prefix + "_iter_" + str(solver_param.max_iter) + ".caffemodel"
    return solverFileName, netFileName, weightsFilePath


def pretrainClassification(ID, archDef, settings, performTraining=True):
    phaseName = get_function_name()

    # create solver
    solver_param = v2.SolverParameter()
    solver_param.base_lr = 1E-4
    solver_param.display = 10
    solver_param.test_iter.append(5)
    solver_param.test_interval = 1000
    solver_param.max_iter = 20000
    solver_param.lr_policy = "fixed"
    solver_param.momentum = 0.9
    solver_param.weight_decay = 0.004
    solver_param.snapshot = 1000
    solver_param.snapshot_prefix = "snapshots/" + ID + "_" + phaseName
    if USE_GPU:
        solver_param.solver_mode = solver_param.GPU
    else:
        solver_param.solver_mode = solver_param.CPU

    # create network
    net_param = caffe.proto.caffe_pb2.NetParameter()

    (dataTrain, dataTest) = dataLayers(net_param, 128)

    # create the conv pool blocks
    blocks = []
    for i in range(settings["blocks"]):
        block = archDef.createEncoderBlock(net_param, i, settings, outputMask=False)
        blocks.append(block)

    # create the middle layer
    middleKernelSize = settings["inputSize"] / (2**settings["blocks"])
    middleConv = plug(blocks[-1][-1], conv(net_param.layers.add(),
                                  name="middle_conv",
                                  ks=middleKernelSize,
                                  nout=50,
                                  stride=1
                                  ))
    relu(middleConv, net_param.layers.add())

    # create additional fully connected layer to classify
    top = plug(middleConv, fullyConnected(net_param.layers.add(), name="fc1", nout=512))
    top = trainPhase(dropout(top, net_param.layers.add(), ratio=0.5))
    top = relu(top, net_param.layers.add())

    top = plug(top, fullyConnected(net_param.layers.add(), name="fc2", nout=100))

    plug(dataTrain, plug(top, softmax(net_param.layers.add())))

    plug(dataTest, plug(top, testPhase(accuracy(net_param.layers.add(), 1))))
    plug(dataTest, plug(top, testPhase(accuracy(net_param.layers.add(), 5))))

    (solverFileName, netFileName) = saveToFiles(ID + "_" + phaseName, solver_param, net_param)

    # train
    if performTraining:
        iterations = solver_param.max_iter
        solver.step(iterations)
        [solver, net] = getSolverNet(solver_param, net_param)
        net.save(weightsFilePath)

    # save
    weightsFilePath = solver_param.snapshot_prefix + "_iter_" + str(solver_param.max_iter) + ".caffemodel"
    return solverFileName, netFileName, weightsFilePath

    # solver_param = v2.SolverParameter()
    # solver_param.base_lr = 1E-4
    # solver_param.display = 1
    # solver_param.test_iter.append(5)
    # solver_param.test_interval = 100
    # solver_param.max_iter = 50
    # solver_param.lr_policy = "fixed"
    # solver_param.momentum = 0.5
    # solver_param.weight_decay = 0.004
    # solver_param.snapshot = 200
    # solver_param.snapshot_prefix = "snapshots/" + ID + "_classify"
    # if USE_GPU:
    #     solver_param.solver_mode = solver_param.GPU
    # else:
    #     solver_param.solver_mode = solver_param.CPU

    # # create network
    # net_param = caffe.proto.caffe_pb2.NetParameter()

    # (dataTrain, dataTest) = dataLayers(net_param, 750)

    # # create the conv pool blocks
    # blocks = []
    # for i in range(settings["blocks"]):
    #     block = archDef.createEncoderBlock(net_param, i, settings, outputMask=False)
    #     blocks.append(block)

    # # create the middle layer
    # middleKernelSize = settings["inputSize"] / (2**settings["blocks"])
    # middleConv = plug(blocks[-1][-1], conv(net_param.layers.add(),
    #                               name="middle_conv",
    #                               ks=middleKernelSize,
    #                               nout=50,
    #                               stride=1
    #                               ))
    # relu(middleConv, net_param.layers.add())

    # # create additional fully connected layer to classify
    # top = plug(middleConv, fullyConnected(net_param.layers.add(), name="fc1", nout=1024))
    # top = trainPhase(dropout(top, net_param.layers.add(), ratio=0.5))
    # top = relu(top, net_param.layers.add())

    # top = plug(top, fullyConnected(net_param.layers.add(), name="fc2", nout=100))

    # plug(dataTrain, plug(top, softmax(net_param.layers.add())))

    # plug(dataTest, plug(top, testPhase(accuracy(net_param.layers.add(), 1))))
    # plug(dataTest, plug(top, testPhase(accuracy(net_param.layers.add(), 5))))

    # (solverFileName, netFileName) = saveToFiles(ID + "_classify", solver_param, net_param)
    # [solver, net] = getSolverNet(solver_param, net_param)

    # # train
    # iterations = 1000
    # solver.step(iterations)

    # # save
    # weightsFilePath = "snapshots/" + ID + "_classify_" + str(iterations)
    # net.save(weightsFilePath)
    # return weightsFilePath


def reconstructIncremental1(ID, archDef, settings, performTraining=True):
    phaseName = get_function_name()

    # create solver
    solver_param = v2.SolverParameter()
    solver_param.base_lr = 1E-7
    solver_param.display = 10
    # solver_param.test_iter.append(5)
    # solver_param.test_interval = 1000
    solver_param.max_iter = 2000
    solver_param.lr_policy = "fixed"
    solver_param.momentum = 0.7
    solver_param.weight_decay = 0.004
    solver_param.snapshot = 1000
    solver_param.snapshot_prefix = "snapshots/" + ID + "_" + phaseName
    if USE_GPU:
        solver_param.solver_mode = solver_param.GPU
    else:
        solver_param.solver_mode = solver_param.CPU

    # create network
    net_param = caffe.proto.caffe_pb2.NetParameter()

    (dataTrain, dataTest) = dataLayers(net_param, 512, labels=False)

    # create the conv pool blocks
    blocks = []
    for i in range(settings["blocks"]):
        block = archDef.createEncoderBlock(net_param, i, settings, outputMask=(i >= settings["blocks"] - 1), freezeBlock=True)
        blocks.append(block)

    # create the middle layer
    middleKernelSize = settings["inputSize"] / (2**settings["blocks"])
    middleConv = plug(blocks[-1][-1], freeze(conv(net_param.layers.add(),
                                  name="middle_conv",
                                  ks=middleKernelSize,
                                  nout=50,
                                  stride=1
                                  )))
    # relu(middleConv, net_param.layers.add())
    plug(middleConv, deconv(net_param.layers.add(),
                                  name="middle_deconv",
                                  ks=middleKernelSize,
                                  nout=settings["featuresPerLayer"],
                                  stride=settings["strideConv"],
                                  pad=1
                                  ))

    unblocks = []
    # if settings["blocks"] >
    # for i in range(settings["blocks"]-1, 1, -1): # only one deconv block
    # only one deconv block
    i = settings["blocks"]-1
    unblock = archDef.createDecoderBlock(net_param, i, blocks[i], settings)
    unblocks.append(unblock)
    top = plug(unblocks[-1][-1], plug(blocks[-2][-1], euclideanLoss(net_param.layers.add())))

    (solverFileName, netFileName) = saveToFiles(ID + "_" + phaseName, solver_param, net_param)

    # train
    if performTraining:
        iterations = solver_param.max_iter
        solver.step(iterations)
        [solver, net] = getSolverNet(solver_param, net_param)
        net.save(weightsFilePath)

    # save
    weightsFilePath = solver_param.snapshot_prefix + "_iter_" + str(solver_param.max_iter) + ".caffemodel"
    return solverFileName, netFileName, weightsFilePath


def reconstructIncremental2(ID, archDef, settings, performTraining=True):
    phaseName = get_function_name()

    # create solver
    solver_param = v2.SolverParameter()
    solver_param.base_lr = 1E-7
    solver_param.display = 10
    # solver_param.test_iter.append(5)
    # solver_param.test_interval = 1000
    solver_param.max_iter = 2000
    solver_param.lr_policy = "fixed"
    solver_param.momentum = 0.7
    solver_param.weight_decay = 0.004
    solver_param.snapshot = 1000
    solver_param.snapshot_prefix = "snapshots/" + ID + "_" + phaseName
    if USE_GPU:
        solver_param.solver_mode = solver_param.GPU
    else:
        solver_param.solver_mode = solver_param.CPU

    # create network
    net_param = caffe.proto.caffe_pb2.NetParameter()

    (dataTrain, dataTest) = dataLayers(net_param, 512, labels=False)

    # create the conv pool blocks
    blocks = []
    for i in range(settings["blocks"]):
        block = archDef.createEncoderBlock(net_param, i, settings, outputMask=(i >= settings["blocks"] - 2), freezeBlock=True)
        blocks.append(block)

    # create the middle layer
    middleKernelSize = settings["inputSize"] / (2**settings["blocks"])
    middleConv = plug(blocks[-1][-1], freeze(conv(net_param.layers.add(),
                                  name="middle_conv",
                                  ks=middleKernelSize,
                                  nout=50,
                                  stride=1
                                  )))
    # relu(middleConv, net_param.layers.add())
    plug(middleConv, deconv(net_param.layers.add(),
                                  name="middle_deconv",
                                  ks=middleKernelSize,
                                  nout=settings["featuresPerLayer"],
                                  stride=settings["strideConv"],
                                  pad=1
                                  ))

    unblocks = []
    for i in range(settings["blocks"]-1, 0, -1): # only two deconv block
        unblock = archDef.createDecoderBlock(net_param, i, blocks[i], settings)
        unblocks.append(unblock)
    top = plug(unblocks[-1][-1], plug(blocks[-3][-1], euclideanLoss(net_param.layers.add())))

    (solverFileName, netFileName) = saveToFiles(ID + "_" + phaseName, solver_param, net_param)

    # train
    if performTraining:
        iterations = solver_param.max_iter
        solver.step(iterations)
        [solver, net] = getSolverNet(solver_param, net_param)
        net.save(weightsFilePath)

    # save
    weightsFilePath = solver_param.snapshot_prefix + "_iter_" + str(solver_param.max_iter) + ".caffemodel"
    return solverFileName, netFileName, weightsFilePath


def reconstructFull(ID, archDef, settings, performTraining=True):
    phaseName = get_function_name()

    # create solver
    solver_param = v2.SolverParameter()
    solver_param.base_lr = 1E-7
    solver_param.display = 10
    # solver_param.test_iter.append(5)
    # solver_param.test_interval = 1000
    solver_param.max_iter = 30000
    solver_param.lr_policy = "fixed"
    solver_param.momentum = 0.7
    solver_param.weight_decay = 0.004
    solver_param.snapshot = 1000
    solver_param.snapshot_prefix = "snapshots/" + ID + "_" + phaseName
    if USE_GPU:
        solver_param.solver_mode = solver_param.GPU
    else:
        solver_param.solver_mode = solver_param.CPU

    # create network
    net_param = caffe.proto.caffe_pb2.NetParameter()

    (dataTrain, dataTest) = dataLayers(net_param, 750, labels=False)

    # create the conv pool blocks
    blocks = []
    for i in range(settings["blocks"]):
        block = archDef.createEncoderBlock(net_param, i, settings, outputMask=True, freezeBlock=True)
        blocks.append(block)

    # create the middle layer
    middleKernelSize = settings["inputSize"] / (2**settings["blocks"])
    middleConv = plug(blocks[-1][-1], freeze(conv(net_param.layers.add(),
                                  name="middle_conv",
                                  ks=middleKernelSize,
                                  nout=50,
                                  stride=1
                                  )))
    # relu(middleConv, net_param.layers.add())
    plug(middleConv, deconv(net_param.layers.add(),
                                  name="middle_deconv",
                                  ks=middleKernelSize,
                                  nout=settings["featuresPerLayer"],
                                  stride=settings["strideConv"],
                                  pad=1
                                  ))

    unblocks = []
    for i in range(settings["blocks"]-1, -1, -1):
        unblock = archDef.createDecoderBlock(net_param, i, blocks[i], settings)
        unblocks.append(unblock)
    # unblock = archDef.createDecoderBlock(net_param, 0, block, settings)
    top = plug(unblocks[-1][-1], locallyConnected(net_param.layers.add(),
                                  name="reconstruct1",
                                  ks=settings["kernelSizeLocal"],
                                  nout=3,
                                  stride=1
                                  ))
    top = plug(top, locallyConnected(net_param.layers.add(),
                                  name="reconstruct2",
                                  ks=settings["kernelSizeLocal"],
                                  nout=3,
                                  stride=1
                                  ))
    top = plug(top, plug(dataTrain, euclideanLoss(net_param.layers.add())))

    (solverFileName, netFileName) = saveToFiles(ID + "_" + phaseName, solver_param, net_param)

    # train
    if performTraining:
        iterations = solver_param.max_iter
        solver.step(iterations)
        [solver, net] = getSolverNet(solver_param, net_param)
        net.save(weightsFilePath)

    # save
    weightsFilePath = solver_param.snapshot_prefix + "_iter_" + str(solver_param.max_iter) + ".caffemodel"
    return solverFileName, netFileName, weightsFilePath


def reconstructFullFC0(ID, archDef, settings, performTraining=True):
    phaseName = get_function_name()

    # create solver
    solver_param = v2.SolverParameter()
    solver_param.base_lr = 1E-7
    solver_param.display = 10
    # solver_param.test_iter.append(5)
    # solver_param.test_interval = 1000
    solver_param.max_iter = 30000
    solver_param.lr_policy = "fixed"
    solver_param.momentum = 0.7
    solver_param.weight_decay = 0.004
    solver_param.snapshot = 1000
    solver_param.snapshot_prefix = "snapshots/" + ID + "_" + phaseName
    if USE_GPU:
        solver_param.solver_mode = solver_param.GPU
    else:
        solver_param.solver_mode = solver_param.CPU

    # create network
    net_param = caffe.proto.caffe_pb2.NetParameter()

    (dataTrain, dataTest) = dataLayers(net_param, 750, labels=False)

    # create the conv pool blocks
    blocks = []
    for i in range(settings["blocks"]):
        block = archDef.createEncoderBlock(net_param, i, settings, outputMask=True, freezeBlock=True)
        blocks.append(block)

    # create the middle layer
    middleKernelSize = settings["inputSize"] / (2**settings["blocks"])
    middleConv = plug(blocks[-1][-1], freeze(conv(net_param.layers.add(),
                                  name="middle_conv_zeros",
                                  ks=middleKernelSize,
                                  nout=50,
                                  stride=1
                                  )))
    # make sure that no information can flow through
    middleConv.convolution_param.weight_filler.type = "constant"
    middleConv.convolution_param.weight_filler.value = 0
    middleConv.convolution_param.bias_filler.type = "constant"
    middleConv.convolution_param.bias_filler.value = 0

    # relu(middleConv, net_param.layers.add())
    plug(middleConv, deconv(net_param.layers.add(),
                                  name="middle_deconv",
                                  ks=middleKernelSize,
                                  nout=settings["featuresPerLayer"],
                                  stride=settings["strideConv"],
                                  pad=1
                                  ))

    unblocks = []
    for i in range(settings["blocks"]-1, -1, -1):
        unblock = archDef.createDecoderBlock(net_param, i, blocks[i], settings)
        unblocks.append(unblock)
    # unblock = archDef.createDecoderBlock(net_param, 0, block, settings)
    top = plug(unblocks[-1][-1], locallyConnected(net_param.layers.add(),
                                  name="reconstruct1",
                                  ks=settings["kernelSizeLocal"],
                                  nout=3,
                                  stride=1
                                  ))
    top = plug(top, locallyConnected(net_param.layers.add(),
                                  name="reconstruct2",
                                  ks=settings["kernelSizeLocal"],
                                  nout=3,
                                  stride=1
                                  ))
    top = plug(top, plug(dataTrain, euclideanLoss(net_param.layers.add())))

    (solverFileName, netFileName) = saveToFiles(ID + "_" + phaseName, solver_param, net_param)

    # train
    if performTraining:
        iterations = solver_param.max_iter
        solver.step(iterations)
        [solver, net] = getSolverNet(solver_param, net_param)
        net.save(weightsFilePath)

    # save
    weightsFilePath = solver_param.snapshot_prefix + "_iter_" + str(solver_param.max_iter) + ".caffemodel"
    return solverFileName, netFileName, weightsFilePath


def reconstructFullFC0unfrozen(ID, archDef, settings, performTraining=True):
    phaseName = get_function_name()

    # create solver
    solver_param = v2.SolverParameter()
    solver_param.base_lr = 1E-8
    solver_param.display = 10
    # solver_param.test_iter.append(5)
    # solver_param.test_interval = 1000
    solver_param.max_iter = 30000
    solver_param.lr_policy = "fixed"
    solver_param.momentum = 0.7
    solver_param.weight_decay = 0.004
    solver_param.snapshot = 1000
    solver_param.snapshot_prefix = "snapshots/" + ID + "_" + phaseName
    if USE_GPU:
        solver_param.solver_mode = solver_param.GPU
    else:
        solver_param.solver_mode = solver_param.CPU


    # create network
    net_param = caffe.proto.caffe_pb2.NetParameter()

    (dataTrain, dataTest) = dataLayers(net_param, 750, labels=False)

    # create the conv pool blocks
    blocks = []
    for i in range(settings["blocks"]):
        block = archDef.createEncoderBlock(net_param, i, settings, outputMask=True, freezeBlock=False)
        blocks.append(block)

    # create the middle layer
    middleKernelSize = settings["inputSize"] / (2**settings["blocks"])
    middleConv = plug(blocks[-1][-1], conv(net_param.layers.add(),
                                  name="middle_conv_zeros",
                                  ks=middleKernelSize,
                                  nout=50,
                                  stride=1
                                  ))
    # make sure that no information can flow through
    middleConv.convolution_param.weight_filler.type = "constant"
    middleConv.convolution_param.weight_filler.value = 0
    middleConv.convolution_param.bias_filler.type = "constant"
    middleConv.convolution_param.bias_filler.value = 0

    # relu(middleConv, net_param.layers.add())
    plug(middleConv, deconv(net_param.layers.add(),
                                  name="middle_deconv",
                                  ks=middleKernelSize,
                                  nout=settings["featuresPerLayer"],
                                  stride=settings["strideConv"],
                                  pad=1
                                  ))

    unblocks = []
    for i in range(settings["blocks"]-1, -1, -1):
        unblock = archDef.createDecoderBlock(net_param, i, blocks[i], settings)
        unblocks.append(unblock)
    # unblock = archDef.createDecoderBlock(net_param, 0, block, settings)
    top = plug(unblocks[-1][-1], locallyConnected(net_param.layers.add(),
                                  name="reconstruct1",
                                  ks=settings["kernelSizeLocal"],
                                  nout=3,
                                  stride=1
                                  ))
    top = plug(top, locallyConnected(net_param.layers.add(),
                                  name="reconstruct2",
                                  ks=settings["kernelSizeLocal"],
                                  nout=3,
                                  stride=1
                                  ))
    top = plug(top, plug(dataTrain, euclideanLoss(net_param.layers.add())))

    (solverFileName, netFileName) = saveToFiles(ID + "_" + phaseName, solver_param, net_param)

    # train
    if performTraining:
        iterations = solver_param.max_iter
        solver.step(iterations)
        [solver, net] = getSolverNet(solver_param, net_param)
        net.save(weightsFilePath)

    # save
    weightsFilePath = solver_param.snapshot_prefix + "_iter_" + str(solver_param.max_iter) + ".caffemodel"
    return solverFileName, netFileName, weightsFilePath

# def reconstructFullFC0unfrozen(ID, archDef, settings, performTraining=True):
#     phaseName = get_function_name()

#     # create solver
#     solver_param = v2.SolverParameter()
#     solver_param.base_lr = 1E-7
#     solver_param.display = 10
#     # solver_param.test_iter.append(5)
#     # solver_param.test_interval = 1000
#     solver_param.max_iter = 30000
#     solver_param.lr_policy = "fixed"
#     solver_param.momentum = 0.7
#     solver_param.weight_decay = 0.004
#     solver_param.snapshot = 1000
#     solver_param.snapshot_prefix = "snapshots/" + ID + "_" + phaseName
#     if USE_GPU:
#         solver_param.solver_mode = solver_param.GPU
#     else:
#         solver_param.solver_mode = solver_param.CPU

#     # create network
#     net_param = caffe.proto.caffe_pb2.NetParameter()

#     (dataTrain, dataTest) = dataLayers(net_param, 750, labels=False)

#     # create the conv pool blocks
#     blocks = []
#     for i in range(settings["blocks"]):
#         block = archDef.createEncoderBlock(net_param, i, settings, outputMask=True)
#         blocks.append(block)

#     # create the middle layer
#     middleKernelSize = settings["inputSize"] / (2**settings["blocks"])
#     middleConv = plug(blocks[-1][-1], conv(net_param.layers.add(),
#                                   name="middle_conv_zeros",
#                                   ks=middleKernelSize,
#                                   nout=50,
#                                   stride=1
#                                   ))
#     # make sure that no information can flow through
#     middleConv.convolution_param.weight_filler.type = "constant"
#     middleConv.convolution_param.weight_filler.value = 0
#     middleConv.convolution_param.bias_filler.type = "constant"
#     middleConv.convolution_param.bias_filler.value = 0

#     # relu(middleConv, net_param.layers.add())
#     plug(middleConv, deconv(net_param.layers.add(),
#                                   name="middle_deconv",
#                                   ks=middleKernelSize,
#                                   nout=settings["featuresPerLayer"],
#                                   stride=settings["strideConv"],
#                                   pad=1
#                                   ))

#     unblocks = []
#     for i in range(settings["blocks"]-1, -1, -1):
#         unblock = archDef.createDecoderBlock(net_param, i, blocks[i], settings)
#         unblocks.append(unblock)
#     # unblock = archDef.createDecoderBlock(net_param, 0, block, settings)
#     top = plug(unblocks[-1][-1], locallyConnected(net_param.layers.add(),
#                                   name="reconstruct1",
#                                   ks=settings["kernelSizeLocal"],
#                                   nout=3,
#                                   stride=1
#                                   ))
#     top = plug(top, locallyConnected(net_param.layers.add(),
#                                   name="reconstruct2",
#                                   ks=settings["kernelSizeLocal"],
#                                   nout=3,
#                                   stride=1
#                                   ))
#     top = plug(top, plug(dataTrain, euclideanLoss(net_param.layers.add())))

#     (solverFileName, netFileName) = saveToFiles(ID + "_" + phaseName, solver_param, net_param)

#     # train
#     if performTraining:
#         iterations = solver_param.max_iter
#         solver.step(iterations)
#         [solver, net] = getSolverNet(solver_param, net_param)
#         net.save(weightsFilePath)

#     # save
#     weightsFilePath = solver_param.snapshot_prefix + "_iter_" + str(solver_param.max_iter) + ".caffemodel"
#     return solverFileName, netFileName, weightsFilePath


def dataLayers(net_param, batch_size, dataset="cifar100", labels=True):
    if labels:
        tops = ["data", "label"]
    else:
        tops = ["data"]

    if dataset == "cifar100":
        dataTest = testPhase(dataLayer(net_param.layers.add(), tops=tops,
                    sourcePath="/dataset/cifar100_lmdb_lab/cifar100_test_lmdb",
                    meanFilePath="/dataset/cifar100_lmdb_lab/mean.binaryproto",
                    batch_size=batch_size))
        dataTrain = trainPhase(dataLayer(net_param.layers.add(), tops=tops,
                    sourcePath="/dataset/cifar100_lmdb_lab/cifar100_train_lmdb",
                    meanFilePath="/dataset/cifar100_lmdb_lab/mean.binaryproto",
                    batch_size=batch_size))
        return (dataTest, dataTrain)
    elif dataset == "cifar10":
        dataTest = testPhase(dataLayer(net_param.layers.add(), tops=tops,
                    sourcePath="/dataset/cifar10/cifar10_test_lmdb",
                    meanFilePath="/dataset/cifar10/mean.binaryproto",
                    batch_size=batch_size))
        dataTrain = trainPhase(dataLayer(net_param.layers.add(), tops=tops,
                    sourcePath="/dataset/cifar10/cifar10_train_lmdb",
                    meanFilePath="/dataset/cifar10/mean.binaryproto",
                    batch_size=batch_size))
        return (dataTest, dataTrain)


if __name__ == '__main__':
    main()

