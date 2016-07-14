layers {
  top: "data"
  name: "data"
  type: DATA
  data_param {
    source: "/dataset/cifar100_lmdb_lab/cifar100_test_lmdb"
    batch_size: 750
    backend: LMDB
  }
  include {
    phase: TEST
  }
  transform_param {
    mean_file: "/dataset/cifar100_lmdb_lab/mean.binaryproto"
  }
}
layers {
  top: "data"
  name: "data"
  type: DATA
  data_param {
    source: "/dataset/cifar100_lmdb_lab/cifar100_train_lmdb"
    batch_size: 750
    backend: LMDB
  }
  include {
    phase: TRAIN
  }
  transform_param {
    mean_file: "/dataset/cifar100_lmdb_lab/mean.binaryproto"
  }
}
layers {
  bottom: "data"
  top: "0_0_conv"
  name: "0_0_conv"
  type: CONVOLUTION
  blobs_lr: 0
  blobs_lr: 0
  weight_decay: 0
  weight_decay: 0
  convolution_param {
    num_output: 32
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "0_0_conv"
  top: "0_0_conv"
  name: "0_0_conv_ReLU"
  type: RELU
}
layers {
  bottom: "0_0_conv"
  top: "0_1_conv"
  name: "0_1_conv"
  type: CONVOLUTION
  blobs_lr: 0
  blobs_lr: 0
  weight_decay: 0
  weight_decay: 0
  convolution_param {
    num_output: 32
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "0_1_conv"
  top: "0_1_conv"
  name: "0_1_conv_ReLU"
  type: RELU
}
layers {
  bottom: "0_1_conv"
  top: "0_pool"
  top: "0_pool_mask"
  name: "0_pool"
  type: POOLING
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layers {
  bottom: "0_pool"
  top: "1_0_conv"
  name: "1_0_conv"
  type: CONVOLUTION
  blobs_lr: 0
  blobs_lr: 0
  weight_decay: 0
  weight_decay: 0
  convolution_param {
    num_output: 32
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "1_0_conv"
  top: "1_0_conv"
  name: "1_0_conv_ReLU"
  type: RELU
}
layers {
  bottom: "1_0_conv"
  top: "1_1_conv"
  name: "1_1_conv"
  type: CONVOLUTION
  blobs_lr: 0
  blobs_lr: 0
  weight_decay: 0
  weight_decay: 0
  convolution_param {
    num_output: 32
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "1_1_conv"
  top: "1_1_conv"
  name: "1_1_conv_ReLU"
  type: RELU
}
layers {
  bottom: "1_1_conv"
  top: "1_pool"
  top: "1_pool_mask"
  name: "1_pool"
  type: POOLING
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layers {
  bottom: "1_pool"
  top: "2_0_conv"
  name: "2_0_conv"
  type: CONVOLUTION
  blobs_lr: 0
  blobs_lr: 0
  weight_decay: 0
  weight_decay: 0
  convolution_param {
    num_output: 32
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "2_0_conv"
  top: "2_0_conv"
  name: "2_0_conv_ReLU"
  type: RELU
}
layers {
  bottom: "2_0_conv"
  top: "2_1_conv"
  name: "2_1_conv"
  type: CONVOLUTION
  blobs_lr: 0
  blobs_lr: 0
  weight_decay: 0
  weight_decay: 0
  convolution_param {
    num_output: 32
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "2_1_conv"
  top: "2_1_conv"
  name: "2_1_conv_ReLU"
  type: RELU
}
layers {
  bottom: "2_1_conv"
  top: "2_pool"
  top: "2_pool_mask"
  name: "2_pool"
  type: POOLING
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layers {
  bottom: "2_pool"
  top: "middle_conv"
  name: "middle_conv"
  type: CONVOLUTION
  blobs_lr: 0
  blobs_lr: 0
  weight_decay: 0
  weight_decay: 0
  convolution_param {
    num_output: 50
    pad: 0
    kernel_size: 4
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "middle_conv"
  top: "middle_deconv"
  name: "middle_deconv"
  type: DECONVOLUTION
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 0
  weight_decay: 0
  convolution_param {
    num_output: 32
    pad: 1
    kernel_size: 4
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "middle_deconv"
  bottom: "2_pool_mask"
  top: "2_unpool"
  name: "2_unpool"
  type: UNPOOLING
  unpooling_param {
    unpool: MAX
    kernel_size: 2
    stride: 2
    unpool_size: 8
  }
}
layers {
  bottom: "2_unpool"
  top: "2_1_deconv"
  name: "2_1_deconv"
  type: DECONVOLUTION
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 0
  weight_decay: 0
  convolution_param {
    num_output: 32
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "2_1_deconv"
  top: "2_1_deconv"
  name: "2_1_deconv_ReLU"
  type: RELU
}
layers {
  bottom: "2_1_deconv"
  top: "2_0_deconv"
  name: "2_0_deconv"
  type: DECONVOLUTION
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 0
  weight_decay: 0
  convolution_param {
    num_output: 32
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "2_0_deconv"
  top: "2_0_deconv"
  name: "2_0_deconv_ReLU"
  type: RELU
}
layers {
  bottom: "2_0_deconv"
  bottom: "1_pool_mask"
  top: "1_unpool"
  name: "1_unpool"
  type: UNPOOLING
  unpooling_param {
    unpool: MAX
    kernel_size: 2
    stride: 2
    unpool_size: 16
  }
}
layers {
  bottom: "1_unpool"
  top: "1_1_deconv"
  name: "1_1_deconv"
  type: DECONVOLUTION
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 0
  weight_decay: 0
  convolution_param {
    num_output: 32
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "1_1_deconv"
  top: "1_1_deconv"
  name: "1_1_deconv_ReLU"
  type: RELU
}
layers {
  bottom: "1_1_deconv"
  top: "1_0_deconv"
  name: "1_0_deconv"
  type: DECONVOLUTION
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 0
  weight_decay: 0
  convolution_param {
    num_output: 32
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "1_0_deconv"
  top: "1_0_deconv"
  name: "1_0_deconv_ReLU"
  type: RELU
}
layers {
  bottom: "1_0_deconv"
  bottom: "0_pool_mask"
  top: "0_unpool"
  name: "0_unpool"
  type: UNPOOLING
  unpooling_param {
    unpool: MAX
    kernel_size: 2
    stride: 2
    unpool_size: 32
  }
}
layers {
  bottom: "0_unpool"
  top: "0_1_deconv"
  name: "0_1_deconv"
  type: DECONVOLUTION
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 0
  weight_decay: 0
  convolution_param {
    num_output: 32
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "0_1_deconv"
  top: "0_1_deconv"
  name: "0_1_deconv_ReLU"
  type: RELU
}
layers {
  bottom: "0_1_deconv"
  top: "0_0_deconv"
  name: "0_0_deconv"
  type: DECONVOLUTION
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 0
  weight_decay: 0
  convolution_param {
    num_output: 32
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "0_0_deconv"
  top: "0_0_deconv"
  name: "0_0_deconv_ReLU"
  type: RELU
}
layers {
  bottom: "0_0_deconv"
  top: "reconstruct1"
  name: "reconstruct1"
  type: LOCAL
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 0
  weight_decay: 0
  local_param {
    num_output: 3
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "reconstruct1"
  top: "reconstruct2"
  name: "reconstruct2"
  type: LOCAL
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 0
  weight_decay: 0
  local_param {
    num_output: 3
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  bottom: "data"
  bottom: "reconstruct2"
  top: "L2_Loss"
  name: "L2_Loss"
  type: EUCLIDEAN_LOSS
}
