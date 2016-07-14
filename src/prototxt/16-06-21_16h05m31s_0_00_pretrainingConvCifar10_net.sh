layers {
  top: "data"
  top: "label"
  name: "data"
  type: DATA
  data_param {
    source: "/dataset/cifar10/cifar10_test_lmdb"
    batch_size: 128
    backend: LMDB
  }
  include {
    phase: TEST
  }
  transform_param {
    mean_file: "/dataset/cifar10/mean.binaryproto"
  }
}
layers {
  top: "data"
  top: "label"
  name: "data"
  type: DATA
  data_param {
    source: "/dataset/cifar10/cifar10_train_lmdb"
    batch_size: 128
    backend: LMDB
  }
  include {
    phase: TRAIN
  }
  transform_param {
    mean_file: "/dataset/cifar10/mean.binaryproto"
  }
}
layers {
  bottom: "data"
  top: "0_0_conv"
  name: "0_0_conv"
  type: CONVOLUTION
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
  bottom: "0_1_conv"
  top: "0_1_conv"
  name: "0_1_conv_ReLU"
  type: RELU
}
layers {
  bottom: "0_1_conv"
  top: "0_pool"
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
  bottom: "1_1_conv"
  top: "1_1_conv"
  name: "1_1_conv_ReLU"
  type: RELU
}
layers {
  bottom: "1_1_conv"
  top: "1_pool"
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
  bottom: "2_1_conv"
  top: "2_1_conv"
  name: "2_1_conv_ReLU"
  type: RELU
}
layers {
  bottom: "2_1_conv"
  top: "2_pool"
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
  blobs_lr: 1
  blobs_lr: 2
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
  top: "middle_conv"
  name: "middle_conv_ReLU"
  type: RELU
}
layers {
  bottom: "middle_conv"
  top: "fc1"
  name: "fc1"
  type: INNER_PRODUCT
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 0
  weight_decay: 0
  inner_product_param {
    num_output: 512
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
  bottom: "fc1"
  top: "fc1"
  name: "fc1_Dropout"
  type: DROPOUT
  dropout_param {
    dropout_ratio: 0.5
  }
  include {
    phase: TRAIN
  }
}
layers {
  bottom: "fc1"
  top: "fc1"
  name: "fc1_Dropout_ReLU"
  type: RELU
}
layers {
  bottom: "fc1"
  top: "fc2_10"
  name: "fc2_10"
  type: INNER_PRODUCT
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 0
  weight_decay: 0
  inner_product_param {
    num_output: 10
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
  bottom: "fc2_10"
  bottom: "label"
  top: "softmax"
  name: "softmax"
  type: SOFTMAX_LOSS
}
layers {
  bottom: "fc2_10"
  bottom: "label"
  top: "accuracy_top_1"
  name: "accuracy_top_1"
  type: ACCURACY
  accuracy_param {
    top_k: 1
  }
  include {
    phase: TEST
  }
}
layers {
  bottom: "fc2_10"
  bottom: "label"
  top: "accuracy_top_5"
  name: "accuracy_top_5"
  type: ACCURACY
  accuracy_param {
    top_k: 5
  }
  include {
    phase: TEST
  }
}
