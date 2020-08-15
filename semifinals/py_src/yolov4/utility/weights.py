"""
MIT License

Copyright (c) 2020 Hyeonki Hong <hhk7734@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np


def load_yolov4(model, weights_file):
    with open(weights_file, "rb") as fd:
        major, minor, revision, seen, _ = _np_fromfile(
            fd, dtype=np.int32, count=5
        )

        csp_darknet53 = model.get_layer("csp_darknet53")

        if not csp_darknet53_set_weights(csp_darknet53, fd):
            return False

        for i in range(78, 110):
            layer_name = "yolo_conv2d_%d" % i

            yolo_conv2d = model.get_layer(layer_name)
            if not yolo_conv2d_set_weights(yolo_conv2d, fd):
                return False

        if len(fd.read()) != 0:
            raise ValueError("Model and weights file do not match.")

    return True


def _np_fromfile(fd, dtype, count):
    data = np.fromfile(fd, dtype=dtype, count=count)
    if len(data) != count:
        if len(data) == 0:
            return None
        else:
            raise ValueError("Model and weights file do not match.")
    return data


def yolo_conv2d_set_weights(yolo_conv2d, fd):
    if yolo_conv2d.strides[0] == 1:
        conv_index = 0
    else:
        conv_index = 1
    filters = yolo_conv2d.filters

    if yolo_conv2d.activation is not None:
        bn = yolo_conv2d.sequential.get_layer(index=conv_index + 1)

        # darknet weights: [beta, gamma, mean, variance]
        bn_weights = _np_fromfile(fd, dtype=np.float32, count=4 * filters)
        if bn_weights is None:
            return False
        # tf weights: [gamma, beta, mean, variance]
        bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

        bn.set_weights(bn_weights)
        conv_bias = None
    else:
        conv_bias = _np_fromfile(fd, dtype=np.float32, count=filters)
        if conv_bias is None:
            return False

    conv = yolo_conv2d.sequential.get_layer(index=conv_index)

    # darknet shape (out_dim, in_dim, height, width)
    conv_shape = (filters, yolo_conv2d.input_dim, *yolo_conv2d.kernel_size)
    conv_weights = _np_fromfile(
        fd, dtype=np.float32, count=np.product(conv_shape)
    )
    if conv_weights is None:
        return False
    # tf shape (height, width, in_dim, out_dim)
    conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

    if conv_bias is None:
        conv.set_weights([conv_weights])
    else:
        conv.set_weights([conv_weights, conv_bias])

    return True


def res_block_set_weights(model, fd):
    for i in range(model.iteration):
        _res_block = model.sequential.get_layer(index=i)
        if not yolo_conv2d_set_weights(_res_block.get_layer(index=0), fd):
            return False
        if not yolo_conv2d_set_weights(_res_block.get_layer(index=1), fd):
            return False

    return True


def csp_res_net_set_weights(model, fd):
    for i in range(6):
        if i == 3:
            if not res_block_set_weights(model.get_layer(index=3), fd):
                return False
        else:
            if not yolo_conv2d_set_weights(model.get_layer(index=i), fd):
                return False

    return True


def csp_darknet53_set_weights(csp_darknet53, fd):
    if not yolo_conv2d_set_weights(csp_darknet53.get_layer(index=0), fd):
        return False

    for i in range(1, 1 + 5):
        if not csp_res_net_set_weights(csp_darknet53.get_layer(index=i), fd):
            return False

    for i in range(6, 6 + 3):
        if not yolo_conv2d_set_weights(csp_darknet53.get_layer(index=i), fd):
            return False

    # index 9 is SPP

    for i in range(10, 10 + 3):
        if not yolo_conv2d_set_weights(csp_darknet53.get_layer(index=i), fd):
            return False

    return True
