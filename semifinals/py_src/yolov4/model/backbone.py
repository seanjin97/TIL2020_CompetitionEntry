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

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Layer
from .common import Mish, YOLOConv2D


class _ResBlock(Model):
    def __init__(self, filters_1: int, filters_2: int):
        super(_ResBlock, self).__init__()
        self.conv1 = YOLOConv2D(filters=filters_1, kernel_size=1)
        self.conv2 = YOLOConv2D(filters=filters_2, kernel_size=3)
        self.add = layers.Add()

    def call(self, x):
        ret = self.conv1(x)
        ret = self.conv2(ret)
        x = self.add([x, ret])
        return x


class ResBlock(Model):
    def __init__(self, filters_1: int, filters_2: int, iteration: int):
        super(ResBlock, self).__init__()
        self.iteration = iteration
        self.sequential = tf.keras.Sequential()
        for _ in range(self.iteration):
            self.sequential.add(
                _ResBlock(filters_1=filters_1, filters_2=filters_2)
            )

    def call(self, x):
        return self.sequential(x)


class CSPResNet(Model):
    """
    Cross Stage Partial connections(CSP)
    """

    def __init__(self, filters_1: int, filters_2: int, iteration: int):
        super(CSPResNet, self).__init__()
        self.pre_conv = YOLOConv2D(filters=filters_1, kernel_size=3, strides=2)

        # Do not change the order of declaration
        self.part2_conv = YOLOConv2D(filters=filters_2, kernel_size=1)

        self.part1_conv1 = YOLOConv2D(filters=filters_2, kernel_size=1)
        self.part1_res_block = ResBlock(
            filters_1=filters_1 // 2, filters_2=filters_2, iteration=iteration
        )
        self.part1_conv2 = YOLOConv2D(filters=filters_2, kernel_size=1)

        self.post_conv = YOLOConv2D(filters=filters_1, kernel_size=1)

    def call(self, x):
        x = self.pre_conv(x)

        part2 = self.part2_conv(x)

        part1 = self.part1_conv1(x)
        part1 = self.part1_res_block(part1)
        part1 = self.part1_conv2(part1)

        x = tf.concat([part1, part2], axis=-1)

        x = self.post_conv(x)
        return x


class SPP(Model):
    """
    Spatial Pyramid Pooling layer(SPP)
    """

    def __init__(self):
        super(SPP, self).__init__()
        self.pool1 = tf.keras.layers.MaxPooling2D(
            (13, 13), strides=1, padding="same"
        )
        self.pool2 = tf.keras.layers.MaxPooling2D(
            (9, 9), strides=1, padding="same"
        )
        self.pool3 = tf.keras.layers.MaxPooling2D(
            (5, 5), strides=1, padding="same"
        )

    def call(self, x):
        return tf.concat([self.pool1(x), self.pool2(x), self.pool3(x), x], -1)


class CSPDarknet53(Model):
    def __init__(self):
        super(CSPDarknet53, self).__init__()
        self.conv0 = YOLOConv2D(filters=32, kernel_size=3)

        self.res_block1 = CSPResNet(filters_1=64, filters_2=64, iteration=1)
        self.res_block2 = CSPResNet(filters_1=128, filters_2=64, iteration=2)
        self.res_block3 = CSPResNet(filters_1=256, filters_2=128, iteration=8)

        self.res_block4 = CSPResNet(filters_1=512, filters_2=256, iteration=8)

        self.res_block5 = CSPResNet(filters_1=1024, filters_2=512, iteration=4)

        self.conv72 = YOLOConv2D(filters=512, kernel_size=1, activation="leaky")
        self.conv73 = YOLOConv2D(
            filters=1024, kernel_size=3, activation="leaky"
        )
        self.conv74 = YOLOConv2D(filters=512, kernel_size=1, activation="leaky")

        self.spp = SPP()

        self.conv75 = YOLOConv2D(filters=512, kernel_size=1, activation="leaky")
        self.conv76 = YOLOConv2D(
            filters=1024, kernel_size=3, activation="leaky"
        )
        self.conv77 = YOLOConv2D(filters=512, kernel_size=1, activation="leaky")

    def call(self, x):
        x = self.conv0(x)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        route1 = x

        x = self.res_block4(x)

        route2 = x

        x = self.res_block5(x)
        x = self.conv72(x)
        x = self.conv73(x)
        x = self.conv74(x)

        x = self.spp(x)

        x = self.conv75(x)
        x = self.conv76(x)
        x = self.conv77(x)

        route3 = x

        return (route1, route2, route3)
