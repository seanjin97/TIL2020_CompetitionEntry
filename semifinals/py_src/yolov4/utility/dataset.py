"""
MIT License

Copyright (c) 2019 YangYun
Copyright (c) 2020 Việt Hùng
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

import os
import cv2
import numpy as np

from . import media
from . import train


class Dataset(object):
    def __init__(
        self,
        anchors: np.ndarray = None,
        dataset_path: str = None,
        dataset_type: str = "converted_coco",
        data_augmentation: bool = True,
        input_size: int = 416,
        num_classes: int = None,
        strides: np.ndarray = None,
        xyscales: np.ndarray = None,
    ):
        self.anchors_ratio = anchors / input_size
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.data_augmentation = data_augmentation
        self.grid_size = input_size // strides
        self.input_size = input_size
        self.num_classes = num_classes
        self.xysclaes = xyscales

        self.dataset = self.load_dataset()

        self.count = 0
        np.random.shuffle(self.dataset)

    def load_dataset(self):
        """
        @return
            yolo: [[image_path, [[x, y, w, h, class_id], ...]], ...]
            converted_coco: unit=> pixel
                [[image_path, [[x, y, w, h, class_id], ...]], ...]
        """
        _dataset = []

        with open(self.dataset_path, "r") as fd:
            txt = fd.readlines()
            if self.dataset_type == "converted_coco":
                for line in txt:
                    # line: "<image_path> xmin,ymin,xmax,ymax,class_id ..."
                    bboxes = line.strip().split()
                    image_path = bboxes[0]
                    xywhc_s = np.zeros((len(bboxes) - 1, 5))
                    for i, bbox in enumerate(bboxes[1:]):
                        # bbox = "xmin,ymin,xmax,ymax,class_id"
                        bbox = list(map(int, bbox.split(",")))
                        xywhc_s[i, :] = (
                            (bbox[0] + bbox[2]) / 2,
                            (bbox[1] + bbox[3]) / 2,
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1],
                            bbox[4],
                        )
                    _dataset.append([image_path, xywhc_s])

            elif self.dataset_type == "yolo":
                for line in txt:
                    # line: "<image_path>"
                    image_path = line.strip()
                    root, _ = os.path.splitext(image_path)
                    with open(root + ".txt") as fd2:
                        bboxes = fd2.readlines()
                        xywhc_s = np.zeros((len(bboxes), 5))
                        for i, bbox in enumerate(bboxes):
                            # bbox = class_id, x, y, w, h
                            bbox = bbox.strip()
                            bbox = list(map(float, bbox.split(",")))
                            xywhc_s[i, :] = (
                                *bbox[1:],
                                bbox[0],
                            )
                        _dataset.append([image_path, xywhc_s])
        return _dataset

    def bboxes_to_ground_truth(self, bboxes):
        """
        @param bboxes: [[x, y, w, h, class_id], ...]

        @return [[x, y, w, h, score, c0, c1, ...], ...]
        """
        ground_truth = [
            np.zeros(
                (
                    self.grid_size[i],
                    self.grid_size[i],
                    3,
                    5 + self.num_classes,
                ),
                dtype=np.float32,
            )
            for i in range(3)
        ]

        for bbox in bboxes:
            # [x, y, w, h, class_id]
            xywh = np.array(bbox[:4], dtype=np.float32)
            class_id = int(bbox[4])

            # smooth_onehot = [0.xx, ... , 1-(0.xx*(n-1)), 0.xx, ...]
            onehot = np.zeros(self.num_classes, dtype=np.float32)
            onehot[class_id] = 1.0
            uniform_distribution = np.full(
                self.num_classes, 1.0 / self.num_classes, dtype=np.float32
            )
            delta = 0.01
            smooth_onehot = (1 - delta) * onehot + delta * uniform_distribution

            ious = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((3, 4), dtype=np.float32)
                anchors_xywh[:, 0:2] = xywh[0:2]
                anchors_xywh[:, 2:4] = self.anchors_ratio[i]
                iou = train.bbox_iou(xywh, anchors_xywh)
                ious.append(iou)
                iou_mask = iou > 0.3

                if np.any(iou_mask):
                    xy_grid = xywh[0:2] * self.grid_size[i]
                    xy_index = np.floor(xy_grid)
                    dxdy = xy_grid - xy_index
                    delta = (self.xysclaes[i] - 1) / 2

                    exist_positive = True
                    for anchor_index, mask in enumerate(iou_mask):
                        if mask:
                            left_mask = dxdy[0] < delta
                            right_mask = dxdy[0] > 1 - delta
                            top_mask = dxdy[1] < delta
                            bottom_mask = dxdy[1] > 1 - delta

                            coordinates = []
                            if left_mask:
                                coordinates.append(xy_index + (-1, 0))
                                if top_mask:
                                    coordinates.append(xy_index + (-1, -1))
                                elif bottom_mask:
                                    coordinates.append(xy_index + (-1, 1))
                            elif right_mask:
                                coordinates.append(xy_index + (1, 0))
                                if top_mask:
                                    coordinates.append(xy_index + (1, -1))
                                elif bottom_mask:
                                    coordinates.append(xy_index + (1, 1))
                            else:
                                coordinates.append(xy_index)
                                if top_mask:
                                    coordinates.append(xy_index + (0, -1))
                                elif bottom_mask:
                                    coordinates.append(xy_index + (0, 1))
                            for coordinate in coordinates:
                                _anchor = anchor_index
                                _x = max(
                                    min(
                                        int(coordinate[0] + 0.01),
                                        self.grid_size[i] - 1,
                                    ),
                                    0,
                                )
                                _y = max(
                                    min(
                                        int(coordinate[1] + 0.01),
                                        self.grid_size[i] - 1,
                                    ),
                                    0,
                                )
                                ground_truth[i][_y, _x, _anchor, 0:4] = xywh
                                ground_truth[i][_y, _x, _anchor, 4:5] = 1.0
                                ground_truth[i][
                                    _y, _x, _anchor, 5:
                                ] = smooth_onehot

            if not exist_positive:
                i = np.argmax(np.array(ious).reshape(-1))
                i = i // 3
                anchor_index = i % 3

                xy_grid = xywh[0:2] * self.grid_size[i]
                xy_index = np.floor(xy_grid)
                dxdy = xy_grid - xy_index
                delta = (self.xysclaes[i] - 1) / 2
                left_mask = dxdy[0] < delta
                right_mask = dxdy[0] > 1 - delta
                top_mask = dxdy[1] < delta
                bottom_mask = dxdy[1] > 1 - delta

                coordinates = []
                if left_mask:
                    coordinates.append(xy_index + (-1, 0))
                    if top_mask:
                        coordinates.append(xy_index + (-1, -1))
                    elif bottom_mask:
                        coordinates.append(xy_index + (-1, 1))
                elif right_mask:
                    coordinates.append(xy_index + (1, 0))
                    if top_mask:
                        coordinates.append(xy_index + (1, -1))
                    elif bottom_mask:
                        coordinates.append(xy_index + (1, 1))
                else:
                    coordinates.append(xy_index)
                    if top_mask:
                        coordinates.append(xy_index + (0, -1))
                    elif bottom_mask:
                        coordinates.append(xy_index + (0, 1))

                for coordinate in coordinates:
                    _anchor = anchor_index
                    _x = max(
                        min(int(coordinate[0] + 0.01), self.grid_size[i] - 1,),
                        0,
                    )
                    _y = max(
                        min(int(coordinate[1] + 0.01), self.grid_size[i] - 1,),
                        0,
                    )
                    ground_truth[i][_y, _x, _anchor, 0:4] = xywh
                    ground_truth[i][_y, _x, _anchor, 4:5] = 1.0
                    ground_truth[i][_y, _x, _anchor, 5:] = smooth_onehot

        ground_truth = np.concatenate(
            [
                ground_truth[i].reshape((-1, 5 + self.num_classes))
                for i in range(3)
            ],
            axis=0,
        )

        return ground_truth

    def preprocess_dataset(self, dataset):
        """
        @param dataset:
            yolo: [image_path, [[x, y, w, h, class_id], ...]]
            converted_coco: unit=> pixel
                [image_path, [[x, y, w, h, class_id], ...]]

        @return image / 255, ground_truth
        """
        image_path = dataset[0]
        if not os.path.exists(image_path):
            raise KeyError("{} does not exist".format(image_path))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.dataset_type == "converted_coco":
            height, width, _ = image.shape
            dataset[1] = dataset[1] / np.array(
                [width, height, width, height, 1]
            )

        resized_image, resized_bboxes = media.resize(
            image, self.input_size, dataset[1]
        )
        resized_image = np.expand_dims(resized_image / 255.0, axis=0)
        ground_truth = self.bboxes_to_ground_truth(resized_bboxes)
        ground_truth = np.expand_dims(ground_truth, axis=0)

        if self.data_augmentation:
            # TODO
            # BoF functions
            pass
        return resized_image, ground_truth

    def __iter__(self):
        self.count = 0
        np.random.shuffle(self.dataset)
        return self

    def __next__(self):
        x, y = self.preprocess_dataset(self.dataset[self.count])

        self.count += 1
        if self.count == len(self.dataset):
            np.random.shuffle(self.dataset)
            self.count = 0

        return x, y

    def __len__(self):
        return len(self.dataset)


def read_classes_names(classes_name_path):
    """loads class name from a file"""
    names = {}
    with open(classes_name_path, "r") as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip("\n")
    return names
