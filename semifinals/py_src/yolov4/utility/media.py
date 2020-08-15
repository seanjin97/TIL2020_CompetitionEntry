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

import cv2
import colorsys
import numpy as np


def resize(image: np.ndarray, input_size: int, ground_truth: np.ndarray = None):
    """
    @param ground_truth: [[center_x, center_y, w, h, class_id], ...]

    Usage:
        image = media.resize(image, yolo.input_size)
        image, ground_truth = media.resize(image, yolo.input_size, ground_truth)
    """
    height, width, _ = image.shape

    if max(height, width) != input_size:
        scale = min(input_size / width, input_size / height)
        new_width = round(width * scale)
        new_height = round(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height))
    else:
        new_width = width
        new_height = height
        resized_image = np.copy(image)

    dw = int(input_size - new_width)
    dh = int(input_size - new_height)

    if dw != 0 or dh != 0:
        dw = dw // 2
        dh = dh // 2
        padded_image = np.full((input_size, input_size, 3), 255, dtype=np.uint8)
        padded_image[
            dh : new_height + dh, dw : new_width + dw, :
        ] = resized_image
    else:
        padded_image = resized_image

    if ground_truth is None:
        return padded_image

    ground_truth = np.copy(ground_truth)

    if dw != 0:
        w_h = new_width / new_height
        ground_truth[:, 0] = w_h * (ground_truth[:, 0] - 0.5) + 0.5
        ground_truth[:, 2] = w_h * ground_truth[:, 2]
    else:
        h_w = new_height / new_width
        ground_truth[:, 1] = h_w * (ground_truth[:, 1] - 0.5) + 0.5
        ground_truth[:, 3] = h_w * ground_truth[:, 3]

    return padded_image, ground_truth


def draw_bbox(image: np.ndarray, bboxes: np.ndarray, classes: dict):
    """
    @parma image: (height, width, channel)
    @param bboxes: (candidates, 4) or (candidates, 5)
            [[center_x, center_y, w, h, class_id], ...]
            [[center_x, center_y, w, h, class_id, propability], ...]

    Usage:
        image = media.draw_bbox(image, bboxes, classes)
    """
    image = np.copy(image)
    height, width, _ = image.shape
    max_size = max(height, width)

    # Create colors
    num_classes = len(classes)
    hsv_tuples = [(1.0 * x / num_classes, 1.0, 1.0) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(
            lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors,
        )
    )

    if bboxes.shape[-1] == 5:
        bboxes = np.concatenate(
            [bboxes, np.full((*bboxes.shape[:-1], 1), 2.0)], axis=-1
        )
    else:
        bboxes = np.copy(bboxes)

    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * width
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * height

    for bbox in bboxes:
        c_x = int(bbox[0])
        c_y = int(bbox[1])
        half_w = int(bbox[2] / 2)
        half_h = int(bbox[3] / 2)
        c_min = (c_x - half_w, c_y - half_h)
        c_max = (c_x + half_w, c_y + half_h)
        class_id = int(bbox[4])
        bbox_color = colors[class_id]
        font_size = min(max_size / 1500, 0.7)
        font_thickness = 1 if max_size < 1000 else 2

        cv2.rectangle(image, c_min, c_max, bbox_color, 3)

        bbox_text = "{}: {:.1%}".format(classes[class_id], bbox[5])
        t_size = cv2.getTextSize(bbox_text, 0, font_size, font_thickness)[0]
        cv2.rectangle(
            image,
            c_min,
            (c_min[0] + t_size[0], c_min[1] - t_size[1] - 3),
            bbox_color,
            -1,
        )
        cv2.putText(
            image,
            bbox_text,
            (c_min[0], c_min[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (0, 0, 0),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return image

def getClasses(image: np.ndarray, bboxes: np.ndarray, classes: dict):
    """
    @parma image: (height, width, channel)
    @param bboxes: (candidates, 4) or (candidates, 5)
            [[center_x, center_y, w, h, class_id], ...]
            [[center_x, center_y, w, h, class_id, propability], ...]

    Usage:
        image = media.draw_bbox(image, bboxes, classes)
    """
    image = np.copy(image)
    height, width, _ = image.shape
    max_size = max(height, width)
    predictions = []
    # Create colors
    # num_classes = len(classes)
    # hsv_tuples = [(1.0 * x / num_classes, 1.0, 1.0) for x in range(num_classes)]
    # colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    # colors = list(
    #     map(
    #         lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
    #         colors,
    #     )
    # )

    if bboxes.shape[-1] == 5:
        bboxes = np.concatenate(
            [bboxes, np.full((*bboxes.shape[:-1], 1), 2.0)], axis=-1
        )
    else:
        bboxes = np.copy(bboxes)

    # bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * width
    # bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * height

    for bbox in bboxes:
        c_x = int(bbox[0])
        c_y = int(bbox[1])
        # half_w = int(bbox[2] / 2)
        # half_h = int(bbox[3] / 2)
        # c_min = (c_x - half_w, c_y - half_h)
        # c_max = (c_x + half_w, c_y + half_h)
        class_id = int(bbox[4])
        predictions.append(class_id)
        # bbox_color = colors[class_id]
        # font_size = min(max_size / 1500, 0.7)
        # font_thickness = 1 if max_size < 1000 else 2

        # cv2.rectangle(image, c_min, c_max, bbox_color, 3)

        # bbox_text = "{}: {:.1%}".format(classes[class_id], bbox[5])
        # t_size = cv2.getTextSize(bbox_text, 0, font_size, font_thickness)[0]
        # cv2.rectangle(
        #     image,
        #     c_min,
        #     (c_min[0] + t_size[0], c_min[1] - t_size[1] - 3),
        #     bbox_color,
        #     -1,
        # )
        # cv2.putText(
        #     image,
        #     bbox_text,
        #     (c_min[0], c_min[1] - 2),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     font_size,
        #     (0, 0, 0),
        #     font_thickness,
        #     lineType=cv2.LINE_AA,
        # )

    return predictions