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
import tensorflow as tf


def make_compiled_loss(
    model,
    iou_type: str = "ciou",
    iou_threshold: float = 0.5,
    classes_threshold: float = 0.25,
    score_classes_threshold: float = 0.25,
):
    if iou_type == "giou":
        xiou_func = bbox_giou
    elif iou_type == "ciou":
        xiou_func = bbox_ciou

    def compiled_loss(y, y_pred):
        """
        @param y:      (batch, candidates, (x, y, w, h, score, c0, c1, ...))
        @param y_pred: (batch, candidates, (x, y, w, h, score, c0, c1, ...))
        """
        y_xywh = y[..., 0:4]
        y_score = y[..., 4:5]
        y_classes = y[..., 5:]

        batch_size = y.shape[0]

        y_pred_xywh = y_pred[..., 0:4]
        y_pred_raw_score = y_pred[..., 4:5]
        y_pred_raw_classes = y_pred[..., 5:]

        y_pred_score = tf.keras.activations.sigmoid(y_pred_raw_score)

        # XIoU loss
        xiou = tf.expand_dims(xiou_func(y_xywh, y_pred_xywh), axis=-1)
        xiou_loss = y_score * (1 - xiou)

        # Score loss
        max_iou = []
        for i in range(batch_size):
            # @param bboxes1: candidates, 1,       xywh
            # @param bboxes2: 1         , answers, xywh
            # @return candidates, answers
            object_mask = tf.reshape(y_score[i, ...], shape=(-1,)) > 0.5
            iou = bbox_iou(
                tf.expand_dims(y_pred_xywh[i, ...], axis=1),
                tf.expand_dims(
                    tf.boolean_mask(y_xywh[i, ...], object_mask), axis=0,
                ),
            )

            max_iou.append(
                tf.reshape(tf.reduce_max(iou, axis=-1), shape=(1, -1, 1))
            )
        max_iou = tf.concat(max_iou, axis=0)
        low_iou_mask = max_iou < iou_threshold

        max_classes = tf.reduce_max(y_pred_raw_classes, axis=-1, keepdims=True)
        low_classes_mask = max_classes < classes_threshold

        low_iou_prob_mask = tf.math.logical_or(low_iou_mask, low_classes_mask)
        low_iou_prob_mask = tf.cast(low_iou_prob_mask, tf.float32)

        score_scale = tf.abs(y_pred_xywh[..., 2:3] * y_pred_xywh[..., 3:4])

        score_loss = (
            y_score + (1.0 - y_score) * score_scale * low_iou_prob_mask
        ) * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_score, logits=y_pred_raw_score
        )

        # Classes loss
        score_classes = y_pred_score * y_pred_raw_classes
        high_score_classes_mask = score_classes > score_classes_threshold
        high_score_classes_mask = tf.cast(high_score_classes_mask, tf.float32)

        classes_loss = (
            y_score * y_classes
            + (1 - y_score) * high_score_classes_mask * low_iou_prob_mask
        ) * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_classes, logits=y_pred_raw_classes
        )

        xiou_loss = tf.reduce_mean(tf.reduce_sum(xiou_loss, axis=[1, 2]))
        score_loss = tf.reduce_mean(tf.reduce_sum(score_loss, axis=[1, 2]))
        classes_loss = tf.reduce_mean(tf.reduce_sum(classes_loss, axis=[1, 2]))

        return xiou_loss, score_loss, classes_loss

    return compiled_loss


def bbox_iou(bboxes1, bboxes2):
    """
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1

    @return (max(a,A), max(b,B), ...)

    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    return iou


def bbox_giou(bboxes1, bboxes2):
    """
    Generalized IoU
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1

    @return (max(a,A), max(b,B), ...)

    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )

    enclose_section = enclose_right_down - enclose_left_up
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

    giou = iou - tf.math.divide_no_nan(enclose_area - union_area, enclose_area)

    return giou


def bbox_ciou(bboxes1, bboxes2):
    """
    Complete IoU
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1

    @return (max(a,A), max(b,B), ...)

    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )

    enclose_section = enclose_right_down - enclose_left_up

    c_2 = enclose_section[..., 0] ** 2 + enclose_section[..., 1] ** 2

    center_diagonal = bboxes2[..., :2] - bboxes1[..., :2]

    rho_2 = center_diagonal[..., 0] ** 2 + center_diagonal[..., 1] ** 2

    diou = iou - tf.math.divide_no_nan(rho_2, c_2)

    v = (
        (
            tf.math.atan(
                tf.math.divide_no_nan(bboxes1[..., 2], bboxes1[..., 3])
            )
            - tf.math.atan(
                tf.math.divide_no_nan(bboxes2[..., 2], bboxes2[..., 3])
            )
        )
        * 2
        / np.pi
    ) ** 2

    alpha = tf.math.divide_no_nan(v, 1 - iou + v)

    ciou = diou - alpha * v

    return ciou
