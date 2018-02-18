from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import range
from builtins import object
import numpy as np
import os
import tensorflow as tf
import cv2
from fast_rcnn import FastRCNN
from neptune_handler import NeptuneHandler


class Trainer(object):

    def __init__(self):
        self.handler = NeptuneHandler(num_channel_names=['regression loss', 'classification loss', 'total loss'],
                                      charts_desc=[('regression loss', 'regression loss (mean of 20 steps)'),
                                                   ('classification loss', 'classification loss (mean of 20 steps)'),
                                                   ('total loss', 'total loss')],
                                      im_channel_names=['region proposals for RoI pooling', 'network detections'])
        self.send_im = False
        im_paths = self.get_im_paths()
        roi_paths = self.get_roi_paths()
        pretrained_path = self.handler.pretrained_path
        self.im_paths = np.asarray(im_paths)
        self.roi_paths = np.asarray(roi_paths)
        self.net = FastRCNN(path=pretrained_path)

    def get_im_paths(self):
        self.im_names = os.listdir(self.handler.im_folder)
        paths = [os.path.join(self.handler.im_folder, el) for el in self.im_names]
        return paths

    def get_roi_paths(self):
        paths = [os.path.join(self.handler.roi_folder, el + '.npy') for el in self.im_names]
        return paths

    def generate_positive_roi(self, gt_boxes):
        res = []
        num_ = []
        for j in range(32):
            epsilon1 = np.random.randint(-10, 10)
            epsilon2 = np.random.randint(-10, 10)
            num_box = np.random.randint(0, len(gt_boxes))
            bxes = gt_boxes[num_box] + np.asarray([epsilon1, epsilon2, epsilon1, epsilon2])
            bxes /= 16
            res.append(bxes)
            num_.append(num_box)
        return np.asarray(res), np.asarray(num_)

    def generate_negative_roi(self, gt_boxes):
        res = []
        for j in range(96):
            epsilon1 = np.random.randint(50, 300)
            epsilon2 = np.random.randint(50, 300)
            num_box = np.random.randint(0, len(gt_boxes))
            bxes = gt_boxes[num_box] + np.asarray([epsilon1, epsilon2, epsilon1 + 100, epsilon2 + 75])
            bxes /= 16
            res.append(bxes)
        return np.asarray(res)

    def relabel(self, gt_boxes, old_shape, new_shape):
        mult = np.asarray([float(new_shape[1])/old_shape[1],
                           float(new_shape[0])/old_shape[0],
                           float(new_shape[1])/old_shape[1],
                           float(new_shape[0])/old_shape[0]])
        bxes = np.copy(gt_boxes)
        bxes = np.float32(bxes)
        for el in bxes:
            el *= mult
        return bxes

    def bbox_transform(self, boxes, shape):
        boxes_ = np.copy(boxes)
        boxes_ = np.float32(boxes_)
        dw = 1./shape[1]
        dh = 1./shape[0]
        x = (boxes_[:, 0] + boxes_[:, 2])*dw/2.
        y = (boxes_[:, 1] + boxes_[:, 3])*dh/2.
        w = (boxes_[:, 2] - boxes_[:, 0])*dw
        h = (boxes_[:, 3] - boxes_[:, 1])*dh
        x = np.reshape(x, (-1, 1))
        y = np.reshape(y, (-1, 1))
        w = np.reshape(w, (-1, 1))
        h = np.reshape(h, (-1, 1))
        return np.hstack([x, y, w, h])

    def transform_inv(self, output, shape):
        dx = shape[1]
        dy = shape[0]
        x = output[:, 0]*dx
        y = output[:, 1]*dy
        w = output[:, 2]*dx
        h = output[:, 3]*dy
        x -= w/2.
        y -= h/2.
        x2 = x + w
        y2 = y + h
        x = np.reshape(x, (-1, 1))
        y = np.reshape(y, (-1, 1))
        x2 = np.reshape(x2, (-1, 1))
        y2 = np.reshape(y2, (-1, 1))
        return np.hstack([x, y, x2, y2])

    def connect_roi_pooling(self):
        self.net.add_roi_pooling()

    def connect_detector(self):
        self.net.connect_detector()

    def get_top_k(self, boxes, scores, k):
        a = np.argsort(np.squeeze(scores))[::-1]
        return boxes[a[:k]]

    # by Ross Girshick

    def non_maximum_supression(self, dets, thresh):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def fit(self):

        def manage_image_sending(string):
            self.send_im = not self.send_im
            if self.send_im:
                return 'image sending on'
            else:
                return 'image sending off'

        self.handler.ctx.job.register_action(name='image sending', handler=manage_image_sending)

        with tf.Session(config=tf.ConfigProto(
                                allow_soft_placement=True,
                                log_device_placement=False)) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            k = 0
            ts = 0
            losses_ = [[], [], []]
            for i in range(self.handler.num_epochs):
                ids = np.arange(len(self.im_paths))
                np.random.shuffle(ids)
                im_paths = self.im_paths[ids]
                roi_paths = self.roi_paths[ids]
                for m in range(len(im_paths)):
                    im = cv2.imread(im_paths[m]).astype('float32')/255.
                    gt_boxes = np.load(roi_paths[m])
                    gt_boxes = self.relabel(gt_boxes, im.shape, (576, 1920))
                    im = cv2.resize(im, (1920, 576))
                    x = im.reshape((1, im.shape[0], im.shape[1], 3))
                    positive_rois, ob_numbers = self.generate_positive_roi(gt_boxes)
                    negative_rois = self.generate_negative_roi(gt_boxes)
                    rois = np.vstack([positive_rois, negative_rois])
                    y_ = np.zeros((len(rois), ), dtype=np.int32)
                    zeros = np.zeros((len(rois), 1))
                    rois = np.hstack([zeros, rois])
                    rois = np.int32(rois)
                    y_[:len(positive_rois)] = 1
                    reg = np.zeros((len(rois), 4))
                    boxes_tr = self.bbox_transform(gt_boxes, im.shape)
                    for j in range(32):
                        reg[j] = boxes_tr[ob_numbers[j]]
                    feed_dict = {self.net.x: x,
                                 self.net.y: reg,
                                 self.net.y_: y_,
                                 self.net.roidb: rois,
                                 self.net.learn_rate: self.handler.learn_rate}
                    sess.run(self.net.opt, feed_dict=feed_dict)
                    tot = sess.run(self.net.loss_, feed_dict=feed_dict)
                    reg = sess.run(self.net.reg_loss, feed_dict=feed_dict)
                    cls = sess.run(self.net.class_loss, feed_dict=feed_dict)
                    losses_[0].append(reg)
                    losses_[1].append(cls)
                    losses_[2].append(tot)
                    if m % 20 == 0:
                        reg = sum(losses_[0])/len(losses_[0])
                        cls = sum(losses_[1])/len(losses_[1])
                        tot = sum(losses_[2])/len(losses_[2])
                        losses_ = [[], [], []]
                        self.handler.send_to_neptune(ts, [reg, cls, tot])
                        ts += 1
                    if self.send_im:
                        boxes = sess.run(self.net.boxes, feed_dict=feed_dict)
                        logits = sess.run(self.net.logits, feed_dict=feed_dict)
                        boxes = self.transform_inv(boxes, im.shape)
                        logits = logits[:, 1]
                        scores = np.reshape(logits, (-1, 1))
                        boxes = np.hstack([boxes, scores])
                        boxes = self.get_top_k(boxes, logits, 20)
                        keep = self.non_maximum_supression(boxes, .3)
                        boxes = boxes[keep, :4]
                        roi_pool = rois[:, 1:]*np.asarray([16, 16, 16, 16])
                        self.handler.send_image_with_proposals(k, im[:, :, [2, 1, 0]], boxes, im.shape)
                        self.handler.send_image_with_proposals(k, im[:, :, [2, 1, 0]], roi_pool, im.shape, True)
                        k += 1
