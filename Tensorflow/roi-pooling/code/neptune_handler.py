from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import range
from builtins import object
from deepsense import neptune
import numpy as np
import cv2
from PIL import Image


class NeptuneHandler(object):

    def __init__(self, num_channel_names, charts_desc, im_channel_names):
        self.ctx = neptune.Context()
        self.learn_rate = self.ctx.params.learning_rate
        self.num_epochs = int(self.ctx.params.num_epochs)
        self.roi_folder = self.ctx.params.roidb
        self.im_folder = self.ctx.params.im_folder
        self.pretrained_path = self.ctx.params.pretrained_path
        self.create_numeric_channels(num_channel_names, self.ctx)
        self.create_charts(charts_desc, self.ctx)
        self.create_image_channels(im_channel_names, self.ctx)

    def create_numeric_channels(self, channel_names, ctx):
        self.numerical_channels = [ctx.job.create_channel(name=name, channel_type=neptune.ChannelType.NUMERIC)
                                   for name in channel_names]

    def create_charts(self, charts_desc, ctx):
        self.charts = [ctx.job.create_chart(name=charts_desc[i][0],
                                            series={charts_desc[i][1]: self.numerical_channels[i]})
                       for i in range(len(self.numerical_channels))]

    def send_to_neptune(self, time_point, values):
        send = lambda ch, val: ch.send(x=time_point, y=val)
        for i in range(len(self.numerical_channels)):
            send(self.numerical_channels[i], values[i])

    def create_image_channels(self, channel_names, ctx):
        self.im_channels = [ctx.job.create_channel(name=name, channel_type=neptune.ChannelType.IMAGE)
                            for name in channel_names]

    def send_image_with_proposals(self, time_step, im, proposals, shape, rois=False):
        width = 340
        height = 150
        im_ = cv2.resize(im, (width, height))
        im_ = np.uint8(im_*255.)
        for proposal in proposals:
            x1 = int(width*proposal[0]/float(shape[1]))
            y1 = int(height*proposal[1]/float(shape[0]))
            x2 = int(width*proposal[2]/float(shape[1]))
            y2 = int(height*proposal[3]/float(shape[0]))
            cv2.rectangle(im_, (x1, y1), (x2, y2), (255, 0, 0), 1)
        pil_im = Image.fromarray(im_)
        if rois:
            neptune_im = neptune.Image(name='all the RoIs', description='region proposals', data=pil_im)
            self.im_channels[0].send(x=time_step, y=neptune_im)
        else:
            neptune_im = neptune.Image(name='chosen RoIs', description='object detections', data=pil_im)
            self.im_channels[1].send(x=time_step, y=neptune_im)
