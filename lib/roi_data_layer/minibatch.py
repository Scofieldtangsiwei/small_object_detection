# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob

def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)

  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES_1),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  im_blob_large, im_blob_small, im_scales = _get_image_blob(roidb, random_scale_inds)

  blob_large = {'data': im_blob_large}
  blob_small = {'data': im_blob_small}

#  assert len(im_scales) == 1, "Single batch only"
#  assert len(roidb) == 1, "Single batch only"

  # gt boxes: (x1, y1, x2, y2, cls)
  if cfg.TRAIN.USE_ALL_GT:
    # Include all ground truth boxes
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
  else:
    # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
    gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
  gt_boxes_large = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes_small = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes_large[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
  gt_boxes_large[:, 4] = roidb[0]['gt_classes'][gt_inds]
  gt_boxes_small[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[1]
  gt_boxes_small[:, 4] = roidb[0]['gt_classes'][gt_inds]
  blob_large['gt_boxes'] = gt_boxes_large
  blob_small['gt_boxes'] = gt_boxes_small
  blob_large['im_info'] = np.array(
    [im_blob_large.shape[1], im_blob_large.shape[2], im_scales[0]],
    dtype=np.float32)

  blob_small['im_info'] = np.array(
    [im_blob_small.shape[1], im_blob_small.shape[2], im_scales[1]],
    dtype=np.float32)

  return blob_large, blob_small

def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  processed_ims = []
  im_scales = []
  for i in range(num_images):
    im = cv2.imread(roidb[i]['image'])
    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    target_size_1 = cfg.TRAIN.SCALES_1[scale_inds[i]]
    target_size_2 = cfg.TRAIN.SCALES_2[scale_inds[i]]
    im_1, im_2, im_scale_1, im_scale_2 = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size_1, target_size_2
                    ,cfg.TRAIN.MAX_SIZE_1, cfg.TRAIN.MAX_SIZE_2)
    im_scales.append(im_scale_1)
    im_scales.append(im_scale_2)
    processed_ims.append(im_1)
    processed_ims.append(im_2)

  # Create a blob to hold the input images
  blob_large, blob_small = im_list_to_blob(processed_ims)

  return blob_large, blob_small, im_scales
