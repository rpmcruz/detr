# DETR

A simple implementation of [DETR](https://arxiv.org/abs/2005.12872) (End-to-End Object Detection with Transformers) for educational and practical purposes.

The implementation seems pretty faithful to the paper and it seems to produce results in the right direction. One difference is that I use much smaller images. The loss has been pretty optimized, but maybe it could be optimized further (I process each element of the batch separately). When training with PASCAL VOC, each epoch takes a little over 3 minutes (GPU A100).