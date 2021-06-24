# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from . import detr, fdetr, fdetrlight, fdetr_w_prior, fdetrlight_w_prior, fdetr_w_tloss

def build_model(args):
    if args.coco_pretrained:
        return detr.build(args)
    else:  # Face detection
        if args.tloss:
            return fdetr_w_tloss.build(args)
        if args.light_detection and args.with_prior:
            return fdetrlight_w_prior.build(args)
        if args.light_detection:
            return fdetrlight.build(args)
        elif args.with_prior:
            return fdetr_w_prior.build(args)
        else:
            return fdetr.build(args)
