# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import random
import time
from pathlib import Path
from util import box_ops
import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from util import box_plot_utils
from engine import evaluate, train_one_epoch
from models import build_model
############## WIDER adaptation
from datasets.wider.WIDERFaceDataset import WIDERFaceDataset
from os.path import join
import cv2
from util.nms import nms

##################################3



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--num_classes', default=91, type=int,
                        help='for coco object detection')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    # Model parameters
    # Changed for face detection
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="Path to the pretrained model")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    # change from 1 to 5
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    #
    parser.add_argument('--ce_loss_coef', default=1, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.2, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # Face detection evaluation
    parser.add_argument('--dataset_path', default='', type=str, help='path to dataset')
    parser.add_argument('--coco_pretrained', action='store_true')
    parser.add_argument('--box_detection_score', default=0.9, type=float)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--freeze_decoder', action='store_true')
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--plot_eval', action='store_true')
    parser.add_argument('--checkpoint_prefix', default='fdetr', type=str)
    parser.add_argument('--nms_overlap_th', default=0.3, type=float)
    parser.add_argument('--light_detection', action='store_true')
    parser.add_argument('--with_prior', action='store_true')
    parser.add_argument('--tloss', action='store_true')
    parser.add_argument('--label_by_size', action='store_true')
    return parser


def main(args):
    print(args)
    device = torch.device(args.device)

    labels_mapper = {}
    if args.coco_pretrained:
        # ad hoc
        labels_mapper[1] = "person"
        labels_mapper[3] = "car"
        labels_mapper[5] = "airplane"
        labels_mapper[8] = "truck"
        labels_mapper[27] = "backpack"
        labels_mapper[65] = "bed"
    elif args.label_by_size:
        args.num_classes = 5
        labels_mapper[0] = "face"
        labels_mapper[1] = "face"
        labels_mapper[2] = "face"
        labels_mapper[3] = "face"
        labels_mapper[4] = "face"
    else:
        args.num_classes = 1
        labels_mapper[0] = "face"


    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    #### Face detection
    import datasets.transforms as T

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if args.freeze_encoder:
        max_size = 1333
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    else:
        max_size = 608
        scales = [480, 512, 544, 576, 608]

    img_transforms_train = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    img_transforms_val = normalize

    dataset_file = join(args.dataset_path, "WIDER_{}_annotations.txt")
    dataset_train = WIDERFaceDataset(args.dataset_path, dataset_file.format('train'), 'train', img_transforms=img_transforms_train)
    dataset_val = WIDERFaceDataset(args.dataset_path, dataset_file.format('val'), 'val', img_transforms=img_transforms_val)
    ####################################3

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)

    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    output_dir = Path(args.output_dir)

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        if args.coco_pretrained:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'], strict=False)

    indices_to_plot = [0,1,2,7,10,20,30]
    if args.eval:
        with torch.no_grad():
            model.eval()
            criterion.eval()

            # Validate training augmentations
            '''
            for idx, (sample, targets) in enumerate(data_loader_train):
                sample = sample.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                gt_boxes = targets[0]['boxes']
                target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                img_h, img_w = target_sizes.unbind(1)
                scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(gt_boxes.device)
                gt_boxes = (gt_boxes * scale_fct)
                gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
                outputs = model(sample)
                results = postprocessors['bbox'](outputs, target_sizes)[0]
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = (sample.tensors[0].permute(1, 2, 0).cpu().numpy() * std) + mean  # need to change to BGR

                if args.plot_eval and idx in indices_to_plot:
                    box_labels = results["labels"].cpu().numpy()
                    box_coords = results["boxes"].cpu().numpy()
                    scores = results["scores"].cpu().numpy()
                    confident_boxes = scores > args.box_detection_score
                    selected_box_labels = box_labels[confident_boxes]
                    selected_box_coords = box_coords[confident_boxes]
                    selected_box_labels = [labels_mapper[label] for label in selected_box_labels]
                break


            '''
            for idx, (sample, targets) in enumerate(data_loader_val):
                sample = sample.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                gt_boxes = targets[0]['boxes']
                target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                img_h, img_w = target_sizes.unbind(1)
                scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(gt_boxes.device)
                gt_boxes = (gt_boxes * scale_fct)

                gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
                print(gt_boxes)
                tic = time.time()
                outputs = model(sample)
                print(time.time()-tic)
                results = postprocessors['bbox'](outputs, target_sizes)[0]

                # collect stats
                if not args.coco_pretrained: # actually training on face detection
                    pass # TODO some stats
                box_labels = results["labels"].cpu().numpy()
                print(box_labels)
                box_coords = results["boxes"].cpu().numpy()
                scores = results["scores"].cpu().numpy()

                if args.plot_eval and idx in indices_to_plot:
                    confident_boxes = scores > args.box_detection_score
                    selected_box_labels = box_labels[confident_boxes]
                    selected_box_coords = box_coords[confident_boxes]
                    selected_box_labels = [labels_mapper[label] for label in selected_box_labels]

                    '''
                    if args.coco_pretrained:
                        selected_box_labels = [labels_mapper[label] for label in selected_box_labels]
                    else:
                        selected_box_labels = None
                        if args.nms_overlap_th > 0.0:
                            # apply nms
                            selected_box_coords_indices = nms(selected_box_coords, scores[is_box], args.nms_overlap_th)
                            selected_box_coords = selected_box_coords[selected_box_coords_indices]
                    '''
                    img = cv2.imread(dataset_val.get_img_path(idx))
                    box_plot_utils.plot_bboxes(img, gt_boxes, selected_box_coords,
                                               selected_box_labels)

            # stats

    if args.train:
        params_to_freeze= []
        if args.freeze_backbone:
            params_to_freeze.append("backbone")
        if args.freeze_encoder:
            params_to_freeze.append("transformer.encoder")
            params_to_freeze.append("input_proj")
        if args.freeze_decoder:
            params_to_freeze.append("transformer.decoder")

        for name, parameter in model.named_parameters():
            for phrase in params_to_freeze:
                if phrase in name:
                    parameter.requires_grad_(False)
                    print("Freezing param: [{}]".format(name))

        # Train
        print("Start training")
        save_every = args.epochs//6
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch,
                args.clip_max_norm)
            lr_scheduler.step()
            print(train_stats)

            if args.output_dir:
                checkpoint_paths = [output_dir / '{}_checkpoint_final.pth'.format(args.checkpoint_prefix)]
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % save_every == 0:
                    checkpoint_paths.append(output_dir / '{}_epoch{}_checkpoint.pth'.format(args.checkpoint_prefix, epoch+1))
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('F-DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
