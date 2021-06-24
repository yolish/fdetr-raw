import numpy as np
from PIL import Image
from torchvision import transforms
from copy import deepcopy
import warnings


def augment(image, bboxes, input_size, visualize=False):
    # Randomly resize the image
    rnd = np.random.rand()
    if rnd < 1 / 3:
        # resize by half
        scaled_shape = (int(0.5 * image.height), int(0.5 * image.width))
        image = transforms.functional.resize(image, scaled_shape)
        bboxes = bboxes / 2

    elif rnd > 2 / 3:
        # double size
        scaled_shape = (int(2 * image.height), int(2 * image.width))
        image = transforms.functional.resize(image, scaled_shape)
        bboxes = bboxes * 2

    # convert from PIL Image to ndarray
    img = np.array(image)

    # Get a random crop of the image and keep only relevant bboxes
    img, bboxes, _ = crop_image(img, bboxes, input_size)

    # Random Flip
    flip = np.random.rand() > 0.5
    if flip:
        img = np.fliplr(img).copy()  # flip the image

        lx1, lx2 = np.array(bboxes[:, 0]), np.array(bboxes[:, 2])
        bboxes[:, 0] = input_size[1] - lx2
        bboxes[:, 2] = input_size[1] - lx1

    if visualize:
        # Visualize stuff
        visualize.visualize_bboxes(Image.fromarray(img.astype('uint8'), 'RGB'),
                                   bboxes)
        # and now we exit
        exit(0)

    # img is type float64. Convert it to uint8 so torch knows to treat it like an image
    img = img.astype(np.uint8)

    return img, bboxes


def crop_image(img, bboxes, input_size):
    """
    Crop a 500x500 patch from the image, taking care for smaller images.
    bboxes is the np.array of all bounding boxes [x1, y1, x2, y2]
    """
    # randomly pick a cropping window for the image
    # We keep the second arg to randint at least 1 since randint is [low, high)
    crop_x1 = np.random.randint(0, np.max([1, (img.shape[1] - input_size[1] + 1)]))
    crop_y1 = np.random.randint(0, np.max([1, (img.shape[0] - input_size[0] + 1)]))
    crop_x2 = min(img.shape[1], crop_x1 + input_size[1])
    crop_y2 = min(img.shape[0], crop_y1 + input_size[0])
    crop_h = crop_y2 - crop_y1
    crop_w = crop_x2 - crop_x1

    # place the cropped image in a random location in a `input_size` image
    paste_box = [0, 0, 0, 0]  # x1, y1, x2, y2
    paste_box[0] = np.random.randint(0, input_size[1] - crop_w + 1)
    paste_box[1] = np.random.randint(0, input_size[0] - crop_h + 1)
    paste_box[2] = paste_box[0] + crop_w
    paste_box[3] = paste_box[1] + crop_h

    # set this to average image colors
    # this will later be subtracted in mean image subtraction
    img_buf = np.zeros((input_size + (3,)))

    # add the average image so it gets subtracted later.
    img_means = [0.485, 0.456, 0.406]
    for i, c in enumerate(img_means):
        img_buf[:, :, i] += c
    # img is a int8 array, so we need to scale the values accordingly
    img_buf = (img_buf * 255).astype(np.int8)

    img_buf[paste_box[1]:paste_box[3], paste_box[0]:paste_box[2], :] = img[crop_y1:crop_y2, crop_x1:crop_x2, :]

    if bboxes.shape[0] > 0:
        # check if overlap is above negative threshold
        tbox = deepcopy(bboxes)
        tbox[:, 0] = np.maximum(tbox[:, 0], crop_x1)
        tbox[:, 1] = np.maximum(tbox[:, 1], crop_y1)
        tbox[:, 2] = np.minimum(tbox[:, 2], crop_x2)
        tbox[:, 3] = np.minimum(tbox[:, 3], crop_y2)

        overlap = 1 - rect_dist(tbox, bboxes)

        # adjust the bounding boxes - first for crop and then for random placement
        bboxes[:, 0] = bboxes[:, 0] - crop_x1 + paste_box[0]
        bboxes[:, 1] = bboxes[:, 1] - crop_y1 + paste_box[1]
        bboxes[:, 2] = bboxes[:, 2] - crop_x1 + paste_box[0]
        bboxes[:, 3] = bboxes[:, 3] - crop_y1 + paste_box[1]

        # correct for bbox to be within image border
        bboxes[:, 0] = np.minimum(input_size[1], np.maximum(0, bboxes[:, 0]))
        bboxes[:, 1] = np.minimum(input_size[0], np.maximum(0, bboxes[:, 1]))
        bboxes[:, 2] = np.minimum(input_size[1], np.maximum(1, bboxes[:, 2]))
        bboxes[:, 3] = np.minimum(input_size[0], np.maximum(1, bboxes[:, 3]))

        # check to see if the adjusted bounding box is invalid
        neg_thresh = 0.3
        invalid = np.logical_or(np.logical_or(bboxes[:, 2] <= bboxes[:, 0], bboxes[:, 3] <= bboxes[:, 1]),
                                overlap < neg_thresh)

        # remove invalid bounding boxes
        ind = np.where(invalid)
        bboxes = np.delete(bboxes, ind, 0)

    return img_buf, bboxes, paste_box

def rect_dist(I, J):
    if len(I.shape) == 1:
        I = I[np.newaxis, :]
        J = J[np.newaxis, :]

    # area of boxes
    aI = (I[:, 2] - I[:, 0] + 1) * (I[:, 3] - I[:, 1] + 1)
    aJ = (J[:, 2] - J[:, 0] + 1) * (J[:, 3] - J[:, 1] + 1)

    x1 = np.maximum(I[:, 0], J[:, 0])
    y1 = np.maximum(I[:, 1], J[:, 1])
    x2 = np.minimum(I[:, 2], J[:, 2])
    y2 = np.minimum(I[:, 3], J[:, 3])

    aIJ = (x2 - x1 + 1) * (y2 - y1 + 1) * (np.logical_and(x2 > x1, y2 > y1))

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            iou = aIJ / (aI + aJ - aIJ)
        except (RuntimeWarning, Exception):
            iou = np.zeros(aIJ.shape)

    # set NaN, inf, and -inf to 0
    iou[np.isnan(iou)] = 0
    iou[np.isinf(iou)] = 0

    dist = np.maximum(np.zeros(iou.shape), np.minimum(np.ones(iou.shape), 1 - iou))

    return dist
