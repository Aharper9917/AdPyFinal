import random
import torch
from PIL import Image
import math
import sys
import random
from PIL import Image
import numpy as np
import types
import collections
import torchvision.transforms

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

from torchvision.transforms import functional as F


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)

            bbox = target["boxes"]
            bbox[:, [0, 2]] = 1 - bbox[:, [2, 0]]

            target["boxes"] = bbox

        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize(object):
    """
        Normalize a tensor image with mean and standard deviation.
        Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
        will normalize each channel of the input ``torch.*Tensor`` i.e.
        ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

        .. note::
            This transform acts out of place, i.e., it does not mutates the input tensor.

        Args:
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor, target):
        """
            Args:
                tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

            Returns:
                Tensor: Normalized Tensor image.
        """
        # print(type(tensor))
        # print(tensor)
        # time.sleep(10000)
        return F.normalize(tensor, self.mean, self.std, self.inplace), target

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Resize(object):
    """
        Resize the input PIL Image to the given size.

        Args:
            size (sequence or int): Desired output size. If size is a sequence like
                (h, w), output size will be matched to this. If size is an int,
                smaller edge of the image will be matched to this number.
                i.e, if height > width, then image will be rescaled to
                (size * height / width, size)
            interpolation (int, optional): Desired interpolation. Default is
                ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (
            isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, target):
        """
            Args:
                img (PIL Image): Image to be scaled.

            Returns:
                PIL Image: Rescaled image.
        """
        return F.resize(img, self.size, self.interpolation), target

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)
