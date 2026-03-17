
import os
import glob

from paddleseg.datasets import Dataset
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose


@manager.DATASETS.add_component
class Ucfel(Dataset):
    """

    The folder structure is as follow:

    ucf_el
        |
        |--image
        |  |--train
        |  |--test
        |
        |--GT
        |  |--train
        |  |--test

    no-defect ---- 0
    crack ----------1
    contact --------2
    interconnnect -3
    corrosion ------4
    Args:
        transforms (list): Transforms for image.
        dataset_root (str):  dataset directory.
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'test'). Default: 'train'.
        edge (bool, optional): Whether to compute edge while training. Default: False
    """
    NUM_CLASSES = 5      # background, crack, contact, interconnect, corrosion
    IGNORE_INDEX = 255
    IMG_CHANNELS = 3    # whether the image is RGB

    def __init__(self, transforms, dataset_root, mode='train', edge=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = list()
        mode = mode.lower()
        self.mode = mode
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = self.IGNORE_INDEX
        self.edge = edge

        if mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        img_dir = os.path.join(self.dataset_root, 'image')
        label_dir = os.path.join(self.dataset_root, 'GT')
        if self.dataset_root is None or not os.path.isdir(
                self.dataset_root) or not os.path.isdir(
                    img_dir) or not os.path.isdir(label_dir):
            raise ValueError(
                "The dataset is not Found or the folder structure is nonconfoumance."
            )

        label_files = sorted(
            glob.glob(
                os.path.join(label_dir, mode, '*.png')))
        img_files = sorted(
            glob.glob(os.path.join(img_dir, mode, '*.jpg')))
        
        print("mode: ", mode)
        print("Found {} images in the folder {}".format(len(img_files), img_dir))

        self.file_list = [
            [img_path, label_path]
            for img_path, label_path in zip(img_files, label_files)
        ]
