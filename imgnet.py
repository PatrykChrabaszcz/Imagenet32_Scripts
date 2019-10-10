from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


import os
import torch
import torch.utils.data as data

##The VisionDataset and StandardTransform code  is copied form torchvision source code.
class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")
#for backwards - compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""


class StandardTransform(object):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input, target):
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self):
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)


class IMGNET(VisionDataset):
    """
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        classes (integer, default: 10): number of classes. will search folder 
            imgnet{classes} for file.
        size (integer, default: 32): size X size should be the image size.
    """


    train_list = []
    for i in range(10):
        train_list.append("train_data_batch_%d" % (i+1))

    test_list = ['val_data']

    def __init__(self, root, train=True, transform=None, target_transform=None, classes = 10, size=32):

        super(IMGNET, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        self.base_folder = "imgnet%d" % classes

        self.data = []
        self.targets = []

        if self.train:
            file_list = self.train_list
        else:
            file_list = self.test_list

        mean = 0.0
        total = 0
        relabeling = None
#now load the picked numpy arrays
        for file_name in file_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')


                x = entry['data']
                y = entry['labels']

                total += len(y)
                y = [i-1 for i in y]

                if relabeling is None:
                    sorted_labels = np.sort(np.unique(y))
                    assert(len(sorted_labels)  == classes)
                    relabeling = {b:i for i,b in enumerate(sorted_labels)}
                y = [relabeling[i] for i in y]

                self.data.append(x)
                self.targets.extend(y)

        self.mean = mean/float(total)
        self.data = np.vstack(self.data).reshape(-1, 3, size, size)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

#doing this so that it is consistent with all other datasets
#to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


##Use stats to extract mean and standard error (mean stderr per channel) for normalization.
    def stats(self):
        means=np.asarray([0,0,0],dtype=float)
        for i in range(len(dset)):
            img, label = dset.__getitem__(i)
            means += np.mean(img,(0,1))

        means  = means/len(dset)
        print("Mean values (per channel): %s" % means)

        stds=np.asarray([0,0,0],dtype=float)
        total = 0
        for i in range(len(dset)):
            img, label = dset.__getitem__(i)
            stds += np.std(img,(0,1))

        stds  = stds/len(dset)
        print("Stadard Error (per channel): %s" % stds)
        return means, stds
