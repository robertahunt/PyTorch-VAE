import os
import PIL
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from ete3 import Tree
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile

# from https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/4
import torchvision.transforms.functional as F


"""======================================================"""
REQUIRES_STORAGE = False


###
class ClassRandomSampler(torch.utils.data.sampler.Sampler):
    """
    Plugs into PyTorch Batchsampler Package.
    """

    def __init__(self, batch_size, image_dict, no_images, **kwargs):
        self.batch_size = batch_size
        self.samples_per_class = 2

        #####
        # Image dict is a dictionary with the names of classes as keys,
        # and indices of samples from that class in the dataloader's sampler as values
        self.image_dict = image_dict
        self.no_images = no_images

        #####
        self.classes = list(self.image_dict.keys())

        ####
        self.sampler_length = no_images // self.batch_size
        assert (
            self.batch_size % self.samples_per_class == 0
        ), "#Samples per class must divide batchsize!"

        self.name = "class_random_sampler"
        self.requires_storage = False

    def __iter__(self):
        for _ in range(self.sampler_length):
            subset = []
            ### Random Subset from Random classes
            draws = self.batch_size // self.samples_per_class

            for _ in range(draws):
                class_key = random.choice(self.classes)
                class_ix_list = [
                    random.choice(self.image_dict[class_key])
                    for _ in range(self.samples_per_class)
                ]
                subset.extend(class_ix_list)

            yield subset

    def __len__(self):
        return self.sampler_length


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 255, "constant")


class RoveVAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        use_triplet_sampling=True,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_triplet_sampling = use_triplet_sampling

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose(
            [
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(5, fill=255),
                SquarePad(),
                transforms.Resize(self.patch_size + 20),
                transforms.RandomResizedCrop(self.patch_size, scale=(0.5, 1.5)),
                transforms.ToTensor(),
            ]
        )

        val_transforms = transforms.Compose(
            [
                transforms.RandomVerticalFlip(),
                SquarePad(),
                transforms.Resize(self.patch_size + 20),
                transforms.RandomResizedCrop(self.patch_size, scale=(0.9, 1.1)),
                transforms.ToTensor(),
            ]
        )

        self.train_dataset = Rove(
            self.data_dir,
            split="train",
            transform=train_transforms,
        )

        self.train_dataset_val_transformed = Rove(
            self.data_dir,
            split="train",
            transform=val_transforms,
        )

        self.val_dataset = Rove(
            self.data_dir,
            split="val",
            transform=val_transforms,
        )

        self.test_dataset = Rove(
            self.data_dir,
            split="test",
            transform=val_transforms,
        )

        if self.use_triplet_sampling:
            train_image_dict, train_no_samples = self.train_dataset.get_image_dict()
            self.train_sampler = ClassRandomSampler(
                self.train_batch_size, train_image_dict, train_no_samples
            )

            val_image_dict, val_no_samples = self.val_dataset.get_image_dict()
            self.val_sampler = ClassRandomSampler(
                self.val_batch_size, val_image_dict, val_no_samples
            )

            test_image_dict, test_no_samples = self.test_dataset.get_image_dict()
            self.test_sampler = ClassRandomSampler(
                self.val_batch_size, test_image_dict, test_no_samples
            )
        else:
            self.train_sampler, self.val_sampler, self.test_sampler = None, None, None

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            # batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            # shuffle=True,
            pin_memory=self.pin_memory,
            batch_sampler=self.train_sampler,
        )

    def train_dataloader_val_transformed(self) -> DataLoader:
        return DataLoader(
            self.train_dataset_val_transformed,
            # batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            # shuffle=True,
            pin_memory=self.pin_memory,
            batch_sampler=self.train_sampler,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            # batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            # shuffle=False,
            pin_memory=self.pin_memory,
            batch_sampler=self.val_sampler,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            # batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            # shuffle=False,
            pin_memory=self.pin_memory,
            batch_sampler=self.test_sampler,
        )


class BIOSCANVAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        use_triplet_sampling=True,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_triplet_sampling = use_triplet_sampling

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = BIOSCAN(
            self.data_dir,
            _set="train",
        )

        self.val_dataset = BIOSCAN(
            self.data_dir,
            _set="validation",
        )

        self.test_dataset = BIOSCAN(
            self.data_dir,
            _set="test",
        )

        if self.use_triplet_sampling:
            train_image_dict, train_no_samples = self.train_dataset.get_image_dict()
            self.train_sampler = ClassRandomSampler(
                self.train_batch_size, train_image_dict, train_no_samples
            )

            val_image_dict, val_no_samples = self.val_dataset.get_image_dict()
            self.val_sampler = ClassRandomSampler(
                self.val_batch_size, val_image_dict, val_no_samples
            )

            test_image_dict, test_no_samples = self.test_dataset.get_image_dict()
            self.test_sampler = ClassRandomSampler(
                self.val_batch_size, test_image_dict, test_no_samples
            )
        else:
            self.train_sampler, self.val_sampler, self.test_sampler = None, None, None

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            # batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            # shuffle=True,
            pin_memory=self.pin_memory,
            batch_sampler=self.train_sampler,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            # batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            # shuffle=False,
            pin_memory=self.pin_memory,
            batch_sampler=self.val_sampler,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            # batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            # shuffle=False,
            pin_memory=self.pin_memory,
            batch_sampler=self.test_sampler,
        )


class BIOSCANTreeVAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        use_triplet_sampling=True,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_triplet_sampling = use_triplet_sampling

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = BIOSCANTree(
            self.data_dir,
            _set="train",
        )

        self.val_dataset = BIOSCANTree(
            self.data_dir,
            _set="validation",
        )

        self.test_dataset = BIOSCANTree(
            self.data_dir,
            _set="test",
        )

        self.train_sampler, self.val_sampler, self.test_sampler = None, None, None

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            batch_sampler=self.train_sampler,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            batch_sampler=self.val_sampler,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            batch_sampler=self.test_sampler,
        )


# Add your custom dataset class here
class Rove(Dataset):
    def __init__(self, data_dir, split, transform):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        self.samples = self.get_samples()
        self.phylogeny = self.get_phylogeny()
        self.species_names = self.phylogeny.get_leaf_names()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = PIL.Image.open(sample["fp"])

        target = sample["fp"]
        species = sample["fp"].split("/")[-2].replace("_", " ").lower()

        if self.transform is not None:
            img = self.transform(img)

        return img, target, species

    def get_samples(self):
        samples = []
        path = os.path.join(self.data_dir, self.split)
        img_fps = glob(os.path.join(path, "*", "*.jpg"))

        for fp in img_fps:
            sample = {}
            sample["species"] = fp.split("/")[-2]
            sample["fp"] = fp
            samples += [sample]

        return samples

    def get_phylogeny(self):
        tree_path = os.path.join(self.data_dir, "phylogeny.nh")
        phylogeny = Tree(tree_path)
        return phylogeny

    def get_image_dict(self):
        """get_image_dict: Used for triplet sampling, gives indices of examples per class

        Returns:
            image_dict (dict):
        """
        image_dict = {}
        for i in range(len(self.samples)):
            sample = self.samples[i]
            if sample["species"] not in image_dict:
                image_dict[sample["species"]] = []
            image_dict[sample["species"]] += [i]
        return image_dict, len(self.samples)


def one_hot_encode_seq(seq):
    out = np.zeros([1, len(seq), 5])
    for i in range(len(seq)):
        if seq[i] == "A":
            out[0, i, 0] = 1
        elif seq[i] == "C":
            out[0, i, 1] = 1
        elif seq[i] == "G":
            out[0, i, 2] = 1
        elif seq[i] == "T":
            out[0, i, 3] = 1
        else:
            out[0, i, 4] = 1
    return out


# Add your custom dataset class here
class BIOSCAN(Dataset):
    def __init__(self, data_dir, _set):
        self.data_dir = data_dir
        self._set = _set

        self.samples = self.get_samples()
        self.phylogeny = self.get_phylogeny()
        self.species_names = sorted(self.phylogeny.get_leaf_names())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples.loc[idx]

        one_hot = sample["one_hot"]
        species = sample["species"]
        sampleid = sample["sampleid"]

        return one_hot, sampleid, species

    def get_samples(self):
        samples_fp = os.path.join(self.data_dir, "sequences.csv")
        df = pd.read_csv(samples_fp)
        df = df[df["set"] == self._set]

        df["one_hot"] = df["aligned_seq"].map(one_hot_encode_seq)
        df = df.reset_index(drop=True)

        return df

    def get_phylogeny(self):
        tree_path = os.path.join(self.data_dir, "phylogeny.nh")
        phylogeny = Tree(tree_path)
        return phylogeny

    def get_image_dict(self):
        """get_image_dict: Used for triplet sampling, gives indices of examples per class

        Returns:
            image_dict (dict):
        """
        image_dict = {}
        for i in range(len(self.samples)):
            sample = self.samples.loc[i]
            if sample["species"] not in image_dict:
                image_dict[sample["species"]] = []
            image_dict[sample["species"]] += [i]
        return image_dict, len(self.samples)


# Add your custom dataset class here
class BIOSCANTree(Dataset):
    def __init__(self, data_dir, _set):
        self.data_dir = data_dir
        self._set = _set

        self.samples = self.get_samples()
        self.genera = sorted(self.samples["genus"].drop_duplicates().values)
        self.phylogeny = self.get_phylogeny()
        self.species_names = sorted(self.phylogeny.get_leaf_names())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = (
            self.samples.groupby("genus")
            .sample(1, random_state=idx)
            .set_index("genus")
            .loc[self.genera]
        )
        one_hot = np.squeeze(np.array(list(sample["one_hot"].values)))
        one_hot = np.transpose(one_hot, (2, 1, 0))
        ids = (list(sample["sampleid"].values),)
        genera = list(self.genera)
        return (
            one_hot,
            ids,
            genera,
        )

    def get_samples(self):
        samples_fp = os.path.join(self.data_dir, "sequences.csv")
        df = pd.read_csv(samples_fp)
        df = df[df["medium_species"] == self._set]

        df["one_hot"] = df["aligned_seq"].map(one_hot_encode_seq)
        df = df.reset_index(drop=True)

        return df

    def get_phylogeny(self):
        tree_path = os.path.join(self.data_dir, "phylogeny.nh")
        phylogeny = Tree(tree_path)
        return phylogeny

    def get_image_dict(self):
        """get_image_dict: Used for triplet sampling, gives indices of examples per class

        Returns:
            image_dict (dict):
        """
        image_dict = {}
        for i in range(len(self.samples)):
            sample = self.samples.loc[i]
            if sample["species"] not in image_dict:
                image_dict[sample["species"]] = []
            image_dict[sample["species"]] += [i]
        return image_dict, len(self.samples)


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.

    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """

    def _check_integrity(self) -> bool:
        return True


class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """

    def __init__(self, data_path: str, split: str, transform: Callable, **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == ".jpg"])

        self.imgs = (
            imgs[: int(len(imgs) * 0.75)]
            if split == "train"
            else imgs[int(len(imgs) * 0.75) :]
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, 0.0  # dummy datat to prevent breaking


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        #       =========================  OxfordPets Dataset  =========================

        #         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                               transforms.CenterCrop(self.patch_size),
        # #                                               transforms.Resize(self.patch_size),
        #                                               transforms.ToTensor(),
        #                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        #         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                             transforms.CenterCrop(self.patch_size),
        # #                                             transforms.Resize(self.patch_size),
        #                                             transforms.ToTensor(),
        #                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        #         self.train_dataset = OxfordPets(
        #             self.data_dir,
        #             split='train',
        #             transform=train_transforms,
        #         )

        #         self.val_dataset = OxfordPets(
        #             self.data_dir,
        #             split='val',
        #             transform=val_transforms,
        #         )

        #       =========================  CelebA Dataset  =========================

        train_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(148),
                transforms.Resize(self.patch_size),
                transforms.ToTensor(),
            ]
        )

        val_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(148),
                transforms.Resize(self.patch_size),
                transforms.ToTensor(),
            ]
        )

        self.train_dataset = MyCelebA(
            self.data_dir,
            split="train",
            transform=train_transforms,
            download=False,
        )

        # Replace CelebA with your dataset
        self.val_dataset = MyCelebA(
            self.data_dir,
            split="test",
            transform=val_transforms,
            download=False,
        )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
