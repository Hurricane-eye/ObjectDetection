import os
import torch
import torch.utils.data
import torchvision.transforms as T
from PIL import Image


class VisDroneDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.images = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(self.root, "annotations"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.images[idx])
        img = Image.open(img_path).convert("RGB")
        annotations_path = os.path.join(self.root, "annotations", self.annotations[idx])
        boxes = []
        labels = []
        with open(annotations_path, 'r') as file:
            try:
                for annotation in file:
                    annotation = list(map(int, annotation.rstrip("\n").split(',')))
                    boxes.append([annotation[0], annotation[1],
                                  annotation[0] + annotation[2],
                                  annotation[1] + annotation[3]])
                    labels.append(annotation[5])

            finally:
                file.close()
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area =(boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((len(labels), ), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": image_id,
                  "area": area, "iscrowd": iscrowd}

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images)


def get_transform(train):
    transforms = []
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


class VisDroneTestSet(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.images = list(sorted(os.listdir(os.path.join(root, "images"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.images[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        return len(self.images)

