import torchvision.transforms as t
from torch.utils import data

from .mpii import MPIIDataset
from .sampler import DistributedSampler


def get_mpii_loader(batch_size, is_train=True):
    transform = t.Compose([t.ToTensor(), t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = MPIIDataset(is_train=is_train, transform=transform)
    dist_sampler = DistributedSampler(dataset, shuffle=is_train)
    batch_sampler = data.BatchSampler(dist_sampler, batch_size, drop_last=not is_train)
    loader = data.DataLoader(dataset=dataset, batch_sampler=batch_sampler, num_workers=8, pin_memory=True)
    return dataset, loader
