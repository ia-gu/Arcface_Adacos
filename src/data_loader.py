from torch import utils
from torchvision import datasets, transforms

class DataLoader:
    def __init__(self, data_path, batch_size, img_size, train_ratio, num_workers, pin_memory, worker_init_fn, generator):
        self.dataset_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.worker_init_fn = worker_init_fn
        self.generator = generator

        self.train_transform = transforms.Compose([transforms.Resize(self.img_size),
                                                    transforms.RandomHorizontalFlip(),
                                                    # transforms.RandomRotation(3),
                                                    transforms.RandomAffine((-3,3),(0.02, 0.02)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        self.dataloaders = self.import_image(batch_size=batch_size, train_ratio=train_ratio, num_workers=num_workers, pin_memory=True)


    def import_image(self, batch_size, train_ratio=0.8, num_workers=0, pin_memory=True):
        data = datasets.ImageFolder(root=self.dataset_path, transform=self.train_transform)

        train_size = int(train_ratio * len(data))
        val_size = len(data) - train_size

        train_data, val_data = utils.data.random_split(data, [train_size, val_size])

        train_loader = utils.data.DataLoader(train_data,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                pin_memory=pin_memory,
                                                worker_init_fn=self.worker_init_fn,
                                                generator=self.generator)

        val_loader = utils.data.DataLoader(val_data,
                                                batch_size=self.batch_size,
                                                shuffle=False,
                                                num_workers=num_workers,
                                                pin_memory=pin_memory,
                                                worker_init_fn=self.worker_init_fn,
                                                generator=self.generator)
        dataloaders = {'train':train_loader, 'val':val_loader}
        return dataloaders
