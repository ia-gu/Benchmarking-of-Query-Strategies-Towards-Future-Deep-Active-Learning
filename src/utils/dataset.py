import torch
from torchvision import transforms, datasets
from torch.utils.data import ConcatDataset

def get_data(data_cfg):
    model_name = 'resnet'
    channels = 1
    # download_path needs to be modified
    # *_data_path also needs to be modified
    if data_cfg.name == 'CIFAR10':
        model_name = 'modified_resnet'
        channels = 3
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        download_path = './downloaded_data/'
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        train_dataset = datasets.CIFAR10(download_path, download=True, train=True, transform=train_transform, target_transform=torch.tensor)
        test_dataset = datasets.CIFAR10(download_path, download=True, train=False, transform=test_transform, target_transform=torch.tensor)

    elif data_cfg.name == 'EuroSAT':
        channels = 3
        classes = ('AnnualCrop', 'Forest', 'Herbaceous', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake')
        train_data_path = '/data/dataset/eurosat/train'
        test_data_path = '/data/dataset/eurosat/test'
        train_transform = transforms.Compose([transforms.RandomVerticalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.ImageFolder(root=train_data_path, transform=train_transform)
        test_dataset = datasets.ImageFolder(root=test_data_path, transform=test_transform)

    elif data_cfg.name == 'BrainTumor':
        classes = ('Glioma', 'Meningioma', 'Pituitary')
        train_data_path = '/data/dataset/brain_tumor/BrainTumor'
        test_data_path = '/data/dataset/brain_tumor/BrainTumor/5'
        train_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.ImageFolder(root=train_data_path+'/'+str(1), transform=train_transform)
        for i in range(2, 5):
            train_dataset = ConcatDataset([train_dataset, (datasets.ImageFolder(root=train_data_path+'/'+str(i), transform=train_transform))])
        test_dataset = datasets.ImageFolder(root=test_data_path, transform=test_transform)

    elif data_cfg.name == 'OCT':
        classes = ('CNV', 'DME', 'DRUSEN', 'NORMAL')
        train_data_path = '/data/dataset/oct_modified/train'
        test_data_path = '/data/dataset/oct_modified/test'
        train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.ImageFolder(root=train_data_path, transform=train_transform)
        test_dataset = datasets.ImageFolder(root=test_data_path, transform=test_transform)
    
    elif data_cfg.name == 'GAPs':
        classes = ('AppliedPatch', 'Crack', 'InlaidPatch', 'IntactRoad', 'OpenJoint', 'Pothol')
        train_data_path = '/data/dataset/gaps/train'
        test_data_path = '/data/dataset/gaps/test'
        train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.ImageFolder(root=train_data_path, transform=train_transform)
        test_dataset = datasets.ImageFolder(root=test_data_path, transform=test_transform)

    elif data_cfg.name == 'KSDD2':
        classes = ('NG', 'OK')
        train_data_path = '/data/dataset/ksdd2/train'
        test_data_path = '/data/dataset/ksdd2/test'
        train_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.ImageFolder(root=train_data_path, transform=train_transform)
        test_dataset = datasets.ImageFolder(root=test_data_path, transform=test_transform)

    elif data_cfg.name == 'ImageNet':
        channels = 3
        classes = list()
        for i in range(1000):
            classes.append(str(i))
        train_data_path = '/imagenet_dataset/imagenet_2012/ILSVRC2012_img_train'
        test_data_path = '/imagenet_dataset/imagenet_2012/ILSVRC2012_img_val_for_ImageFolder'
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.2, 1.)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        train_dataset = datasets.ImageFolder(root=train_data_path, transform=train_transform)
        test_dataset = datasets.ImageFolder(root=test_data_path, transform=test_transform)
    
    else:
        raise IOError('Enter Valid dataset!')

    return train_dataset, test_dataset, model_name, channels, classes