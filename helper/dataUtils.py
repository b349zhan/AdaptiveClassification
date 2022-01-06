class UnLabel_Dataset(Dataset):
    """
    UnLabel_Dataset: define a Dataset for unlabeled data
    """
    def __init__(self, imgs, ground_truth, transform):
        """
        input -> imgs: all images in the dataset
                 ground_truth: the ground truth of each image
                 transform: transform, could be None
        """
        self.imgs = imgs
        self.ground_truth = ground_truth
        self.transform = transform
    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        tar = self.ground_truth[idx]
        if self.transform:
            img = self.transform(img)
        return img[None:], tar
    
class get_img():
    def __init__(self, classes=[]):
        """
        input -> classes: classes will be replaced.
        """
        self.classes = classes
    def __call__(self, dataset):
        tensor_targets = torch.tensor(dataset.targets)
        semasks = torch.any(torch.stack( [torch.eq(tensor_targets, aelem).logical_or_(torch.eq(tensor_targets, aelem)) for aelem in self.classes], dim=0), dim = 0)
        imgs = dataset.data[semasks]
        tars = torch.tensor(dataset.targets)[semasks.tolist()].tolist()
        return imgs, tars
    
class replace_label():
    def __init__(self, labels=[], tar_label=7):
        """
        input -> labels: the labels(classed) will be replaced.
                 tar_label: the labels in 'labels' will be replaced by tar_label in __call__()
        """
        super(replace_label, self).__init__()
        if len(labels)!=NUM_UNTARGET_CLASS:
            warnings.warn('Length of labels neq to NUM_UNTARGET_CLASS!')
        self.labels = labels
        self.tar_label = tar_label
    def __call__(self, labels):
        labels_cpy = torch.clone(labels)
        semasks = torch.any(torch.stack([torch.eq(labels, aelem).logical_or_(torch.eq(labels, aelem)) for aelem in self.labels], dim=0), dim = 0)
        labels_cpy[semasks] = self.tar_label
        return labels_cpy
    
