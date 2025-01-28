"""
MIL_structures data flow Structure   Script  ver： Nov 26th 14:00
fixme: find out using the ratio is better or worse?
"""

import sys
sys.path.append('../')

import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from utils.dual_augmentation import Four_step_dual_augmentation
from utils.tools import to_2tuple, find_all_files


def shuffle_in_group(patch_label, device='cpu'):
    """
    FYD 的 分组打乱函数，分两步：1、同类别打乱（含背景）；2、重组后图片内部打乱
    :param label: 每个patch的标签，flatten后的
    :param group_num: 组别数，cls+1
    :param batch_size: 重组后的图片数量
    :return: shuffle_idx，打乱的索引
    """

    batch_size, num_patches, cls = patch_label.shape
    group_num = cls + 1
    # padding an empty label before all the CLS, so we can identify the background-only patches
    padding_label = torch.zeros([batch_size, num_patches, 1],
                                requires_grad=False, device=device)  # bgr_label: (B, P, 1)
    # new_patch_labels: (B, P, 1 + cls), 1 is the background patches
    new_patch_label = torch.cat((padding_label, patch_label), dim=-1)
    # obtain the class-index of each patch by max of CLS-mask-ratio
    _, label = torch.max(new_patch_label, dim=-1)
    # idx.shape = [B, P], each value is the corresponding cls-index of each patch
    label = torch.flatten(label)  # [B*P]

    # generate a randomized group order
    ran_group = np.random.permutation(group_num)
    # create the shuffle_idx init with 0
    shuffle_idx = torch.zeros([label.size()[0], 1], dtype=torch.int64, requires_grad=False, device=device)
    for group in ran_group:
        # Take the groups with a same group index, then shuffle each of their patches' index
        idx_ori = torch.nonzero((label == group)).to(device)
        idx = torch.randperm(idx_ori.nelement())
        idx_tar = idx_ori.view(-1)[idx].view(idx_ori.size())
        shuffle_idx = shuffle_idx.scatter(0, idx_ori, idx_tar)

    # shuffle each image patches
    patch_num = int(label.size()[0] / batch_size)
    shuffle_idx_split = torch.chunk(shuffle_idx, batch_size, dim=0)  # separate the patches
    for i in range(batch_size):
        idx = torch.randperm(shuffle_idx_split[i].nelement())
        shuffle_idx[i * patch_num:(i+1) * patch_num] = \
            shuffle_idx_split[i].view(-1)[idx].view(shuffle_idx_split[i].size())
    shuffle_idx = torch.flatten(shuffle_idx)  # return the shuffle indices for the B*P sequence
    assert label.size() == shuffle_idx.size()
    return shuffle_idx


class to_patch:
    """
    Split a image into patches, each patch with the size of patch_size
    """

    def __init__(self, patch_size=(16, 16)):
        patch_size = to_2tuple(patch_size)
        self.patch_h = patch_size[0]
        self.patch_w = patch_size[1]

    def __call__(self, x):
        c, h, w = x.shape

        assert h // self.patch_h == h / self.patch_h and w // self.patch_w == w / self.patch_w

        num_patches = (h // self.patch_h) * (w // self.patch_w)

        # patch encoding
        # (c, h, w)
        # -> (c, h // self.patch_h, self.patch_h, w // self.patch_w, self.patch_w)
        # -> (h // self.patch_h, w // self.patch_w, self.patch_h, self.patch_w, c)
        # -> (n_patches, patch_size^2*c)
        patches = x.view(
            c,
            h // self.patch_h,
            self.patch_h,
            w // self.patch_w,
            self.patch_w).permute(1, 3, 2, 4, 0).reshape(num_patches, -1)  # it can also used in transformer Encoding

        # patch split
        # (n_patches, patch_size^2*c)
        # -> (num_patches, self.patch_h, self.patch_w, c)
        # -> (num_patches, c, self.patch_h, self.patch_w)
        patches = patches.view(num_patches,
                               self.patch_h,
                               self.patch_w,
                               c).permute(0, 3, 1, 2)

        '''
        # check
        for i in range(len(patches)):
            recons_img = ToPILImage()(patches[i])
            recons_img.save(os.path.join('./patch_play', 'recons_target'+str(i)+'.jpg'))
        # patch compose to image
        # (num_patches, c, self.patch_h, self.patch_w)
        # -> (h // self.patch_h, w // self.patch_w, c, self.patch_h, self.patch_w)
        # -> (c, h // self.patch_h, self.patch_h, w // self.patch_w, self.patch_w)
        # -> (c, h, w)
        patches = patches.view(h // self.patch_h,
                               w // self.patch_w,
                               c,
                               self.patch_h,
                               self.patch_w).permute(2, 0, 3, 1, 4).reshape(c, h, w)
        '''

        '''
        # visual check
        # reshape
        composed_patches = patches.view(h // self.patch_h,
                                        w // self.patch_w,
                                        c,
                                        self.patch_h,
                                        self.patch_w).permute(2, 0, 3, 1, 4).reshape(c, h, w)
        # view pic
        from torchvision.transforms import ToPILImage
        composed_img = ToPILImage()(bag_image[0])  # transform tensor image to PIL image
        composed_img.save(os.path.join('./', 'composed_img.jpg'))
        '''

        return patches


# calculate the soft_label of classed masked-area
class soft_label_calculation:
    """
    calculate the masked regions in each patch, remain their classification information in the dimmention
    (num_patches, c, self.patch_h, self.patch_w) -> (numpatches, cls)  (cls)=[0,mask_area,0,....]
    input:
    patch_mask: (num_patches, c, self.patch_h, self.patch_w)  binary mask (the values are equal in all 3 channels)
    bag_label: (cls)  one-hot label  [0., ..., 1., 0., ....]
    output:
    soft_labels: (numpatches, cls)  classed masked-area calculation
    """

    def __init__(self):
        pass

    def __call__(self, patch_mask, bag_label):
        # patch_mask: binary mask (the values are equal in all 3 channels)
        # (numpatches, 3, patch_h, patch_w) -> (3, numpatches, patch_h, patch_w)
        patch_mask = patch_mask.transpose(0, 1)[0]  # (numpatches, patch_h, patch_w)
        numpatches, patch_h, patch_w = patch_mask.shape

        # -> (numpatches, 1)  number of masked pixels on each patch
        masked_pixels = patch_mask.sum(axis=[1, 2]).unsqueeze(1)  # (numpatches, 1): MPi

        # -> (numpatches, cls)  Obtain one-hot mask pixels (MP) on each patch
        soft_labels = masked_pixels * bag_label
        # bag_label: one-hot tensor([0., ..., 1., 0., ....]) len = cls  torch.Size([cls])
        # on each patch: tensor([0., ..., MPi, 0., ....]) len = cls  torch.Size([cls])

        # Mask ratio (MR) fixme: find out using the ratio is better or worse?
        soft_labels = soft_labels / (patch_h * patch_w)

        # print(soft_labels.shape)  # torch.Size([numpatches, cls])
        # print(bag_label)  # tensor([0., ..., 1., 0., ....]) len = cls
        # print(soft_labels[0])  # tensor([0., ..., MRi, 0., ....]) len = cls

        return soft_labels


class MILDataset(torch.utils.data.Dataset):
    def __init__(self, input_root, mode="train", data_augmentation_mode=0,
                 edge_size=384, patch_size=32, suffix='.jpg'):

        super().__init__()

        patch_size = to_2tuple(patch_size)

        self.data_root = os.path.join(input_root, 'data')

        # get class_names
        class_names = [filename for filename in os.listdir(self.data_root)
                       if os.path.isdir(os.path.join(self.data_root, filename))]
        class_names.sort()
        self.class_names = class_names
        cls_idxs = [i for i in range(len(class_names))]
        self.class_id_dict = dict(zip(class_names, cls_idxs))

        self.mode = mode

        self.input_ids = sorted(find_all_files(self.data_root, suffix=suffix))

        # get data augmentation and transform
        Dual_transform, DualImage, train_domain_transform, transform = \
            Four_step_dual_augmentation(data_augmentation_mode, edge_size)

        # apply the on-time synchornized transform on image and mask together
        self.Dual_transform = Dual_transform
        # val & test use DualImage to convert PIL Image
        self.DualImage = DualImage
        # ColorJitter for image only
        self.train_domain_transform = train_domain_transform
        # lastly, the synchornized separate transform
        self.transform = transforms.Compose([
            transform,  # Operation on PIL images and ToTensor
            transforms.Lambda(to_patch(patch_size)),  # Patch split operation
        ])

        self.soft_label_transform = soft_label_calculation()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # get data path
        imageName = self.input_ids[idx]
        # get label id
        label = self.class_id_dict[os.path.split(os.path.split(imageName)[0])[-1]]

        # get one-hot bag_label
        bag_label = torch.zeros(len(self.class_names))  # [k] k is the total cls num
        bag_label[label] = 1  # set one to 1, others 0

        # get bag label (each label is a catalog idx, long-int encoding)
        label = torch.ones(1) * label

        # get data and mask
        # CV2 0-255 hwc，in totensor step it will be transformed to chw.  ps:PIL(0-1 hwc)
        image = np.array(cv2.imread(imageName), dtype=np.float32)

        # maskname is replace the last 'data' by 'mask'
        maskName = "data".join(imageName.split("data")[:-1]) + 'mask' + "".join(imageName.split("data")[-1:])
        # mask: 0/255 cv2 hwc
        mask = np.array(cv2.imread(maskName))

        if self.mode == "train":
            # notice the difference in different image format，pil numpy value is 1/255 wrt cv2 numpy
            # 0/255 mask are transformed to binary mask
            # rotate + filp
            image, mask = self.Dual_transform(image, mask)
            # image color jitter shifting
            image = self.train_domain_transform(image)

            # crop + resize + patch split
            patch_image = self.transform(image)
            patch_mask = self.transform(mask)  # TODO: maybe for future usage of segmentation tasks

            # soft_label_calculation
            patch_labels = self.soft_label_transform(patch_mask, bag_label)

            return patch_image, patch_labels, label

        else:  # Val and Test datasets
            # 0/255 mask -> binary mask
            image, mask = self.DualImage(image, mask)
            # notice the difference in different image format，pil numpy is 1/255 in cv2 numpy

            # crop + resize + patch split
            patch_image = self.transform(image)
            patch_mask = self.transform(mask)

            # soft_label_calculation
            patch_labels = self.soft_label_transform(patch_mask, bag_label)

            return patch_image, patch_labels, label


class shuffle_and_compose:
    def __init__(self, edge_size=(384, 384), patch_size=(16, 16), shuffle=True, same_cls_shuffle=False, device='cpu'):
        edge_size = to_2tuple(edge_size)
        patch_size = to_2tuple(patch_size)

        # MIL_structures use shuffle patch, CLS use normal patch
        self.shuffle = shuffle
        self.h = edge_size[0]
        self.w = edge_size[1]
        self.patch_h = patch_size[0]
        self.patch_w = patch_size[1]

        self.same_cls_shuffle = same_cls_shuffle

        self.device = device

    def __call__(self, patch_image, patch_labels):
        """
        input:
        batch of patches: (B, num_patches, C, patch_h, patch_w)
        labels of patches: (B, num_patches, cls)  (cls)=[0,mask_area,0,....]
        batch_cls_long_int: [B,C]
        output:
        composed patches: (B, C, h, w)
        composed labels: (B, cls+1) [B, mask_area, type_cls_area..] a composed label with a conjugate design
        """
        B, num_patches, C, patch_h, patch_w = patch_image.shape
        B_p, num_patches_p, CLS = patch_labels.shape

        assert (self.h // patch_h * self.w // patch_w) == num_patches \
               and self.patch_h == patch_h and self.patch_w == patch_w

        assert B == B_p and num_patches == num_patches_p  # check the size of label matchs the input data

        # Shuffle
        if self.shuffle:
            if self.same_cls_shuffle:
                # shuffle in group
                shuffle_indices = shuffle_in_group(patch_labels, device=self.device)
            else:
                # [B*P] indices to shuffle the [B, P, ...] (among the batch and patches)
                shuffle_indices = torch.randperm(B * num_patches, device=self.device)

            composed_patches = patch_image.view(B * num_patches, C, patch_h, patch_w)[shuffle_indices] \
                .view(B, num_patches, C, patch_h, patch_w)

            composed_labels = patch_labels.view(B * num_patches, CLS)[shuffle_indices] \
                .view(B, num_patches, CLS)

        else:
            composed_patches, composed_labels = patch_image, patch_labels

        # Compose: a batch of patches compose to a batch of images
        # (B, num_patches, C, self.patch_h, self.patch_w)
        # view -> (B, h // self.patch_h, w // self.patch_w, C, self.patch_h, self.patch_w)
        # permute -> (B, C, h // self.patch_h, self.patch_h, w // self.patch_w, self.patch_w)
        # reshape -> (B, C, h, w)
        bag_image = composed_patches.view(B,
                                          self.h // patch_h,
                                          self.w // patch_w,
                                          C,
                                          patch_h,
                                          patch_w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, self.h, self.w)

        # calculate a composed label with a conjugate design
        # calculate the sum of all mask ratio in each cls, using one-hot anno
        composed_labels = composed_labels.sum(axis=1)  # (B, CLS)
        # all mask ratio sum
        total_labels = composed_labels.sum(axis=1).unsqueeze(1)  # (B, 1)
        bag_labels = torch.cat((total_labels, composed_labels), 1)  # (B, 1+CLS)
        # fixme total_labels 是总masked的标注，严谨的话应该按照总最大mask数字相减做取反

        # average mask ratio in each cls
        # bag_labels /= num_patches  fixme with or without this?
        '''
        # view pic
        from torchvision.transforms import ToPILImage
        composed_img = ToPILImage()(bag_image[0].to(cpu))  # transform tensor image to PIL image
        composed_img.save(os.path.join('./', 'composed_img.jpg'))
        '''

        return bag_image, bag_labels


# non-shuffle step data distributer
class non_shuffle_distributer:
    def __init__(self, edge_size=(384, 384), patch_size=(16, 16), device='cpu'):
        edge_size = to_2tuple(edge_size)
        patch_size = to_2tuple(patch_size)

        self.compose = shuffle_and_compose(edge_size, patch_size, shuffle=False, device=device)

    def __call__(self, patch_image, patch_labels, label):
        bag_image, bag_labels = self.compose(patch_image, patch_labels)
        # change labels of a batch: [B，1] -> [B]
        label = label.squeeze(-1).long()  # flatten batch label for cross-entropy loss
        return bag_image, bag_labels, label


# shuffle step data distributer
class shuffle_distributer:
    def __init__(self, edge_size=(384, 384), patch_size=(16, 16), same_cls_shuffle=False, device='cpu'):
        edge_size = to_2tuple(edge_size)
        patch_size = to_2tuple(patch_size)

        self.shuffle_compose = shuffle_and_compose(edge_size, patch_size, shuffle=True,
                                                   same_cls_shuffle=same_cls_shuffle, device=device)

    def __call__(self, patch_image, patch_labels):
        bag_image, bag_labels = self.shuffle_compose(patch_image, patch_labels)
        return bag_image, bag_labels


if __name__ == '__main__':
    from torchvision.transforms import ToPILImage

    total_path = r'E:/Study/code/'
    dataset_name = 'Colored_Dataset_MIL'
    dataroot = total_path + 'datasets/Colored_Dataset_MIL'
    res_path = total_path + 'same_cls_shuffle/' + dataset_name
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    # device enviorment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    phase = 'train'
    num_workers = 0
    batch_size = 4
    patch_size = 128
    edge_size = 384
    data_augmentation_mode = 3

    shuffle_patch_distributer = shuffle_distributer(edge_size, patch_size, same_cls_shuffle=True, device=device)

    # 2 dataset obj is prepared here and combine together
    dataset = MILDataset(os.path.join(dataroot, phase), mode=phase, data_augmentation_mode=data_augmentation_mode,
                         suffix='.jpg', edge_size=edge_size, patch_size=patch_size)

    class_names = dataset.class_names

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Iterate over data.
    for patch_image, patch_labels, labels in dataloader:  # use different dataloder in different phase

        # get datas
        patch_image = patch_image.to(device)
        patch_labels = patch_labels.to(device)
        labels = labels.to(device)

        # get bag data from the distributers
        MIL_bag_image, MIL_bag_labels = shuffle_patch_distributer(patch_image, patch_labels)

        for batch_idx in range(MIL_bag_image.shape[0]):
            composed_img = ToPILImage()(MIL_bag_image[batch_idx].to('cpu'))  # transform tensor image to PIL image
            composed_img.save(os.path.join(res_path, 'composed_img_' + str(batch_idx) + '.jpg'))
            print(batch_idx)