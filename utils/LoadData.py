import torch.utils.data as data
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch

from PIL import Image
import os.path as osp
import glob
import os
import numpy as np
from torchvision import utils
class Load_ImagesDataset(data.Dataset):
    def __init__(self, input_path, gt_path, mask_path, is_trained = True):
        self.in_path = input_path
        self.gt_path = gt_path
        self.mask_path = mask_path
        self.is_trained = is_trained
    def transform(self, input, gt, mask):
        output_size = (256, 256)
        hCROP = wCROP = 256
        shape = np.shape(input)
        resized_op = transforms.Resize(output_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        if shape[0] < hCROP or shape[1] < wCROP:
            if (shape[0] < hCROP):
                hPadding = hCROP
            else:
                hPadding = shape[0]
            if (shape[1] < wCROP):
                wPadding = hCROP
            else:
                wPadding = shape[1]

            padded_input = np.zeros((hPadding, wPadding, 3))
            padded_gt = np.zeros((hPadding, wPadding, 3))
            padded_mask = np.zeros((hPadding, wPadding, 3))

            padded_input[:shape[0], :shape[1]] = input
            padded_gt[:shape[0], :shape[1]] = gt

            padded_mask[:shape[0], :shape[1]] = mask

            padded_input = Image.fromarray(padded_input.astype('uint8'))
            padded_gt = Image.fromarray(padded_gt.astype('uint8'))
            padded_mask = Image.fromarray(padded_mask.astype('uint8'))

            
            input_crop = resized_op(padded_input)
            gt_crop = resized_op(padded_gt)
            mask_crop = resized_op(padded_mask)
        else:
            input_crop = resized_op(input)
            gt_crop = resized_op(gt)
            mask_crop = resized_op(mask)

        # Random Horizontal Flip
        if self.is_trained == True:
            p = np.random.randint(0, 2)
            if p == 1:
                input_crop = TF.hflip(input_crop)
                gt_crop = TF.hflip(gt_crop)
                mask_crop = TF.hflip(mask_crop)
        # To TENSOR
        input_crop = TF.to_tensor(input_crop)
        gt_crop = TF.to_tensor(gt_crop)
        mask_crop = TF.to_tensor(mask_crop)

        # Normalize
        Normalize_ = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        input_crop = Normalize_(input_crop)
        gt_crop = Normalize_(gt_crop)

        return input_crop, gt_crop, mask_crop
    def transform_cropped(self, input, gt, mask):

        output_size = (256, 256)
        hCROP = wCROP = 256
        shape = np.shape(input)
        if shape[0] < hCROP or shape[1] < wCROP:
            if (shape[0] < hCROP):
                hPadding = hCROP
            else:
                hPadding = shape[0]
            if (shape[1] < wCROP):
                wPadding = hCROP
            else:
                wPadding = shape[1]

            padded_input = np.zeros((hPadding, wPadding, 3))
            padded_gt = np.zeros((hPadding, wPadding, 3))
            padded_mask = np.zeros((hPadding, wPadding, 3))

            padded_input[:shape[0], :shape[1]] = input
            padded_gt[:shape[0], :shape[1]] = gt

            padded_mask[:shape[0], :shape[1]] = mask

            padded_input = Image.fromarray(padded_input.astype('uint8'))
            padded_gt = Image.fromarray(padded_gt.astype('uint8'))
            padded_mask = Image.fromarray(padded_mask.astype('uint8'))

            if self.is_trained == True:
                i, j, h, w = transforms.RandomCrop.get_params(padded_input, output_size)
                input_crop = TF.crop(padded_input, i, j, h, w)
                gt_crop = TF.crop(padded_gt, i, j, h, w)
                mask_crop = TF.crop(padded_mask, i, j, h, w)
            else:
                input_crop = TF.center_crop(padded_input, output_size=(256,256))
                gt_crop = TF.center_crop(padded_gt, output_size=(256,256))
                mask_crop = TF.center_crop(padded_mask, output_size=(256,256))


        else:
            if self.is_trained == True:
                i, j, h, w = transforms.RandomCrop.get_params(input, output_size)
                input_crop = TF.crop(input, i, j, h, w)
                gt_crop = TF.crop(gt, i, j, h, w)
                mask_crop = TF.crop(mask, i, j, h, w)
            else:
                input_crop = TF.center_crop(input, output_size=(256,256))
                gt_crop = TF.center_crop(gt, output_size=(256,256))
                mask_crop = TF.center_crop(mask, output_size=(256,256))

        # Random Horizontal Flip
        if self.is_trained == True:
            p = np.random.randint(0, 2)
            if p == 1:
                input_crop = TF.hflip(input_crop)
                gt_crop = TF.hflip(gt_crop)
                mask_crop = TF.hflip(mask_crop)

            # if p == 2:
            #     input_crop = TF.vflip(input_crop)
            #     gt_crop = TF.vflip(gt_crop)
            #     mask_crop = TF.vflip(mask_crop)

            # if p == 3:
            #     input_crop = TF.rotate(input_crop, 90)
            #     gt_crop = TF.rotate(gt_crop, 90)
            #     mask_crop = TF.rotate(mask_crop, 90)

            # if p == 4:
            #     input_crop = TF.rotate(input_crop, 180)
            #     gt_crop = TF.rotate(gt_crop, 180)
            #     mask_crop = TF.rotate(mask_crop, 180)

            # if p == 5:
            #     input_crop = TF.rotate(input_crop, -90)
            #     gt_crop = TF.rotate(gt_crop, -90)
            #     mask_crop = TF.rotate(mask_crop, -90)

        # To TENSOR
        input_crop = TF.to_tensor(input_crop)
        gt_crop = TF.to_tensor(gt_crop)
        mask_crop = TF.to_tensor(mask_crop)

        # Normalize
        Normalize_ = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        input_crop = Normalize_(input_crop)
        gt_crop = Normalize_(gt_crop)

        return input_crop, gt_crop, mask_crop

    def __getitem__(self, index):
        # Read RGB image
        input_img = Image.open(self.in_path[index])
        gt_img = Image.open(self.gt_path[index])
        mask = Image.open(self.mask_path[index])

        input_img = input_img.convert('RGB')
        gt_img = gt_img.convert('RGB')
        mask = mask.convert('RGB')

        input_img_crop, gt_img_crop, mask_crop = self.transform(input_img, gt_img, mask)

        return input_img_crop, gt_img_crop, mask_crop

    def __len__(self):
        return len(self.in_path)

class Load_ImagesDataset_4outputs(data.Dataset):
    def __init__(self, input_path, gt_path, mask_path, i52_path, is_trained = True):
        self.in_path = input_path
        self.gt_path = gt_path
        self.mask_path = mask_path
        self.i52_path = i52_path
        self.is_trained = is_trained
    def transform(self, input, gt, mask, i52):
        # output_size = (512, 512)    
        output_size = (256, 256)
        hCROP = wCROP = 256
        shape = np.shape(input)
        resized_op = transforms.Resize(output_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        if shape[0] < hCROP or shape[1] < wCROP:
            if (shape[0] < hCROP):
                hPadding = hCROP
            else:
                hPadding = shape[0]
            if (shape[1] < wCROP):
                wPadding = hCROP
            else:
                wPadding = shape[1]

            padded_input = np.zeros((hPadding, wPadding, 3))
            padded_gt = np.zeros((hPadding, wPadding, 3))
            padded_mask = np.zeros((hPadding, wPadding, 3))
            padded_i52 = np.zeros((hPadding, wPadding, 3))

            padded_input[:shape[0], :shape[1]] = input
            padded_gt[:shape[0], :shape[1]] = gt

            padded_mask[:shape[0], :shape[1]] = mask
            padded_i52[:shape[0], :shape[1]] = i52

            padded_input = Image.fromarray(padded_input.astype('uint8'))
            padded_gt = Image.fromarray(padded_gt.astype('uint8'))
            padded_mask = Image.fromarray(padded_mask.astype('uint8'))
            padded_i52 = Image.fromarray(padded_i52.astype('uint8'))
            
            input_crop = resized_op(padded_input)
            gt_crop = resized_op(padded_gt)
            mask_crop = resized_op(padded_mask)
            i52_crop = resized_op(padded_i52)
        else:
            input_crop = resized_op(input)
            gt_crop = resized_op(gt)
            mask_crop = resized_op(mask)
            i52_crop = resized_op(i52)

        # Random Horizontal Flip
        if self.is_trained == True:
            p = np.random.randint(0, 2)
            if p == 1:
                input_crop = TF.hflip(input_crop)
                gt_crop = TF.hflip(gt_crop)
                mask_crop = TF.hflip(mask_crop)
                i52_crop = TF.hflip(i52_crop)
        # To TENSOR
        input_crop = TF.to_tensor(input_crop)
        gt_crop = TF.to_tensor(gt_crop)
        mask_crop = TF.to_tensor(mask_crop)
        i52_crop = TF.to_tensor(i52_crop)
        # Normalize
        Normalize_ = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        input_crop = Normalize_(input_crop)
        gt_crop = Normalize_(gt_crop)
        i52_crop = Normalize_(i52_crop)

        return input_crop, gt_crop, mask_crop, i52_crop

    def __getitem__(self, index):
        # Read RGB image
        input_img = Image.open(self.in_path[index])
        gt_img = Image.open(self.gt_path[index])
        mask = Image.open(self.mask_path[index])
        i52 = Image.open(self.i52_path[index])

        input_img = input_img.convert('RGB')
        gt_img = gt_img.convert('RGB')
        mask = mask.convert('RGB')
        i52_img = i52.convert('RGB')

        input_img_crop, gt_img_crop, mask_crop, i52_img_crop = self.transform(input_img, gt_img, mask, i52_img)

        return input_img_crop, gt_img_crop, mask_crop, i52_img_crop

    def __len__(self):
        return len(self.in_path)

class Load_TestImagesDataset(data.Dataset):
    def __init__(self, input_path, gt_path, mask_path, is_trained = True):
        self.in_path = input_path
        self.gt_path = gt_path
        self.mask_path = mask_path
        self.is_trained = is_trained
    def transform(self, input, gt, mask):#resize
        output_size = (256, 256)
        resized_op = transforms.Resize(output_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
       
        input_crop = resized_op(input)
        gt_crop = resized_op(gt)
        mask_crop = resized_op(mask)

        
        # To TENSOR
        input_crop = TF.to_tensor(input_crop)
        gt_crop = TF.to_tensor(gt_crop)
        mask_crop = TF.to_tensor(mask_crop)
        # Normalize
        Normalize_ = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        input_crop = Normalize_(input_crop)
        gt_crop = Normalize_(gt_crop)

        # mask_crop[mask_crop < 0.5] = 0
        # mask_crop[mask_crop >=0.5] = 1

        return input_crop, gt_crop, mask_crop

    def transform_crop(self, input, gt, mask):#crop

        # To TENSOR
        input_crop = TF.to_tensor(input)
        gt_crop = TF.to_tensor(gt)
        mask_crop = TF.to_tensor(mask)

        # Normalize
        Normalize_ = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        input_crop = Normalize_(input_crop)
        gt_crop = Normalize_(gt_crop)

        return input_crop, gt_crop, mask_crop

    def __getitem__(self, index):
        # Read RGB image
        input_img = Image.open(self.in_path[index])
        gt_img = Image.open(self.gt_path[index])
        mask = Image.open(self.mask_path[index])

        input_img = input_img.convert('RGB')
        gt_img = gt_img.convert('RGB')
        mask = mask.convert('RGB')



        input_img_crop, gt_img_crop, mask_crop = self.transform(input_img, gt_img, mask)

        return input_img_crop, gt_img_crop, mask_crop

    def __len__(self):
        return len(self.in_path)
