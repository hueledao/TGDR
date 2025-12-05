import os
from glob import glob
from scipy import linalg
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
import multiprocessing
from math import exp
# from torchvision.models import inception_v3
from Network.inception import InceptionV3
from Network.lpips_network import get_network, LinLayers
from Network.lpips_utils import get_state_dict

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1.0 / (torch.sqrt(mse) + 1e-6))


# class SSIM:
#     """Structure Similarity
#     img1, img2: [0, 255]"""

#     def __init__(self):
#         self.name = "SSIM"

#     # @staticmethod
#     def __call__(img1, img2):
#         if not img1.shape == img2.shape:
#             raise ValueError("Input images must have the same dimensions.")
#         if img1.ndim == 2:  # Grey or Y-channel image
#             return _ssim(img1, img2)
#         elif img1.ndim == 3:
#             if img1.shape[2] == 3:
#                 ssims = []
#                 for i in range(3):
#                     ssims.append(ssim(img1, img2))
#                 return np.array(ssims).mean()
#             elif img1.shape[2] == 1:
#                 return self._ssim(np.squeeze(img1), np.squeeze(img2))
#         else:
#             raise ValueError("Wrong input image dimensions.")

#     # @staticmethod
#     def _ssim(img1, img2):
#         C1 = (0.01 * 255) ** 2
#         C2 = (0.03 * 255) ** 2

#         img1 = img1.astype(np.float64)
#         img2 = img2.astype(np.float64)
#         kernel = cv2.getGaussianKernel(11, 1.5)
#         window = np.outer(kernel, kernel.transpose())

#         mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#         mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#         mu1_sq = mu1**2
#         mu2_sq = mu2**2
#         mu1_mu2 = mu1 * mu2
#         sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
#         sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
#         sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

#         ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#         return ssim_map.mean()
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)
class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).
    Arguments:
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type).to("cuda")

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list).to("cuda")
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        feat_x, feat_y = self.net(x), self.net(y)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        return torch.sum(torch.cat(res, 0)) / x.shape[0]

# def to_cuda(elements):
#     """
#     Transfers elements to cuda if GPU is available
#     Args:
#         elements: torch.tensor or torch.nn.module
#         --
#     Returns:
#         elements: same as input on GPU memory, if available
#     """
#     if torch.cuda.is_available():
#         return elements.cuda()
#     return elements
# class PartialInceptionNetwork(nn.Module):

#     def __init__(self, transform_input=True):
#         super().__init__()
#         self.inception_network = inception_v3(pretrained=True)
#         self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
#         self.transform_input = transform_input

#     def output_hook(self, module, input, output):
#         # N x 2048 x 8 x 8
#         self.mixed_7c_output = output

#     def forward(self, x):
#         """
#         Args:
#             x: shape (N, 3, 299, 299) dtype: torch.float32 in range 0-1
#         Returns:
#             inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float32
#         """
#         assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
#                                              ", but got {}".format(x.shape)
#         x = x * 2 -1 # Normalize to [-1, 1]

#         # Trigger output hook
#         self.inception_network(x)

#         # Output: N x 2048 x 1 x 1 
#         activations = self.mixed_7c_output
#         activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1,1))
#         activations = activations.view(x.shape[0], 2048)
#         return activations
# def get_activations(images, batch_size):
#     """
#     Calculates activations for last pool layer for all iamges
#     --
#         Images: torch.array shape: (N, 3, 299, 299), dtype: torch.float32
#         batch size: batch size used for inception network
#     --
#     Returns: np array shape: (N, 2048), dtype: np.float32
#     """
#     assert images.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
#                                               ", but got {}".format(images.shape)

#     num_images = images.shape[0]
#     inception_network = PartialInceptionNetwork()
#     inception_network = to_cuda(inception_network)
#     inception_network.eval()
#     n_batches = int(np.ceil(num_images  / batch_size))
#     inception_activations = np.zeros((num_images, 2048), dtype=np.float32)
#     for batch_idx in range(n_batches):
#         start_idx = batch_size * batch_idx
#         end_idx = batch_size * (batch_idx + 1)

#         ims = images[start_idx:end_idx]
#         ims = to_cuda(ims)
#         activations = inception_network(ims)
#         activations = activations.detach().cpu().numpy()
#         assert activations.shape == (ims.shape[0], 2048), "Expexted output shape to be: {}, but was: {}".format((ims.shape[0], 2048), activations.shape)
#         inception_activations[start_idx:end_idx, :] = activations
#     return inception_activations



# def calculate_activation_statistics(images, batch_size):
#     """Calculates the statistics used by FID
#     Args:
#         images: torch.tensor, shape: (N, 3, H, W), dtype: torch.float32 in range 0 - 1
#         batch_size: batch size to use to calculate inception scores
#     Returns:
#         mu:     mean over all activations from the last pool layer of the inception model
#         sigma:  covariance matrix over all activations from the last pool layer 
#                 of the inception model.
#     """
#     act = get_activations(images, batch_size)
#     mu = np.mean(act, axis=0)
#     sigma = np.cov(act, rowvar=False)
#     return mu, sigma


# # Modified from: https://github.com/bioinf-jku/TTUR/blob/master/fid.py
# def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
#     """Numpy implementation of the Frechet Distance.
#     The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
#     and X_2 ~ N(mu_2, C_2) is
#             d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
#     Stable version by Dougal J. Sutherland.
#     Params:
#     -- mu1 : Numpy array containing the activations of the pool_3 layer of the
#              inception net ( like returned by the function 'get_predictions')
#              for generated samples.
#     -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
#                on an representive data set.
#     -- sigma1: The covariance matrix over activations of the pool_3 layer for
#                generated samples.
#     -- sigma2: The covariance matrix over activations of the pool_3 layer,
#                precalcualted on an representive data set.
#     Returns:
#     --   : The Frechet Distance.
#     """

#     mu1 = np.atleast_1d(mu1)
#     mu2 = np.atleast_1d(mu2)

#     sigma1 = np.atleast_2d(sigma1)
#     sigma2 = np.atleast_2d(sigma2)

#     assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
#     assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

#     diff = mu1 - mu2
#     # product might be almost singular
#     covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
#     if not np.isfinite(covmean).all():
#         msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
#         warnings.warn(msg)
#         offset = np.eye(sigma1.shape[0]) * eps
#         covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

#     # numerical error might give slight imaginary component
#     if np.iscomplexobj(covmean):
#         if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
#             m = np.max(np.abs(covmean.imag))
#             raise ValueError("Imaginary component {}".format(m))
#         covmean = covmean.real

#     tr_covmean = np.trace(covmean)

#     return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# def preprocess_image(im):
#     """Resizes and shifts the dynamic range of image to 0-1
#     Args:
#         im: np.array, shape: (H, W, 3), dtype: float32 between 0-1 or np.uint8
#     Return:
#         im: torch.tensor, shape: (3, 299, 299), dtype: torch.float32 between 0-1
#     """
#     assert im.shape[2] == 3
#     assert len(im.shape) == 3
#     if im.dtype == np.uint8:
#         im = im.astype(np.float32) / 255
#     im = cv2.resize(im, (299, 299))
#     im = np.rollaxis(im, axis=2)
#     im = torch.from_numpy(im)
#     assert im.max() <= 1.0
#     assert im.min() >= 0.0
#     assert im.dtype == torch.float32
#     assert im.shape == (3, 299, 299)

#     return im


# def preprocess_images(images, use_multiprocessing):
#     """Resizes and shifts the dynamic range of image to 0-1
#     Args:
#         images: np.array, shape: (N, H, W, 3), dtype: float32 between 0-1 or np.uint8
#         use_multiprocessing: If multiprocessing should be used to pre-process the images
#     Return:
#         final_images: torch.tensor, shape: (N, 3, 299, 299), dtype: torch.float32 between 0-1
#     """
#     if use_multiprocessing:
#         with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
#             jobs = []
#             for im in images:
#                 job = pool.apply_async(preprocess_image, (im,))
#                 jobs.append(job)
#             final_images = torch.zeros(images.shape[0], 3, 299, 299)
#             for idx, job in enumerate(jobs):
#                 im = job.get()
#                 final_images[idx] = im#job.get()
#     else:
#         final_images = torch.stack([preprocess_image(im) for im in images], dim=0)
#     assert final_images.shape == (images.shape[0], 3, 299, 299)
#     assert final_images.max() <= 1.0
#     assert final_images.min() >= 0.0
#     assert final_images.dtype == torch.float32
#     return final_images


# def calculate_fid(images1, images2, use_multiprocessing, batch_size):
#     """ Calculate FID between images1 and images2
#     Args:
#         images1: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
#         images2: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
#         use_multiprocessing: If multiprocessing should be used to pre-process the images
#         batch size: batch size used for inception network
#     Returns:
#         FID (scalar)
#     """
#     images1 = preprocess_images(images1, use_multiprocessing)
#     images2 = preprocess_images(images2, use_multiprocessing)
#     mu1, sigma1 = calculate_activation_statistics(images1, batch_size)
#     mu2, sigma2 = calculate_activation_statistics(images2, batch_size)
#     fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
#     return fid


# def load_images(path):
#     """ Loads all .png or .jpg images from a given path
#     Warnings: Expects all images to be of same dtype and shape.
#     Args:
#         path: relative path to directory
#     Returns:
#         final_images: np.array of image dtype and shape.
#     """
#     image_paths = []
#     image_extensions = ["png", "jpg"]
#     for ext in image_extensions:
#         print("Looking for images in", os.path.join(path, "*.{}".format(ext)))
#         for impath in glob.glob(os.path.join(path, "*.{}".format(ext))):
#             image_paths.append(impath)
#     first_image = cv2.imread(image_paths[0])
#     W, H = first_image.shape[:2]
#     image_paths.sort()
#     image_paths = image_paths
#     final_images = np.zeros((len(image_paths), H, W, 3), dtype=first_image.dtype)
#     for idx, impath in enumerate(image_paths):
#         im = cv2.imread(impath)
#         im = im[:, :, ::-1] # Convert from BGR to RGB
#         assert im.dtype == final_images.dtype
#         final_images[idx] = im
#     return final_images

def get_activations(images, model, batch_size=64, dims=2048,
                    cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : the images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size depends
                     on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    d0 = images.shape[0]
    if batch_size > d0:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = d0

    n_batches = d0 // batch_size
    if d0 % batch_size != 0:
        n_batches += 1
    n_used_imgs = d0

    pred_arr = np.empty((n_used_imgs, dims))
    with torch.no_grad():
        for i in tqdm(range(n_batches)):
            if verbose:
                print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                      end='', flush=True)
            start = i * batch_size
            end = min(start + batch_size, d0)

            batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
            batch = Variable(batch)
            if cuda:
                batch = batch.cuda()

            pred = model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred_arr[start:end] = pred.cpu().data.numpy().reshape(end - start, -1)

        if verbose:
            print(' done')

    return pred_arr

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
def calculate_activation_statistics(images, model, batch_size=64,
                                    dims=2048, cuda=False, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(images, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma
def _compute_statistics_of_path(path, model, batch_size, dims, cuda):
    npz_file = os.path.join(path, 'statistics.npz')
    if os.path.exists(npz_file):
        f = np.load(npz_file)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        # path = pathlib.Path(path)
        # files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        files = list(glob(path + '/*.jpg')) + list(glob(path + '/*.png'))
        files = sorted(files, key=lambda x: x.split('/')[-1])
        
        imgs = []
        for fn in tqdm(files):
            imgs.append(cv2.imread(str(fn)).astype(np.float32)[:, :, ::-1])
        # print(len(imgs))
        imgs = np.array(imgs)
        # Bring images to shape (B, 3, H, W)
        imgs = imgs.transpose((0, 3, 1, 2))

        # Rescale images to be between 0 and 1
        imgs /= 255

        m, s = calculate_activation_statistics(imgs, model, batch_size, dims, cuda)
        # np.savez(npz_file, mu=m, sigma=s)

    return m, s
def calculate_fid_given_paths(paths, batch_size, cuda, dims):
    """Calculates the FID of two paths"""

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    print('calculate ',paths[0])
    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size, dims, cuda)
    # print('calculate path2 statistics...')
    print('calculate ',paths[1])
    m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size, dims, cuda)
    print('calculate frechet distance...')
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value
def get_inpainting_metrics(src, tgt):
    # input_paths = sorted(glob(src + '/*'), key=lambda x: x.split('/')[-1])
    # output_paths = sorted(glob(tgt + '/*'), key=lambda x: x.split('/')[-1])
    fid = calculate_fid_given_paths([src, tgt], batch_size=128, cuda=True, dims=2048)#64, 192, 768, 2048
    return fid