import lpips
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

import flip
from viewer.common import *

class flipError:
    BLACK = np.zeros((1,1,3), dtype=np.float32)

    def __init__(self) -> None:
        self.errorMap = flipError.BLACK
        self.meanError = 0
        self.gt = flipError.BLACK
        self.test = flipError.BLACK

    def evaluate(self, gt: np.ndarray, test: np.ndarray):
        '''input np.array (H, W, 3) linear image'''
        if gt is None or gt.shape != test.shape:
            self.errorMap = flipError.BLACK
            self.meanError = 0
            print("Not match")
            return
        
        self.gt = gt
        self.test = test
        
        #print("test:" ,self.test.max(),"gt:", self.gt.max())
        self.errorMap, self.meanError, parameters = flip.evaluate(gt, test, "HDR")

        return self.meanError
    def getGT(self, return_srgb = True):
        '''return float image'''
        if return_srgb:
            return linear_to_srgb(self.gt)
        else:
            return self.gt
    
    def getTest(self, return_srgb = True):
        '''return float image'''
        if return_srgb:
            return linear_to_srgb(self.test)
        else:
            return self.test
    
    def getErrorMap(self):
        '''return error map'''
        return self.errorMap
    

class Metric:
    def __init__(self) -> None:
        self.lpips_loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

    def L1(self, gt: np.ndarray, test: np.ndarray):
        '''L1 distance'''
        return np.linalg.norm(gt - test, ord=1)
    
    def L2(self, gt: np.ndarray, test: np.ndarray):
        '''L2 distance'''
        return np.linalg.norm(gt - test)
    
    def SSIM(self, batch_gt, batch_test):
        '''input np.array (B, H, W, 3) sRGB [0, 1]'''
        sumError = 0
        N = batch_gt.shape[0]

        for gt, test in zip(batch_gt, batch_test):
            sumError += SSIM(gt, test, data_range = 1.0, channel_axis=2)
        return sumError / N
    
    def PSNR(self, batch_gt, batch_test):
        '''input np.array (B, H, W, 3) sRGB [0, 1]'''
        sumError = 0
        N = batch_gt.shape[0]

        for gt, test in zip(batch_gt, batch_test):
            sumError += PSNR(gt, test, data_range = 1.0)

        return sumError / N
    
    def LPIPS(self, batch_gt, batch_test):
        '''input np.array (B, H, W, 3) sRGB [0, 1]'''
        # torch (B, 3, H, W)
        batch_gt = torch.tensor(batch_gt).permute(0, 3, 1, 2)
        batch_test = torch.tensor(batch_test).permute(0, 3, 1, 2)
        with torch.no_grad():
            batchError = self.lpips_loss_fn_alex(batch_gt, batch_test, normalize = True)
        return torch.mean(batchError).item()
    
    def evaluate(self, gt: np.ndarray, test: np.ndarray, input_linear = False, input_uint8 = False):
        '''
            input np.array
            - shape (B, H, W, 3) or (H, W, 3)
            - linear RGB or sRGB
            - float or uint8
        '''
        shape = gt.shape
        assert gt.shape == test.shape
        if input_linear:
            assert input_uint8 == False

        if len(shape) == 3:
            gt = np.expand_dims(gt, axis=0)
            test = np.expand_dims(test, axis=0)

        B, H, W, C = gt.shape

        if input_uint8:
            gt = uint8_to_float(gt)
            test = uint8_to_float(test)

        if input_linear:
            srgb_gt = linear_to_srgb(gt)
            srgb_test = linear_to_srgb(test)
            linear_gt = gt
            linear_test = test
        else:
            srgb_gt = gt
            srgb_test = test
            linear_gt = srgb_to_linear(gt)
            linear_test = srgb_to_linear(test)
        
        srgb_gt = srgb_gt.clip(0, 1)
        srgb_test = srgb_test.clip(0, 1)

        psnr = self.PSNR(srgb_gt, srgb_test)

        ssim = self.SSIM(srgb_gt, srgb_test)

        lpips = self.LPIPS(srgb_gt, srgb_test)

        linear_gt_1D = linear_gt.reshape(-1)
        linear_test_1D = linear_test.reshape(-1)
        L1 = self.L1(linear_gt_1D, linear_test_1D)
        L2 = self.L2(linear_gt_1D, linear_test_1D)

        return {
            "PSNR": psnr, 
            "SSIM": ssim, 
            "LPIPS": lpips, 
            "L1": L1, 
            "L2": L2
        }

