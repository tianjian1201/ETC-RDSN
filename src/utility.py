import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue
from skimage.measure import compare_psnr as psnr, compare_ssim as ssim
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import cv2

import pdb


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            #print("debug:"+ str(self.dir))
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)


        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    
    if hr.nelement() == 1: return 0
    # diff = (sr - hr)/rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if sr.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = sr.new_tensor(gray_coeffs).view(1, 3, 1, 1)/256
            sr = sr.mul(convert).sum(dim=1)
        if hr.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = hr.new_tensor(gray_coeffs).view(1, 3, 1, 1)/256
            hr = hr.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    
    sr = sr[..., shave:-shave, shave:-shave]
    hr = hr[..., shave:-shave, shave:-shave]
    valid = (sr-hr)/rgb_range
    mse = valid.pow(2).mean()
    return -10 * math.log10(mse)


def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_last_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

def transfer_model(pretrained_file, model):
    pretrained_dict = torch.load(pretrained_file)
    # pretrained_dict = model.load_state_dict(torch.load(pretrained_file, map_location=lambda storage, loc: storage))
    model_dict = model.state_dict()
    pretrained_dict = transfer_state_dict(pretrained_dict, model_dict)
    model_dict.update(pretrained_dict)  # 更新(合并)模型的参数
    model.load_state_dict(model_dict)
    torch.save(model.state_dict(), "../experiment/RDCN_v4_x4/model/model_pre.pt")


def transfer_state_dict(pretrained_dict, model_dict):
    # state_dict2 = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            # state_dict.setdefault(k, v)
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict

# number of parameter
def getParameter():
    from model import rdn as rdn
    from option import args

    model = rdn.RDN(args)
    total =sum([param.nelement() for param in model.parameters()])
    print("Number of params: %.2fM"%(total/1e6))

def getResult(p1, p2):
    # p1: SR图像路径
    # p2: HR图像路径
    gray_coeffs = [65.738, 129.057, 25.064]

    for dataset in ['Set5', 'Set14', 'BSDS100', 'Urban100']:
    # for dataset in ['Set5']:
        psnrs = []
        ssims = []
        if dataset == 'BSDS100':
            path1 = p1 + 'B100'+'/'
        else:
            path1 = p1 + dataset+'/'
        path2 = p2+dataset+'/'

        files = os.listdir(path1)
        for file in files:
            img1 = cv2.imread(os.path.join(path1, file))
            if dataset == 'Urban100':
                temp = file.split('_x')[0]
                temp = temp.replace('g', 'g_')
                img2 = cv2.imread(os.path.join(path2, temp+".png"))
            else:
                img2 = cv2.imread(os.path.join(path2, file.split('_x')[0]+".png"))

            img1 = cv2.split(cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb))[0].astype(np.float32)
            img2 = cv2.split(cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb))[0].astype(np.float32)
            # if(img1.shape[-1]>1 and img2.shape[-1]>1):
            #     img1 = cv2.split(rgb2ycbcr(img1))[0]
            #     img2 = cv2.split(rgb2ycbcr(img2))[0]

            psnrs.append(getPsnr(img1, img2, 2))
            ssims.append(getSSIM(img1, img2, 2))
            # print("{}:      {:.2f}/{:.3f}".format(file, psnrs[-1], ssims[-1]))#input each image psnr,ssim
        print("{:.2f}/{:.3f}".format(np.mean(psnrs),np.mean(ssims)))

def rgb2ycbcr(rgb_img):
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    ycbcr_img = np.zeros(rgb_img.shape)
    offset = np.array([16, 128, 128])

    for x in range(rgb_img.shape[0]):
        for y in range(rgb_img.shape[1]):
            ycbcr_img[x, y, :] = np.round(np.dot(mat, rgb_img[x, y, :] * 1.0 / 255) + offset)
    return ycbcr_img


def getPsnr(ref, target,scale):
    # target:目标图像  ref:参考图像
    # 一开始转化为 float64 会降低 PSNR 的值
    # target_data = np.asarray(target, dtype=np.float64)
    # ref_data = np.asarray(ref, dtype=np.float64)

    target_data = np.asarray(target)
    ref_data = np.asarray(ref)

    if target_data.size != ref_data.size:
        if(target_data.shape[0]>ref_data.shape[0]):
            target_data = target_data[0:ref_data.shape[0]]
        elif(target_data.shape[0]<ref_data.shape[0]):
            ref_data = ref_data[0:target_data.shape[0]]

        if (target_data.shape[1] > ref_data.shape[1]):
            target_data = target_data[:, 0:ref_data.shape[1]]
        elif (target_data.shape[1] < ref_data.shape[1]):
            ref_data = ref_data[:, 0:target_data.shape[1]]
    target_data = target_data[scale:-scale,scale:-scale]
    ref_data = ref_data[scale:-scale,scale:-scale]


    diff = (ref_data - target_data)
    mse = np.mean(np.square(diff))
    return 10 * np.log10(255*255/mse)

def getSSIM(ref,target, scale):
    # target:目标图像  ref:参考图像
    target_data = np.array(target)
    ref_data = np.array(ref)

    if target_data.size != ref_data.size:
        if (target_data.shape[0] > ref_data.shape[0]):
            target_data = target_data[0:ref_data.shape[0]]
        elif (target_data.shape[0] < ref_data.shape[0]):
            ref_data = ref_data[0:target_data.shape[0]]

        if (target_data.shape[1] > ref_data.shape[1]):
            target_data = target_data[:, 0:ref_data.shape[1]]
        elif (target_data.shape[1] < ref_data.shape[1]):
            ref_data = ref_data[:, 0:target_data.shape[1]]
    target_data = target_data[scale:-scale, scale:-scale]
    ref_data = ref_data[scale:-scale, scale:-scale]

    return ssim(target_data, ref_data, data_range=255)

# 获取获取图像
def getYCrCb():
    basepath = '/media/wangct/E7D0AC3987855C5C/dataset/benchmark/Source_Urban100/HR/'
    savepath = '/media/wangct/E7D0AC3987855C5C/dataset/benchmark/Urban100/HR/'
    for name in os.listdir(basepath):
        path = basepath + name
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        # simg = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)), interpolation=cv2.INTER_CUBIC)

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        temppath = '{}{}'.format(savepath, name)
        # spath = '{}/s{}'.format(savepath, name)
        cv2.imwrite(temppath, img)
        # cv2.imwrite(spath,simg)
    cv2.destroyAllWindows()

# get multi-scale
def getScale():
    basepath = '/media/wangct/E7D0AC3987855C5C/dataset/benchmark/Set5/HR/'
    savepath = '/media/wangct/E7D0AC3987855C5C/dataset/benchmark/Set5/LR_bicubic/X4/'
    for name in os.listdir(basepath):
        path = basepath + name
        img = cv2.imread(path)
        simg = cv2.resize(img, (int(img.shape[1] /4), int(img.shape[0] / 4)), interpolation=cv2.INTER_CUBIC)

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        temppath = '{}{}{}{}'.format(savepath, name.split('.')[0],'x4.',name.split('.')[1])
        # spath = '{}/s{}'.format(savepath, name)
        cv2.imwrite(temppath, simg)
        # cv2.imwrite(spath,simg)
    cv2.destroyAllWindows()



