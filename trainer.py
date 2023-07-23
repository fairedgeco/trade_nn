import abc
import glob
import pathlib

import numpy as np
import torch
from tensorboardX import SummaryWriter
import time
import os
import matplotlib.pyplot as plt
from torch import nn

from utils import tprint
from utils import to_device_async
from utils import Nvtx
from utils import to_cpu_numpy 

import torch.cuda.profiler as profiler
from utils import tprint
from utils import TimeElapsed
from contextlib import nullcontext


class Trainer(object):
    """
    set seed
    set n_epochs, n_steps
    save/load model
    validation
    logging
    distributed
    """

    def __init__(self, data_loader, model_name, model, optimizer_fn, final_steps, step=0, ckpt_path=None, log_path=None, n_epochs=None, save_steps=None, log_steps=10, device='cuda', use_amp=False, nvprof_iter_start=None, nvprof_iter_end=None, pyprof_enabled=False, detect_anomaly=False, seed=None):
        self.data_loader = data_loader
        self.model_name = model_name
        self.model = model
        self.n_epochs = n_epochs
        self.save_steps = save_steps
        self.log_steps = log_steps
        self.ckpt_path = ckpt_path
        self.log_path = log_path
        self.final_steps = final_steps
        self.step = step
        self.device = device
        self.use_amp = use_amp
        self.nvprof_iter_start = nvprof_iter_start
        self.nvprof_iter_end = nvprof_iter_end
        self.pyprof_enabled = pyprof_enabled
        self.detect_anomaly = detect_anomaly

        # model
        self.model.train()
        to_device_async(self.model, self.device)
        num_param = sum(param.numel() for param in model.parameters())
        tprint('The number of {} parameters: {}'.format(
            self.model_name, num_param))

        # optimizer
        self.optimizer = optimizer_fn(model)

        # automatic mixed precision
        if self.use_amp:
            from apex import amp
            self.model, self.optimizer = amp.initialize(self.model, 
                                                        self.optimizer, 
                                                        opt_level='O1')

        # profile
        if nvprof_iter_start and nvprof_iter_end is not None and pyprof_enabled:
            from apex import pyprof
            pyprof.nvtx.init()

        # data parallel
        self.model = nn.DataParallel(self.model)

        # set seed
        if seed is None:
            seed = np.random.randint(2**16)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # data loader
        self.data_loader_iter = self.repeat(self.data_loader, n_epochs)

        # logging
        if log_path:
            # tensorboard log path : {log_path}/YYYYMMDD-HHMMMSS
            log_path = os.path.join(log_path, time.strftime('%Y%m%d-%H%M%S'))
            self.tbwriter = SummaryWriter(log_dir=log_path, flush_secs=10)

        # checkpoint path
        if self.ckpt_path:
            self.ckpt_path = os.path.join(self.ckpt_path, self.model_name)
            pathlib.Path(self.ckpt_path).mkdir(parents=True, exist_ok=True)

            # load checkpoint
            self.load()

    def train(self):
        try:
            with torch.autograd.profiler.emit_nvtx(enabled=self.pyprof_enabled):
                for i in range(self.step+1, self.final_steps + 1):
                    self.step = i
                    tprint("------------- TRAIN step : {} -------------".format(i))

                    if self.nvprof_iter_start and i == self.nvprof_iter_start:
                        profiler.start()
                        timer = TimeElapsed(name="Training time during profiling", format=":.6f")
                        timer.start()

                    with Nvtx("step #{}".format(self.step)) if self.pyprof_enabled else nullcontext() as gs:
                        loss, meta = self.do_step()

                    if self.nvprof_iter_end and i == self.nvprof_iter_end:
                        profiler.stop()
                        timer.end()
        

                    if self.step % self.log_steps == 0:
                        self.log(loss, meta)

                    if self.ckpt_path and self.save_steps and i % self.save_steps == 0:
                        self.save()

            tprint("Training has been done.")
        except StopIteration:  # done by n_epochs
            tprint("Training has been done. (by n_epochs)")
        except KeyboardInterrupt:
            tprint("Training has been canceled.")

    def loss(self, inputs, model):
        cnn_input = inputs['cnn_input']
        lstm_input = inputs['lstm_input']
        target = inputs['label']

        # tprint(cnn_input)
        # tprint(lstm_input)

        output = model(lstm_input, cnn_input)
        cross_entropy_loss = nn.CrossEntropyLoss()

        cross_entropy = cross_entropy_loss(output, target)
        #meta = {"cross_lost": to_cpu_numpy(output)}
        meta = {} 
        return cross_entropy, meta


    def do_step(self):
        with Nvtx("data load", enabled=False) if self.pyprof_enabled else nullcontext() as gs:
            data = next(self.data_loader_iter)

        with torch.autograd.set_detect_anomaly(mode=self.detect_anomaly):
            with Nvtx("forward") if self.pyprof_enabled else nullcontext() as gs:
                loss, meta = self.loss(data, self.model)
        
            self.optimizer.zero_grad()

            with Nvtx("backward") if self.pyprof_enabled else nullcontext() as gs:
                if self.use_amp:
                    from apex import amp
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

        with Nvtx("weight update") if self.pyprof_enabled else nullcontext() as gs:
            self.optimizer.step()

        return loss, meta

    def log(self, loss, meta):
        self.console_log('train', loss, meta)
        if self.log_path:
            self.tensorboard_log('train', loss)

    def save(self):
        state_dict = {
            'step': self.step,
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict(),
        }
        torch.save(state_dict, self.ckpt_path +
                   '/checkpoint_{:06d}.pt'.format(self.step))

        tprint('[Save] Model "{}". Step={}.'.format(
            self.model_name, self.step))

    def load(self, load_optim=True):
        files_exist = glob.glob(os.path.join(self.ckpt_path, '*'))
        if files_exist:
            # load the latest created file.
            latest_file = max(files_exist, key=os.path.getctime)
            state_dict = torch.load(latest_file)

            self.step = state_dict['step']
            self.model.load_state_dict(state_dict['model'])
            if load_optim:
                self.optimizer.load_state_dict(state_dict['optim'])

            tprint('[Load] Checkpoint \'{}\'. Step={}'.format(
                latest_file, self.step))
        else:
            tprint('No checkpoints in {}. Load skipped.'.format(self.ckpt_path))

    def console_log(self, tag, loss, meta):
        # console logging
        msg = 'loss: {:.6f}'.format(loss)
        for key, value in meta.items():
            msg += ',\t{}: {:.4f}'.format(key, value)
        tprint(msg)

    def tensorboard_log(self, tag, loss):
        self.tbwriter.add_scalar(
            '{}/loss'.format(tag), loss, global_step=self.step)

    @staticmethod
    def repeat(iterable, n_repeat=None):
        cnt = 0
        while n_repeat is None or cnt < n_repeat:
            for x in iterable:
                yield x
            cnt += 1
        return StopIteration()
    
