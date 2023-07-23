import time
from torch.cuda import nvtx
import os
import yaml
import torch


def tprint(msg):
    print('[{}] {}'.format(time.strftime('%Y%m%d %H:%M:%S'), msg))



class Nvtx(object):
    
    def __init__(self, name, enabled=True):
        self.name = name
        self.enabled = enabled

    def __call__(self, f):
        def wrapped_f(*args, **kwargs):
            with Nvtx(self.name, self.enabled):
                return f(*args, **kwargs)
        return wrapped_f
    
    def __enter__(self):
        if self.enabled:
            nvtx.range_push(self.name)
    
    def __exit__(self, *exc_info):
        if self.enabled:
            nvtx.range_pop()



def to_device_async(tensor, device):
    return tensor.to(device, non_blocking=True)


def to_gpu_async(cpu_tensor):
    return cpu_tensor.to('cuda', non_blocking=True)


def to_cpu_numpy(gpu_tensor):
    if not isinstance(gpu_tensor, torch.Tensor):
        return gpu_tensor
    return gpu_tensor.detach().cpu().numpy()

def remove_module_in_state_dict(state_dict):
    """
    If model is saved with DataParallel, checkpoint keys is started with 'module.' remove it and return new state dict
    :param checkpoint:
    :return: new checkpoint
    """
    new_state_dict = {}
    for key, val in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = val
    return new_state_dict



class TimeElapsed(object):
    
    def __init__(self, name, device='cuda', cuda_sync=False, format=""):
        self.name = name
        self.device = device
        self.cuda_sync = cuda_sync
        self.format = format
    
    def __enter__(self):
        self.start()
    
    def __exit__(self, *exc_info):
        self.end()

    def start(self):
        if self.device == 'cuda' and self.cuda_sync:
            torch.cuda.synchronize()
        self.start_time = time.time()

    def end(self):
        if not hasattr(self, "start_time"):
            return
        if self.device == 'cuda' and self.cuda_sync:
            torch.cuda.synchronize()
        self.end_time = time.time()
        self.time_elapsed = self.end_time - self.start_time
        tprint(("[{}] Time elapsed: {" + self.format + "}").format(self.name, self.time_elapsed))

def load_hparam(filepath):
    hparam_dict = dict()

    if not filepath:
        return hparam_dict

    stream = open(filepath, 'r')
    docs = yaml.load_all(stream, Loader=yaml.Loader)
    for doc in docs:
        for k, v in doc.items():
            hparam_dict[k] = v
    return hparam_dict


def merge_dict(new, default):
    if isinstance(new, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in new:
                new[k] = v
            else:
                new[k] = merge_dict(new[k], v)
    return new


class Dotdict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value

    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError as e:
            return None


class Hparam(Dotdict):

    __getattr__ = Dotdict.__getattr__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__

    def __init__(self, root_path):
        self.hp_root_path = root_path
        super(Hparam, self).__init__()

    def set_hparam(self, filename, hp_commandline=dict()):

        def get_hp(file_path):
            """
            It merges parent_yaml in yaml recursively.
            :param file_rel_path: relative hparam file path.
            :return: merged hparam dict.
            """
            hp = load_hparam(file_path)
            if 'parent_yaml' not in hp:
                return hp
            parent_path = os.path.join(self.hp_root_path, hp['parent_yaml'])

            if parent_path == file_path:
                raise Exception('To set myself({}) on parent_yaml is not allowed.'.format(file_path))

            base_hp = get_hp(parent_path)
            hp = merge_dict(hp, base_hp)
            
            return hp

        hparam_path = os.path.join(self.hp_root_path, filename)

        hp = get_hp(hparam_path)
        hp = merge_dict(hp_commandline, hp)

        hp = Dotdict(hp)

        for k, v in hp.items():
            setattr(self, k, v)