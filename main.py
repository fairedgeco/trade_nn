import torch 
import fire
import os
import pprint

from nn import CNN_LSTM_MIX_MODEL
from data_reader import QQQDataSet
from torch.utils.data import DataLoader
from utils import tprint, Hparam
from trainer import Trainer

HP_ROOT_PATH = os.path.dirname(__file__)
hp = Hparam(HP_ROOT_PATH)
pp = pprint.PrettyPrinter(indent=4, width=1000)


def train(hparam = "train.yaml", **kwargs):
    hp.set_hparam(hparam, kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tprint("Hparams:\n{}".format(pp.pformat(hp)))
    tprint("Device count: {}".format(torch.cuda.device_count()))
    model = CNN_LSTM_MIX_MODEL(
        lstm_hidden_size=hp.lstm_hidden_size,
        lstm_input_size=hp.lstm_input_size,
        lstm_num_layers=hp.lstm_num_layers,
        name = "Mix_Model"
    )
    dataset = QQQDataSet(
        csv_file_name = hp.csv_file,
        past_window_size = hp.past_window_size,
        window_size= hp.pivot_window_size,
    )
    data_loader = DataLoader(dataset,
                             drop_last=True,
                             batch_size=hp.batch_size,
                             num_workers=hp.n_workers,)

    def get_optimizer(model):
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hp.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9)
        return optimizer
    
    trainer = Trainer(data_loader,
                      'MixModel',
                      model,
                      optimizer_fn=get_optimizer,
                      final_steps=hp.final_steps,
                      log_steps=hp.log_step,
                      ckpt_path=hp.checkpoint_path,
                      save_steps=hp.save_step,
                      log_path=hp.log_path,
                      device=device,
                      n_epochs=hp.epochs,
                      use_amp=hp.use_amp,
                      nvprof_iter_start=hp.nvprof_iter_start,
                      nvprof_iter_end=hp.nvprof_iter_end,
                      pyprof_enabled=hp.pyprof_enabled,
                      )
    trainer.train()
    
if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    fire.Fire(train)