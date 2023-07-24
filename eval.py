import torch 
import numpy as np
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


def cal_rate(label, max_idx, idx):
    select_label = label[idx]
    select_output = max_idx[idx]

    correct_count = (select_label == select_output).float().sum().item()
    return correct_count, select_output.shape[0]

def test(hparam = "train.yaml", **kwargs):
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
    tprint("Checkpoint File: {}".format(hp.check_point_file))
    state_dict = torch.load(hp.check_point_file)
    remove_prefix = 'module.'
    state_dict['model'] = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict['model'].items()}

    model.load_state_dict(state_dict['model'])

    dataset = QQQDataSet(
        csv_file_name = hp.csv_file,
        past_window_size = hp.past_window_size,
        window_size= hp.pivot_window_size,
        train_mode = False
    )
    data_loader = DataLoader(dataset,
                             drop_last=True,
                             batch_size=hp.batch_size,
                             num_workers=hp.n_workers,)
    model.eval()    
    with torch.no_grad():
        sum_recall_correct_count = 0
        sum_recall_count = 0

        sum_correct_correct_count = 0
        sum_correct_count = 0

        sum_up_correct_ret = []
        sum_down_correct_ret = []
        sum_up_recall_ret = []
        sum_down_recall_ret = []

        for test_sample in data_loader:
            output = model(test_sample['lstm_input'], test_sample['cnn_input'])
            max_val, max_idx = torch.max(output, 1)
            label = test_sample['label']
            # recall rate
            recall_up_idx = (label == 0).nonzero(as_tuple=True)[0]
            recall_down_idx = (label == 1).nonzero(as_tuple=True)[0]

            # correct rate
            correct_up_idx = (max_idx == 0).nonzero(as_tuple=True)[0]
            correct_down_idx = (max_idx == 1).nonzero(as_tuple=True)[0]

            recall_up_and_down_idx = torch.concat((recall_up_idx, recall_down_idx))
            correct_up_and_down_idx = torch.concat((correct_up_idx, correct_down_idx))
            #up_and_down_idx = up_idx 
            #if up_and_down_idx.shape[0] == 0:
            #    continue

            recall_up_and_down_label = label[recall_up_and_down_idx]
            recall_up_and_down_output = max_idx[recall_up_and_down_idx]

            recall_correct_count = (recall_up_and_down_label == recall_up_and_down_output).float().sum().item()
            sum_recall_correct_count += recall_correct_count
            sum_recall_count += recall_up_and_down_label.shape[0] 
            #print("Sample acc:", correct_count / up_and_down_label.shape[0])

            correct_up_and_down_label = label[correct_up_and_down_idx]
            correct_up_and_down_output = max_idx[correct_up_and_down_idx]
            correct_correct_count = (correct_up_and_down_label == correct_up_and_down_output).float().sum().item()
            sum_correct_correct_count += correct_correct_count
            sum_correct_count += correct_up_and_down_label.shape[0]

            sum_up_correct_ret.append(cal_rate(label, max_idx, correct_up_idx))
            sum_down_correct_ret.append(cal_rate(label, max_idx, correct_down_idx))
            sum_up_recall_ret.append(cal_rate(label, max_idx, recall_up_idx))
            sum_down_recall_ret.append(cal_rate(label, max_idx, recall_down_idx))

        sum_up_correct_ret = np.array(sum_up_correct_ret).sum(0)
        sum_down_correct_ret = np.array(sum_down_correct_ret).sum(0)
        sum_up_recall_ret = np.array(sum_up_recall_ret).sum(0)
        sum_down_recall_ret = np.array(sum_down_recall_ret).sum(0)



        print("Acc:", sum_correct_correct_count / sum_correct_count)
        print("Recall:", sum_recall_correct_count/ sum_recall_count)
        print("up acc:", sum_up_correct_ret[0] / sum_up_correct_ret[1], sum_up_correct_ret[1])
        print("down acc:", sum_down_correct_ret[0] / sum_down_correct_ret[1], sum_down_correct_ret[1])
        print("up recall:", sum_up_recall_ret[0] / sum_up_recall_ret[1], sum_up_recall_ret[1])
        print("down recall:", sum_down_recall_ret[0] / sum_down_recall_ret[1], sum_down_recall_ret[1])


    
if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    fire.Fire(test)

