import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class QQQDataSet(Dataset):
    def __init__(self, csv_file_name, past_window_size = 4):
        self.past_window_size = past_window_size
        with open(csv_file_name, "r") as fp:
            self.pd_data= pd.read_csv(fp)
            self.pd_data = self.pd_data[0:int(len(self.pd_data) * 0.85)]
        self.get_windows_pivot()
    
    def __len__(self):
        return len(self.pivot_list) - self.past_window_size 

    def __getitem__(self, idx):
        pivot_range = self.pivot_list[idx : idx + self.past_window_size]
        original_data = self.pd_data[idx * 10: (idx + self.past_window_size) * 10]
        lstm_data = []
        cnn_data = []
        for i in range(self.past_window_size):
            pivot = pivot_range[i]
            lstm_item = [pivot[3], pivot[4] ]
            lstm_data.append(lstm_item)
            cnn_item = []
            for j in range(10):
                price_data = original_data.iloc[i + j]
                minute_data = [price_data.Open, price_data.High, price_data.Low, price_data.Close]
                cnn_data.append(minute_data)

        
        lstm_data = np.array(lstm_data)
        cnn_data = np.array(cnn_data)
        cnn_data = cnn_data - cnn_data[0][0]
        cnn_data = cnn_data.T

        target = self.pivot_list[idx + self.past_window_size]
        if target[3] > 0 and target[4] > 0:
            target_label = 0
        elif target[3] <= 0 and target[4] <= 0:
            target_label = 1
        elif target[3] > 0 and target[4] <= 0:
            target_label = 2
        elif target[3] <= 0 and target[4] > 0:
            target_label = 3 
        else:
            print("Can not find target_label", target[3], target[4])
        #target_label = [target[3], target[4]]
        
        trainning_item = {
            "lstm_input" : lstm_data.astype(np.float32),
            "cnn_input" : cnn_data.astype(np.float32),
            "label" : target_label
        }

        return trainning_item 


    
    def get_windows_pivot(self):
        self.pivot_list = []
        for i in range(0, len(self.pd_data), 10):
            lagging_window = self.pd_data[i: i+10]
            #print(lagging_window)
            #max_index = lagging_window.High.idxmax()
            #min_index = lagging_window.Low.idxmin()
            #print(max_index, min_index)
            max_price = lagging_window.High.max()
            min_price = lagging_window.Low.min()
            if len(self.pivot_list) == 0:
                self.pivot_list.append([lagging_window['DateTime'].iloc[0], max_price, min_price])
                #print(self.pivot_list[-1])
                continue
            previous_pivot = self.pivot_list[-1]
            max_label = max_price - previous_pivot[1]
            min_label = min_price - previous_pivot[2]
            #if max_price > previous_pivot[1]:
            #    max_label = 'HH'
            #else:
            #    max_label = 'LH'
            #if min_price > previous_pivot[2]:
            #    min_label = 'HL'
            #else:
            #    min_label = 'LL'
            
            self.pivot_list.append([lagging_window['DateTime'].iloc[0], max_price, min_price, max_label, min_label])
            #if max_label == 'HH' and min_label == 'HL':
            #    print(self.pivot_list[-1], "Uptrends")
            #elif max_label == 'LH' and min_label == 'LL':
            #    print(self.pivot_list[-1], "Downtrends")
            #else:
            #    print(self.pivot_list[-1])
        self.pd_data = self.pd_data[10:]
        self.pivot_list = self.pivot_list[1:]



if __name__ == "__main__":

    data_loader = QQQDataSet("./QQQ_whole_ET.csv")

    print(data_loader.__getitem__(0))
    print(data_loader.__len__() / 64)