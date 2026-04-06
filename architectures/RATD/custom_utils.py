import numpy as np
import torch


##for loading the data when creating retreival indices --retriever mode of retrieval_builder.py
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path, flag='train'):
        super().__init__()
    
        
        data = np.load(path)
        positions = data["positions"]
        train_test_split = int(data["train_test_split"])
        
        #test_size=int(data["test_size"])
        #val_size=int(data["val_size"])
        
        if flag == 'train':
            self.data = positions[:, :train_test_split ]
        elif flag == 'val':
            self.data = positions[:, train_test_split:train_test_split]
            #self.data = positions[:val_size, val_start:]

        elif  "test":
            self.data = positions[:, train_test_split:train_test_split+100]
            #self.data = positions[:test_size, train_test_split:]
        
        self.data_x = self.data  
        print(self.data_x.shape)
        
    def __len__(self):
        return self.data.shape[1]
    
    def __getitem__(self, idx):
        return self.data[idx, :] #retrusn the trjectory of idnex idx
    
