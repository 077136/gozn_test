import os
import json
from datetime import datetime
import numpy as np
import logging
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from utils import train, test, get_logger, CustomDataset, collate_fun
from model import Net


def main():
    log = logging.getLogger(__name__)

    device = 'cuda'#'cpu'#"cuda:9" 
    start_time = datetime.now().isoformat()
    log.info(start_time)
    os.makedirs('results', exist_ok=True)
    os.makedirs('weights', exist_ok=True)
    save_path = os.path.join(os.path.expanduser('./results'), start_time)
    os.makedirs(save_path, exist_ok=True)


    data_path = os.path.join('path','to','data')
    train_set = CustomDataset(os.path.join(data_path,'train'))
    val_set = CustomDataset(os.path.join(data_path,'val'))
    
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fun)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fun)    

    net = Net()
    net = net.to(device)
    #net = torch.nn.DataParallel(net, device_ids=[0, 1])
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(net.parameters(), lr=0.005, weight_decay=5e-4)

    kwargs = {
            "net": net,
            "train_loader": train_loader,
            "test_loader": val_loader,
            "criterion": criterion,
            "optimizer": optimizer,
            "device": device,
        }

    best_loss = 1e5
    for epoch in range(0, 5):
            train(epoch, **kwargs)
            test_loss = test(**kwargs)
            log.info(test_loss)
            if test_loss < best_loss:
                torch.save(net.state_dict(), f'./weights/{start_time}.pth')
                #torch.save((net.module.state_dict(), './wn.pth'))
    torch.save(net.state_dict(), './weights/last_weights.pth')
            


if __name__ == "__main__":
    main()