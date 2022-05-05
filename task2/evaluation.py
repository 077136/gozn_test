import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from model import Net


def eval():
    net = Net()
    net.load_state_dict(torch.load('./weights/last_weights.pth'))
    net.eval()
    a = np.load('./test_item.npy')
    a = torch.from_numpy(a).unsqueeze(0).float()
    out = net(a).detach()
    
    fig, axes = plt.subplots(2, 1)
    axes[0].imshow(a.permute(2,1,0), label="in")
    axes[1].imshow(out.permute(2,1,0), label="out")
    plt.savefig('./test_item.png')

if __name__ == "__main__":
    eval()