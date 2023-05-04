import numpy as np
import torch
import torch.nn.functional as F

if __name__ == '__main__':
    a=torch.ones(1,2)*3
    b=torch.zeros(1,2)
    c=F.mse_loss(a,b)
    print(c)
