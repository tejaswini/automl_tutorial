import random
import numpy as np
import torch

def set_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
#        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(1)