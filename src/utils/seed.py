import os
import random

import numpy as np
import torch


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS determinism is limited; still set seeds
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
