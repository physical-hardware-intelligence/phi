# Vendored from real-stanford/diffusion_policy (MIT). Retrieved 2026-08-13.
# Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab

import torch.nn as nn

class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
