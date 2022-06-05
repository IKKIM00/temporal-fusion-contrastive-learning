import torch
import torch.nn as nn


class BaseLogit(nn.Module):

    def __init__(self, model_params, static_use=False):
        super(BaseLogit, self).__init__()

        params = dict(model_params)

        self.output_dim = int(params['encoder_output_dim'])
        self.num_classes = int(params['num_classes'])

        self.logits = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.output_dim // 2, self.output_dim // 4),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.output_dim // 4, self.num_classes)
            # nn.ReLU(),
            # nn.Dropout(p=0.2)
        )
        self.globalMaxPool1D = nn.AdaptiveMaxPool1d(1)

    def forward(self, encoder_output):

        x_flat = self.globalMaxPool1D(encoder_output).squeeze()
        logits = self.logits(x_flat)
        return logits





class SimclrLogit(nn.Module):

    def __init__(self, model_params):
        super(SimclrLogit, self).__init__()
        params = dict(model_params)
        self.num_classes = int(params['num_classes'])
        self.logits = nn.Linear(96, self.num_classes)

    def forward(self, encoder_output):

        return self.logits(encoder_output)






class CSSHARLogit(nn.Module):
    def __init__(self, model_params):
        super(CSSHARLogit, self).__init__()

        params = dict(model_params)
        self.input_seq = int(params['input_seq'])
        self.num_classes = int(params['num_classes'])

        self.logits = nn.Sequential(
            nn.Linear(self.input_seq * 96, 96),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, self.num_classes)
        )

    def forward(self, encoder_output):

        return self.logits(encoder_output)


""" 
CPC 계열은 Linear classifier로서 동작하게 만듦
"""


class CPCHARLogit(nn.Module):
    

    def __init__(self, model_params):
        super(CPCHARLogit, self).__init__()

        params = dict(model_params)
        self.num_classes = int(params['num_classes'])
        self.timestep = int(params['timestep'])
        self.logits = nn.Sequential(
        
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.num_classes)
        )
        

    def forward(self, c_t):

        c_t = c_t.permute(0, 2, 1).contiguous()
        c_t = c_t[:, self.timestep,:]
        out = self.logits(c_t)

        return 