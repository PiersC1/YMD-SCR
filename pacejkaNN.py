import torch
import torch.nn as nn
import numpy as np
import json


class SimpleNN(nn.Module):
    def __init__(self, size):
        super(SimpleNN, self).__init__()
        layers = []
        for i in range(len(size) - 1):
            layers.append(nn.Linear(size[i], size[i+1]))
            if i < len(size) - 2:
                layers.append(nn.GELU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

class PacejkaHead(nn.Module):
    def __init__(self, model_size):
        super(PacejkaHead, self).__init__()

        # print(model_size)

        assert model_size[-1] == 2

        self.FFN = SimpleNN(model_size)

        # self.E = torch.tensor([1.0]).to(self.device)

    def forward(self, main_block_outs, alpha):
        
        params = self.FFN(main_block_outs)

        B = params[..., 0]
        D = params[..., 1]

        out = D * torch.sin(2.2 * torch.atan(torch.atan(B * alpha)))

        return out

class PacejkaNN_splitHead(nn.Module):
    def __init__(self, size, SA_index, SA_mean, SA_var, split = -2):

        # Expects size[:split] to be backbone sizes
        super(PacejkaNN_splitHead, self).__init__()
        layers = []

        ##### BACKBONE_SIZE #####
        backbone_size = size[:split]
        for i in range(len(backbone_size) - 1):
            if i == 0:
                layers.append(nn.Linear(backbone_size[0] - 1, backbone_size[i+1]))
                # layers.append(nn.Linear(backbone_size[0], backbone_size[i+1]))
            else:
                layers.append(nn.Linear(backbone_size[i], backbone_size[i+1]))
            
            if i < len(backbone_size) - 2:
                layers.append(nn.GELU())
        
            
        self.network = nn.Sequential(*layers)
        #########################

        ##### HEADS #############

        self.SA_index = SA_index
        self.SA_mean = SA_mean
        self.SA_var = SA_var

        self.input_columns = list(range(size[0]))
        self.input_columns.remove(self.SA_index)

        head_size = size[split-1:]

        # self.pacejka = PacejkaFunction()
        # self.pacejka = PacejkaFYMZ()

        self.FY_head = PacejkaHead(head_size + [2])
        self.MZ_head = PacejkaHead(head_size + [2])
        # self.MX_head = PacejkaHead(head_size + [2])

        # self.FX_head = SimpleHead(head_size + [1])
        # #########################

        with open("./NN_model/norm_params.json") as f:
            self.NORM_PARAMS = json.load(f)

    def forward(self, x):
        denorm_SA = (x[..., self.SA_index] * self.SA_var) + self.SA_mean
        x = x[..., self.input_columns]
        params = self.network(x)

        fy = self.FY_head(params, denorm_SA)
        mz = self.MZ_head(params, denorm_SA)
        # mx = self.MX_head(params, denorm_SA)
        # fx = self.FX_head(params, denorm_SA)

        # return torch.stack((fy, mz, mx, fx), axis = -1)
        return torch.stack((fy, mz), axis = -1)

    def compute_forces(self, FZ, SA):
        
        INPUT_COLUMNS = ['FZ', 'IA', 'P', 'SA', 'TSTC', 'V', 'tire_dim1', 'tire_dim2']
        # Pressure is in KPa
        SAMPLE_INPUT = torch.zeros((1,8)).to(torch.float32)


        SA = np.rad2deg(SA)
        # converting FZ, SA to normalized values
        norm_fz = (FZ - self.NORM_PARAMS["FZ"][0])/ self.NORM_PARAMS["FZ"][1]
        norm_sa = (SA - self.NORM_PARAMS["SA"][0])/ self.NORM_PARAMS["SA"][1]

        SAMPLE_INPUT[0][INPUT_COLUMNS.index("FZ")] = norm_fz
        SAMPLE_INPUT[0][INPUT_COLUMNS.index("SA")] = norm_sa

        out = self.forward(SAMPLE_INPUT)[0].detach()
        fy, mz = out[0].item(), out[1].item()

        # denorming the outputs

        fy = (fy * self.NORM_PARAMS["FY"][1]) + self.NORM_PARAMS["FY"][0]
        mz = (mz * self.NORM_PARAMS["MZ"][1]) + self.NORM_PARAMS["MZ"][0]

        scale = .93
        fy *= scale
        mz *= scale
        return fy, mz



def load_model(norm_path, model_weights):

    INPUT_COLUMNS = ['FZ', 'IA', 'P', 'SA', 'TSTC', 'V', 'tire_dim1', 'tire_dim2']
    with open(norm_path) as f:
        NORM_PARAMS = json.load(f)

    weights = torch.load(model_weights, map_location=torch.device('cpu'))

    SA_index = INPUT_COLUMNS.index("SA")
    SA_mean = NORM_PARAMS["SA"][0]
    SA_var = NORM_PARAMS["SA"][1]

    model = PacejkaNN_splitHead([len(INPUT_COLUMNS), 32, 32, 16], SA_index=SA_index, SA_mean=SA_mean, SA_var=SA_var, split=-1)
    model.load_state_dict(weights)

    return model