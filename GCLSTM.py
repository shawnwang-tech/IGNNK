import torch
from torch import nn
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse


class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(-1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return hy


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        hx, cx = hidden

        x = x.view(-1, x.size(-1))

        gates = self.x2h(x) + self.h2h(hx)

        gates = gates.squeeze()

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)

        hy = torch.mul(outgate, F.tanh(cy))

        return (hy, cy)

class LSTM(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, num_nodes, batch_size, device):
        super(LSTM, self).__init__()
        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.hid_dim = 32
        self.out_dim = 1
        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)
        self.lstm_cell = LSTMCell(self.hid_dim, self.hid_dim)

    def forward(self, pm25_hist, feature):
        pm25_pred = []
        h0 = torch.zeros(self.batch_size * self.num_nodes, self.hid_dim).to(self.device)
        hn = h0
        c0 = torch.zeros(self.batch_size * self.num_nodes, self.hid_dim).to(self.device)
        cn = c0
        xn = pm25_hist[:, -1]
        for i in range(self.pred_len):
            x = torch.cat((xn, feature[:, self.hist_len+i]), dim=-1)
            x = self.fc_in(x)
            hn, cn = self.lstm_cell(x, (hn, cn))
            xn = hn.view(self.batch_size, self.num_nodes, self.hid_dim)
            xn = self.fc_out(xn)
            pm25_pred.append(xn)
        pm25_pred = torch.stack(pm25_pred, dim=1)
        return pm25_pred


import torch.nn.functional as F
from torch_geometric.nn import ChebConv
import torch
from torch import nn


class GC_LSTM(nn.Module):
    def __init__(self, h, z, K):
        super(GC_LSTM, self).__init__()

        # TODO, x feature
        self.in_dim = 1
        self.hid_dim = 32
        self.out_dim = 1
        self.gcn_out = 32

        self.time_dimension = h

        self.device = torch.device("cuda:0")

        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)

        self.conv = ChebConv(self.hid_dim, self.gcn_out, K=1)
        self.lstm_cell = LSTMCell(self.hid_dim + self.gcn_out, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, x, A_q, A_h, A):
        batch_size = x.size(0)
        num_nodes = x.size(2)
        num_steps = x.size(1)

        # TODO, x feature
        x = x[..., None]
        in_dim = 1

        edge_index, edge_weight = dense_to_sparse(A.to(torch.device("cuda:0")))
        edge_index = edge_index.view(2, 1, -1).repeat(1, batch_size, 1) + torch.arange(batch_size).view(1, -1, 1).to(torch.device("cuda:0")) * batch_size
        edge_index = edge_index.view(2, -1)

        pm25_pred = []
        h0 = torch.zeros(batch_size * num_nodes, self.hid_dim).to(self.device)
        hn = h0
        c0 = torch.zeros(batch_size * num_nodes, self.hid_dim).to(self.device)
        cn = c0
        xn = x[:, 0]
        for i in range(num_steps):

            x = self.fc_in(xn)
            x = F.relu(x)

            x_gcn = x.contiguous()
            x_gcn = x_gcn.view(batch_size * num_nodes, -1)
            x_gcn = F.relu(self.conv(x_gcn, edge_index))
            x_gcn = x_gcn.view(batch_size, num_nodes, -1)
            x = torch.cat((x, x_gcn), dim=-1)
            hn, cn = self.lstm_cell(x, (hn, cn))
            xn = hn.view(batch_size, num_nodes, self.hid_dim)
            xn = self.fc_out(xn)
            pm25_pred.append(xn)

        pm25_pred = torch.stack(pm25_pred, dim=1)[..., 0]

        return pm25_pred