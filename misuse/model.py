import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import numpy as np
from .struct import struct_attn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Propogator(nn.Module):
    def __init__(self, node_dim):
        super(Propogator, self).__init__()
        self.node_dim = node_dim
        self.reset_gate = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.Tanh()
        )

    def forward(self, node_representation, adjmatrixs):  # ICLR2016 fomulas implementation
        a = torch.bmm(adjmatrixs, node_representation)
        joined_input1 = torch.cat((a, node_representation), 2)
        z = self.update_gate(joined_input1)
        r = self.reset_gate(joined_input1)
        joined_input2 = torch.cat((a, r * node_representation), 2)
        h_hat = self.tansform(joined_input2)
        output = (1 - z) * node_representation + z * h_hat
        return output


class EncoderGGNN(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EncoderGGNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.state_dim = 256
        self.n_steps = 5
        self.propogator = Propogator(self.state_dim)
        self.out1 = nn.Sequential(
            nn.Linear(self.state_dim + self.state_dim, self.state_dim),
            nn.Tanh()
        )
        self.out2 = nn.Sequential(  # this is new adding for graph-level outputs
            nn.Linear(self.state_dim + self.state_dim, self.state_dim),
            nn.Sigmoid()
        )
        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, images, adjmatrixs, lengths):
        lengths = torch.Tensor(lengths).reshape(-1, 1).to(device)
        embeddings = self.embed(images).to(device)
        node_representation = embeddings
        init_node_representation = node_representation
        for i_step in range(self.n_steps):  # time_step updating
            node_representation = self.propogator(node_representation, adjmatrixs)
        gate_inputs = torch.cat((node_representation, init_node_representation), 2)
        # gate_outputs = self.out1(gate_inputs)
        # features = torch.sum(gate_outputs, 1)
        # features = features / lengths

        # graph-level models with soft attention
        gate_outputs1 = self.out1(gate_inputs)
        gate_outputs2 = self.out2(gate_inputs)
        gate_outputs = gate_outputs1 * gate_outputs2
        features = struct_attn(gate_outputs, 1)    # attention pooling
        features = features / lengths

        return features
