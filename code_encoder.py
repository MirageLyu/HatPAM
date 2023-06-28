import torch
import torch.nn as nn
import ggnn_utils
from torch.utils.data import DataLoader
from torch.nn import functional as F
import json
from gensim.models import Word2Vec
from tqdm import tqdm
from natural_encoder import load_transformer_xl_model, load_transformer_xl_tokenizer, NaturalLanguageEmbeddingLayer
from dataset import MisuseDataset


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available')
    print("Device name: " + torch.cuda.get_device_name(0))
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device('cpu')


def sequence_to_padding(x, length):
    # declare the shape, it can work for x of any shape.
    ret_tensor = torch.zeros(length)
    ret_tensor[:x.shape[-1]] = x
    return ret_tensor



class NLPLModel(nn.Module):
    def __init__(self, nl_od, transfo_xl_model, gnn_ln_in_d, gnn_ln_out_d, whole_out_d, num_nodes):
        super().__init__()
        self.nl_layer = NaturalLanguageEmbeddingLayer(transfoxl_model=transfo_xl_model, out_dimension=nl_od).to(device)
        self.gnn_layer = ggnn_utils.GraphClsGGNN(gnn_ln_in_d, 1, 5, 1, 6).to(device)
        self.gnn_layer2 = ggnn_utils.GraphClsGGNN(gnn_ln_in_d, 1, 5, 1, 6).to(device)
        self.gnn_linear = nn.Linear(gnn_ln_in_d, gnn_ln_out_d)
        self.linear2 = nn.Linear(nl_od + gnn_ln_out_d, whole_out_d)
        # self.linear2 = nn.Linear(gnn_ln_out_d, whole_out_d)

    def forward(self, nl_input_ids, data):
        nl_output = self.nl_layer(nl_input_ids)
        pl_output = self.gnn_layer(data[0])
        pl_output2 = self.gnn_layer2(data[1])
        # print(pl_output.shape)
        # pl_output = torch.mean(pl_output, dim=0)
        # pl_output = self.gnn_linear(torch.flatten(pl_output))
        # pl_output = F.relu(pl_output)
        # print(pl_output.shape)
        nl_output = nl_output.view(nl_output.shape[-1])
        out_concat = torch.cat((nl_output, pl_output, pl_output2), 0)
        out = F.softmax(self.linear2(out_concat))
        return out.view(1, out.shape[-1])
        # return out


lrate = 0.001
epoch_num = 200

loss_fn = nn.CrossEntropyLoss()

# vocab = load_nonterminal_vocab()
vocab = open("Rules_new.txt", "r").read().splitlines()
vocab = [x.strip() for x in vocab]
vocab_size = len(vocab)

train_dataset = MisuseDataset(vocab=vocab)
train_data = DataLoader(train_dataset, batch_size=1, shuffle=True)

model = NLPLModel(nl_od=300, transfo_xl_model=load_transformer_xl_model(),
                  gnn_ln_in_d=train_dataset.max_node_num*300,
                  gnn_ln_out_d=300, whole_out_d=vocab_size, num_nodes=train_dataset.max_node_num).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=5e-4)
model.train()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

# net_model = Net(train_dataset.node_embed_dim, vocab_size, train_dataset.max_node_num).to(device)
# net_optimizer = torch.optim.Adam(net_model.parameters(), lr=lrate, weight_decay=5e-4)
# net_model.train()

for epoch in range(1, epoch_num+1):
    loss_all = 0
    for x, y, nl_data in train_data:


        model.zero_grad()
        optimizer.zero_grad()
        nl_data = torch.tensor(nl_data)
        out = model(nl_data.to(device), x.to(device))
        loss = loss_fn(out, y.to(device))
        loss_all += loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()


        # # IGNORE: Test code
        # optimizer.zero_grad()
        # out = model(data.to(device))
        # # print(out.shape)
        # # print(y.shape)
        # loss = loss_fn(out, y.to(device))
        # loss_all += loss.item()
        # loss.backward()
        # optimizer.step()


        print("Cur loss: " + str(loss.item()))
        print("Batch loss: " + str(loss_all))

    print("\n" + "="*20)
    print("Batch total loss: " + str(loss_all))
    print("="*20 + "\n")

# print(train_data)


# if __name__ == '__main__':
#     # embed_original_data_from_direct_json()
#     # vocab = load_nonterminal_vocab()
#     # initliaze_dataloader("embed_training_data_small.json", vocab)
#     train()
#     pass
