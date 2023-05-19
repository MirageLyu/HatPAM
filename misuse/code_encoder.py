import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn.conv.gated_graph_conv import GatedGraphConv
from torch_geometric.nn.glob import GlobalAttention
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import DataLoader as gnn_DataLoader
from torch.utils.data import DataLoader
from torch.nn import functional as F
import json
from gensim.models import Word2Vec
from tqdm import tqdm
from natural_encoder import load_transformer_xl_model, load_transformer_xl_tokenizer, NaturalLanguageEmbeddingLayer
from dataset import EncoderDataset


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available')
    print("Device name: " + torch.cuda.get_device_name(0))
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device('cpu')


# def initliaze_dataloader(data_path, vocab):
#     training_data = json.load(open("embed_training_data_small.json", "r"))["embed_data"]
#     DataList = []
#     for data in training_data:
#         x = torch.tensor(data["nodes"])
#         edges_in = data["edges_in"]
#         edges_out = data["edges_out"]
#         edges_index = torch.tensor([edges_in, edges_out], dtype=torch.long)
#
#         label = data["node"]
#         if len(label) > 12 and label[-12:] == "_fu_nc_na_me":
#             label = "Token"
#         y = vocab.index(label)
#         DataList.append(Data(x, edges_index, None, y))
#     return gnn_DataLoader(DataList, batch_size=1)
#
# class TreeData:
#     def __init__(self, nodes, edges):
#         """
#         Initialize ast nodes and edges set
#         :param nodes: a list of node
#         :param edges: [[],[]] two lists that one from-node list and one to-node list
#         """
#         self.nodes = nodes
#         self.edges = edges
#         self.embed_nodes = []
#
#         # tree-edge weight and data-flow path weight
#         self.TREE_EDGE_WEIGHT = 1
#         self.DATAFLOW_PATH_WEIGHT = 1
#
#         # do embedding flag
#         self.is_embed = False
#
#     def add_nodes(self, nodes: [int]):
#         self.nodes += nodes
#
#     def add_edges(self, edges):
#         self.edges += edges
#
#     def do_embedding(self, vocab):
#         if self.is_embed:
#             return
#         if not WORD2VEC_MODEL_TRAINED:
#             # TODO: construct vocab from python syntax rules? or from existing AST nodes.
#             # TODO: Use extracted rules by TreeGen/GrammarCNN
#             vocab = ['<unk_token>', '<pad_token>']
#             train_word2vec_model(vocab, 1, "w2v.model")
#         w2v_model = load_word2vec_model("w2v.model")
#         for node in self.nodes:
#             # TODO: implement node's data structure
#             if w2v_model.wv.has_index_for(node.name):
#                 self.embed_nodes.append(w2v_model.wv[node.name])
#             else:
#                 self.embed_nodes.append(w2v_model.wv['<unk_token>'])
#
#         # init word2vec model here
#         # write to embed_nodes
#
#     def traversal_sequence(self):
#         """
#         :return: return pre-order traversal sequence
#         """
#         pass
#
#     def do_positional_encoding(self):
#         # TODO: do scaled, normalized, stack-like positional encoding here.
#         if self._positional_encoding is None:
#             if self.parent:
#                 self._positional_encoding = [
#                     0.0 for _ in range(self.parent.num_children())]
#                 self._positional_encoding[self.branch] = 1.0
#                 self._positional_encoding += self.parent.get_positional_encoding()
#             else:
#                 self._positional_encoding = []
#         return self._positional_encoding
#
#
# def load_nonterminal_vocab():
#     vocab = list(json.load(open("vocab.json", "r")).keys())
#     return vocab
#
#
# class TreeBasedAttentionLayer(nn.Module):
#     def __init__(self):
#         super(TreeBasedAttentionLayer, self).__init__()
#         # simply do self-attention on tree-traversal sequence
#
#     def forward(self, tree_data: TreeData):
#         pass

#
# def sequence_to_padding(x, length):
#     # declare the shape, it can work for x of any shape.
#     ret_tensor = torch.zeros(length)
#     ret_tensor[:x.shape[-1]] = x
#     return ret_tensor


class GatedGSNN(nn.Module):
    def __init__(self, num_nodes, hidden_channels=200, out_channels=300):
        super(GatedGSNN, self).__init__()
        self.gnn_layer1 = GatedGraphConv(hidden_channels, num_nodes)
        self.gnn_layer2 = GatedGraphConv(out_channels, num_nodes)
        # Optional Linear Layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # print(x)
        # print(x.shape)
        # print(edge_index)
        # print(x.shape)
        # print(edge_index.shape)

        x = F.relu(self.gnn_layer1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.gnn_layer2(x, edge_index)

        # x = self.gnn_layer1(x, edge_index)

        # Here, x is a sequence with length equal with the number of nodes.

        return x


class Net(nn.Module):
    def __init__(self, num_node_features, num_classes, num_nodes):
        super(Net, self).__init__()
        self.conv1 = gnn.GatedGraphConv(300, num_nodes)
        self.conv2 = gnn.GatedGraphConv(num_classes, num_nodes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # print(x.shape)
        # print(edge_index.shape)

        x = self.conv1(x, edge_index)
        # print(x.shape)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = torch.mean(x, dim=0)

        return F.log_softmax(x).view(1, x.shape[-1])


class CodeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.attention_layer = TreeBasedAttentionLayer()
        self.ggs_nn = GatedGSNN()

    def forward(self, data):
        """
        :param df_paths: dataflow paths for current partial tree
        :param input_tree: A tree structure data, with node embedding has already been applied.
        :return:
        """
        # input_tree_with_pe = input_tree.get_positional_encoding()
        # input_tree_with_pe.add_tree(input_tree)
        # attn_input_tree_pe = self.attention_layer(input_tree_with_pe)
        # attn_input_tree_pe.add_tree(input_tree)
        # attn_input_tree_pe.add_dataflow_paths(df_paths)
        # graph_sequence = self.ggs_nn(attn_input_tree_pe)
        output = self.ggs_nn(data)

        return output
class NLPLModel(nn.Module):
    def __init__(self, nl_od, transfo_xl_model, gnn_ln_in_d, gnn_ln_out_d, whole_out_d, num_nodes):
        super().__init__()
        self.nl_layer = NaturalLanguageEmbeddingLayer(transfoxl_model=transfo_xl_model, out_dimension=nl_od).to(device)
        self.gnn_layer = GatedGSNN(num_nodes=num_nodes).to(device)
        self.gnn_linear = nn.Linear(gnn_ln_in_d, gnn_ln_out_d)
        self.linear2 = nn.Linear(nl_od + gnn_ln_out_d, whole_out_d)
        # self.linear2 = nn.Linear(gnn_ln_out_d, whole_out_d)

    def forward(self, nl_input_ids, data):
        nl_output = self.nl_layer(nl_input_ids)
        pl_output = self.gnn_layer(data)
        # print(pl_output.shape)
        pl_output = torch.mean(pl_output, dim=0)
        # pl_output = self.gnn_linear(torch.flatten(pl_output))
        # pl_output = F.relu(pl_output)
        # print(pl_output.shape)
        nl_output = nl_output.view(nl_output.shape[-1])
        out_concat = torch.cat((nl_output, pl_output), 0)
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

train_dataset = EncoderDataset(vocab=vocab)
train_data = DataLoader(train_dataset, batch_size=1, shuffle=True)

# model = NLPLModel(nl_od=300, transfo_xl_model=load_transformer_xl_model(),
#                   gnn_ln_in_d=train_dataset.max_node_num*300,
#                   gnn_ln_out_d=300, whole_out_d=vocab_size, num_nodes=train_dataset.max_node_num).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=5e-4)
# model.train()
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

net_model = Net(train_dataset.node_embed_dim, vocab_size, train_dataset.max_node_num).to(device)
net_optimizer = torch.optim.Adam(net_model.parameters(), lr=lrate, weight_decay=5e-4)
net_model.train()

for epoch in range(1, epoch_num+1):
    loss_all = 0
    for x, edge, y, nl_data in train_data:
        x = x[0, :, :]
        edge = edge[0, :, :]
        data = Data(x, edge, None, y)


        # model.zero_grad()
        # optimizer.zero_grad()
        # nl_data = torch.tensor(nl_data)
        # out = model(nl_data.to(device), data.to(device))
        # loss = loss_fn(out, y.to(device))
        # loss_all += loss.item()
        # loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # optimizer.step()
        # scheduler.step()


        # IGNORE: Test code
        net_optimizer.zero_grad()
        out = net_model(data.to(device))
        # print(out.shape)
        # print(y.shape)
        loss = loss_fn(out, y.to(device))
        loss_all += loss.item()
        loss.backward()
        net_optimizer.step()


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
