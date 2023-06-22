import math
import torch
import torch.nn as nn
from torch.nn import Sequential, ModuleList, BatchNorm1d, Linear, ReLU
from torch_geometric.nn import GINEConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch.nn import functional as F
from Pretrain.dataset import GraphFeaturizer
from Prediction.collator import pad_x_unsqueeze

# weight initialization for layers
def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

class AtomEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super(AtomEmbedding, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.atom_feature_dims = GraphFeaturizer().get_atom_feature_dims()
        for i, dim in enumerate(self.atom_feature_dims):
            emb = nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])
        return x_embedding

class BondEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super(BondEmbedding, self).__init__()
        self.bond_embedding_list = torch.nn.ModuleList()
        self.bond_feature_dims = GraphFeaturizer().get_bond_feature_dims()
        for i, dim in enumerate(self.bond_feature_dims):
            emb = nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.bond_embedding_list[i](x[:, i])

        return x_embedding

class GINE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int, train_eps=True):
        super(GINE, self).__init__()
        self.train_eps = train_eps
        self.num_layers = num_layers
        self.convs = ModuleList()
        self.norms = ModuleList()
        self.atom_embed = Linear(in_channels, hidden_channels)
        self.bond_embed = Linear(in_channels, hidden_channels)
        for _ in range(num_layers):
            nn = Sequential(
                Linear(hidden_channels, 2 * hidden_channels),
                BatchNorm1d(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels)
            )
            self.convs.append(GINEConv(nn, self.train_eps))
            self.norms.append(BatchNorm1d(hidden_channels))
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.atom_embed(x)
        edge_attr = self.bond_embed(edge_attr)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.norms[i](x)
            x = F.relu(x)
        if self.num_layers > 1:
            x = self.lin(x)
        return x

class GINet(nn.Module):
    """
    in_channels: dimensionality of atom_embeddings, bond_embeddings
    out_channels: dimensionality of graph_embeddings
    """
    def  __init__(self, num_layers, in_channels, hidden_channels,
                 out_channels, train_eps=True, pool='add'):
        super(GINet, self).__init__()
        self.num_layers = num_layers
        self.in_channel = in_channels
        self.hidden_channel = hidden_channels
        self.out_channel = out_channels
        self.train_eps = train_eps

        self.GINE = GINE(in_channels = self.in_channel,
                         hidden_channels = self.hidden_channel,
                         out_channels = self.out_channel,
                         num_layers = self.num_layers,
                         train_eps = self.train_eps)

        self.atom_embedding = AtomEmbedding(self.in_channel)
        self.bond_embedding = BondEmbedding(self.in_channel)
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool

    def forward(self, data):
        x = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        x = self.atom_embedding(x)
        edge_attr = self.bond_embedding(edge_attr)
        x = self.GINE(x, edge_index, edge_attr)
        x = self.pool(x, data.frag_batch)

        return x

class FragFeature(nn.Module):
    """
    fragment feature extractor -> pre-train model: FragCLR
    output size: [n_frag, n_frag_feature]
    transform -> [n_graph, n_frag, n_frag_feature]
    """
    def __init__(self, num_GNN_layers, in_channels, hidden_channels, out_channels, train_eps, pool):
        super(FragFeature, self).__init__()
        self.GraphEncoder = GINet(num_GNN_layers,
                                  in_channels,
                                  hidden_channels,
                                  out_channels,
                                  train_eps,
                                  pool)

    def forward(self, data):
        x = self.GraphEncoder(data)
        # preprocess
        graph = []
        frag_num = data.frag_num
        slice_begin = 0
        frag_cnt = 0
        for i in range(len(frag_num)):
            slice_end = slice_begin + frag_num[i]
            frag_feat = x[slice_begin:slice_end, :]
            frag_cnt += frag_num[i]
            slice_begin = slice_end
            graph.append(frag_feat)
        assert slice_end == x.size(0)
        max_frag_num = max(g.size(0) for g in graph)
        # [n_graph, n_frag, n_frag_features]
        x = torch.cat([pad_x_unsqueeze(g, max_frag_num) for g in graph])
        return x

class GraphFragFeature(nn.Module):
    """
    Compute fragment features for each fragment in the graph.
    """
    def __init__(self, num_heads, num_in_degree, num_out_degree, in_features, hidden_dim, n_layers):
        super(GraphFragFeature, self).__init__()
        self.num_heads = num_heads
        # 1 for graph token
        self.feat_lin = nn.Linear(in_features, hidden_dim)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, hidden_dim)
        # initate parameters
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, frag_feature, data):
        in_degree, out_degree = data.in_degree, data.out_degree
        # x frag feature
        n_graph, n_frag = frag_feature.size()[:2]
        # frag feature + in degree + out_degree
        frag_feature = (
                self.feat_lin(frag_feature)
                + self.in_degree_encoder(in_degree)
                + self.out_degree_encoder(out_degree)
        )
        #graph_token.weight: [1, n_hidden] --> [1, 1, n_hidden] --> [n_graph, 1, n_hidden]
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        graph_frag_feature = torch.cat([graph_token_feature, frag_feature], dim=1)

        return graph_frag_feature

class GraphAttnBias(nn.Module):
    '''
    Compute attention bias for each head.
    '''
    def __init__(
            self,
            num_heads,
            num_edges,
            num_spatial,
            num_edge_dis,
            edge_type,
            multi_hop_max_dist,
            num_offset,
            n_layers
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist
        self.edge_encoder = nn.Embedding(num_edges * num_offset + 1, num_heads, padding_idx=0)
        self.edge_type = edge_type
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(
                num_edge_dis * num_heads * num_heads, 1
            )
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)
        # initiate parameters
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, frag_feature, data):
        attn_bias, spatial_pos = data.attn_bias, data.spatial_pos
        edge_input, attn_edge_type = data.edge_input, data.attn_edge_type
        n_graph, n_frag = frag_feature.size()[:2]
        #graph_attn_bias
        #add virtual node [VNode] as the whole representation of graph feature
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [n_graph, n_head, n_frag+1, n_frag+1]

        # spatial pos
        # [n_graph, n_frag, n_frag, n_head] -> [n_graph, n_head, n_frag, n_frag]
        # unreachable path:510, ii:0
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # reset spatial pos here
        # [Vnode] is connected to all other nodes in graph, the distance of shortest path is 1 for ([Vnode],v_j)
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        if self.edge_type == "multi_hop":
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, x > 1 to x - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]
            # [n_graph, n_frag, n_frag, max_dist, n_edge_features]
            # [n_graph, n_frag, n_frag, max_dist, n_edge_features, n_head]
            edge_input = self.edge_encoder(edge_input).mean(-2)
            # # [n_graph, n_frag, n_frag, max_dist, n_head]
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
                max_dist, -1, self.num_heads
            )
            # edge_input_flat[max_dist, n_graph, n_frag, n_frag, n_head] --> [max_dist, n_graph*n_frag*n_frag, n_heads]
            # self.edge_dis_encoder.weight [num_edge_dis * n_heads * n_heads, 1] --> [num_edge_dis, n_heads, n_heads]
            # --> [max_dist, n_heads, n_heads] torch.bmm three-dimensional matrix multiply
            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads
                )[:max_dist, :, :],
            )
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_frag, n_frag, self.num_heads
            ).permute(1, 2, 3, 0, 4)
            edge_input = (
                    edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))
            ).permute(0, 3, 1, 2)
        else:
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)

        return graph_attn_bias

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        """
        hidden_size: output of hidden layer
        attention_dropout_rate: dropout rate inside attention for training
        num_heads: number of heads to repeat the same attention structure
        """
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        # split into several heads
        assert hidden_size % num_heads == 0
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5
        # 1/sqrt(d_k)
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        """
        add attn_bias, then softmax and matmul
        """
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        # eq7,softmax and matmul after adding bias
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        # different from original transformer, LayerNorm before self-attention and FFN
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x

class FragTransformer(nn.Module):
    def __init__(
            self,
            task,
            edge_type,
            multi_hop_max_dist,
            num_GNN_layers,
            in_channels,
            hidden_channels,
            out_channels,
            train_eps,
            pool,
            num_in_degree,
            num_out_degree,
            num_edges,
            num_spatial,
            num_edge_dis,
            num_offset,
            num_encoder_layers,
            num_attention_heads,
            embedding_dim,
            dropout_rate,
            input_dropout_rate,
            ffn_dim,
            attention_dropout_rate,
    ):
        super(FragTransformer, self).__init__()
        self.task = task
        self.num_heads = num_attention_heads
        self.frag_feature = FragFeature(num_GNN_layers,
                                        in_channels,
                                        hidden_channels,
                                        out_channels,
                                        train_eps,
                                        pool)
        self.graph_frag_feature = GraphFragFeature(
                                        num_attention_heads,
                                        num_in_degree,
                                        num_out_degree,
                                        out_channels,
                                        embedding_dim,
                                        num_encoder_layers)
        self.graph_attn_bias = GraphAttnBias(
                                        num_attention_heads,
                                        num_edges,
                                        num_spatial,
                                        num_edge_dis,
                                        edge_type,
                                        multi_hop_max_dist,
                                        num_offset,
                                        num_encoder_layers)
        self.input_dropout = nn.Dropout(input_dropout_rate)
        encoders = [EncoderLayer(embedding_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_attention_heads)
                    for _ in range(num_encoder_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(embedding_dim)
        if self.task == 'classification':
            out_dim = 2
        if self.task == 'regression':
            out_dim = 1
        self.downstream_out_proj = nn.Linear(
            embedding_dim, out_dim)
        self.apply(lambda module: init_params(module, n_layers=num_encoder_layers))

    def forward(self, data):
        """
        batched_data:
            attn_bias: [n_graph, n_frag+1, n_frag+1], the shortest path beyond the spatial_pos_max: -inf; else:0
            spatial_pos: [n_graph, n_frag, n_frag], the shortest path between frags in the graph
            x: [n_graph, n_frag, n_frag_features], fragment feature extracted by feature extractor(GIN/GINE)
            in_degree: [n_graph, n_frag]
            out_degree: [n_graph, n_frag]
            edge_input: [n_graph, n_frag, n_frag, multi_hop_max_dist, n_edge_features]
            attn_edge_type: [n_graph,n_frag,n_frag,n_edge_features], edge feature
        """
        frag_feature = self.frag_feature(data)
        graph_frag_feature = self.graph_frag_feature(frag_feature, data)
        attn_bias = self.graph_attn_bias(frag_feature, data)
        # Calculating #
        output = self.input_dropout(graph_frag_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, attn_bias)
        output = self.final_ln(output) # h(l)

        output = self.downstream_out_proj(output[:, 0, :])

        return output

    def load_my_state_dict(self, state_dict, module):
        # FragTransformer
        own_state = self.state_dict()
        # FragCLR
        for name, param in state_dict.items():
            if (module + '.' + name) not in own_state:
                continue
            # print(name)
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[module + '.' + name].copy_(param)


if __name__ == '__main__':
    model = FragTransformer(task='classification',
                            edge_type='multi_hop',
                            multi_hop_max_dist=20,
                            num_encoder_layers=6,
                            num_GNN_layers=3,
                            in_channels=128,
                            hidden_channels=256,
                            out_channels=256,
                            train_eps=True,
                            pool='add',
                            num_in_degree=24,
                            num_out_degree=24,
                            num_edges=128,
                            num_spatial=128,
                            num_edge_dis=128,
                            num_offset=128,
                            num_attention_heads=8,
                            embedding_dim=512,
                            dropout_rate=0.1,
                            input_dropout_rate=0.1,
                            ffn_dim=512,
                            attention_dropout_rate=0.1)
    print(model)
