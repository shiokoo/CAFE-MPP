import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Sequential, ModuleList, BatchNorm1d, Linear, ReLU
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch.autograd import Variable
from torch_geometric.nn import GINEConv
from Pretrain.dataset import GraphFeaturizer

#####for fragment SMILES string######
class TransformerEncoder(nn.Module):
    """
    Only use encoder part of Transformer to extract features from SMILES string.
    parameters:
        --num_layers: number of encoder layers
        --num_heads: number of heads in multi-head attention
        --embed_dim: embedding dimension
        --ffn_hidden: hidden size of feed forward network
        --drop_rate: dropout rate in encoder layer (after multi-head attention and feed forward network)
        --ffn_drop_rate: dropout rate in feed forward network
        --attention_drop_rate: dropout rate in multi-head attention
        --input_drop_rate: dropout rate in input embedding
        --vocab: vocabulary size (for SMILES)
        --max_len: maximum length of input sequence (for SMILES tokenizer)
    """
    def __init__(self, num_layers, num_heads, embed_dim, ffn_hidden, drop_rate,
                 ffn_drop_rate, attention_drop_rate, input_drop_rate, vocab, max_len):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Sequential(nn.Embedding(vocab, embed_dim, padding_idx=0),
                                       PositionEncoding(embed_dim, max_len))
        self.input_drop_rate = nn.Dropout(input_drop_rate)
        self.encoder_layers = \
            nn.ModuleList(
            [TransformerEncoderLayer(embed_dim, ffn_hidden, drop_rate, ffn_drop_rate, attention_drop_rate, num_heads)
            for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(embed_dim)
        self._init_param()

    def forward(self, x, att_bias=None):
        x = self.embedding(x)
        x = self.input_drop_rate(x)
        for layer in self.encoder_layers:
            x = layer(x, att_bias)
        x = self.final_norm(x)
        return x

    def _init_param(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, ffn_hidden, drop_rate, ffn_drop_rate, attention_drop_rate, num_heads):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, attention_drop_rate)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_hidden, ffn_drop_rate)
        self.self_attention_norm = nn.LayerNorm(embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.self_attention_dropout = nn.Dropout(drop_rate)
        self.ffn_dropout = nn.Dropout(drop_rate)

    def forward(self, x, bias=None):
        y = self.self_attention(x, x, x, bias)
        x = x + self.self_attention_dropout(self.self_attention_norm(y)) 
        y = self.ffn(x)
        x = x + self.ffn_dropout(self.ffn_norm(y))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, drop_rate=0):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.att_size = embed_dim // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(embed_dim, self.num_heads * self.att_size)
        self.linear_k = nn.Linear(embed_dim, self.num_heads * self.att_size)
        self.linear_v = nn.Linear(embed_dim, self.num_heads * self.att_size)
        self.dropout = nn.Dropout(drop_rate)
        # spelling check: dropout_layer -> output_layer
        self.dropout_layer = nn.Linear(self.num_heads * self.att_size, embed_dim)

    def forward(self, q, k, v, bias=None):
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_v(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)

        # Compute Scaled Dot Product Attention
        x = torch.matmul(q, k) / math.sqrt(d_k)
        if bias is not None:
            x = x + bias
        x = torch.softmax(x, dim=3)
        x = self.dropout(x)
        x = x.matmul(v)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * d_v)
        x = self.dropout_layer(x)
        return x

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, drop_rate=0):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.layer2 = nn.Linear(ffn_size, hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class PositionEncoding(nn.Module):
    def __init__(self, embed_dim, max_len):
        super(PositionEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x

########for fragment graphs##########
class AtomEmbedding(nn.Module):
    """
    atom embedding layer
    """
    def __init__(self, emb_dim):
        super(AtomEmbedding, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.atom_feature_dims = GraphFeaturizer().get_atom_feature_dims()
        for _, dim in enumerate(self.atom_feature_dims):
            emb = nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])
        return x_embedding

class BondEmbedding(nn.Module):
    """
    bond embedding layer
    """
    def __init__(self, emb_dim):
        super(BondEmbedding, self).__init__()
        self.bond_embedding_list = torch.nn.ModuleList()
        self.bond_feature_dims = GraphFeaturizer().get_bond_feature_dims()
        for _, dim in enumerate(self.bond_feature_dims):
            emb = nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.bond_embedding_list[i](x[:, i])

        return x_embedding

class GINE(nn.Module):
    """
    GINE layer to update atom embeddings
    """
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
    GIN model with edge features to extract features from graphs.
    in_channels: dimensionality of atom embeddings, bond embeddings
    out_channels: dimensionality of graph_embeddings
    """
    def __init__(self, num_layers, in_channels, hidden_channels,
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
        # readout
        x = self.pool(x, data.batch)

        return x

class FragCLR(nn.Module):
    '''
    Pre-training model for generating fragment representations via contrastive learning.
    params:
        num_TF_layers: number of TransformerEncoder layers
        num_GNN_layers: number of GINet layers
    output:
        seq_rep: representations of SMILES strings generated by TransformerEncoder
        graph_rep: representations of fragment graphs generated by GINet
        graph_aug_rep: representations of augmented fragment graphs (hard negative) generated by GINet
        seq_out, graph_out, graph_aug_out: outputs of representations through the projection head
    '''
    def __init__(self, feat_dim, num_TF_layers, num_heads, embed_dim, ffn_hidden, drop_rate, ffn_drop_rate,
                 attention_drop_rate, input_drop_rate, vocab, max_len, num_GNN_layers, in_channels,
                 hidden_channels, out_channels, train_eps=True, pool="add"):
        super(FragCLR, self).__init__()
        self.seq_emb_dim = embed_dim
        self.graph_emb_dim = out_channels
        self.feat_dim = feat_dim
        self.SeqEncoder = TransformerEncoder(num_TF_layers,
                                             num_heads,
                                             embed_dim,
                                             ffn_hidden,
                                             drop_rate,
                                             ffn_drop_rate,
                                             attention_drop_rate,
                                             input_drop_rate,
                                             vocab,
                                             max_len)

        self.GraphEncoder = GINet(num_GNN_layers,
                                  in_channels,
                                  hidden_channels,
                                  out_channels,
                                  train_eps,
                                  pool)

        self.seq_feat_lin = nn.Linear(self.seq_emb_dim, self.feat_dim)
        self.graph_feat_lin = nn.Linear(self.graph_emb_dim, self.feat_dim)

        self.seq_projection = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim//2)
        )
        self.graph_projection = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim//2)
        )

    def forward(self, seq_frag, graph_frag, graph_frag_transform):
        seq_emb = self.SeqEncoder(seq_frag)
        seq_emb = seq_emb.sum(-2)
        seq_rep = self.seq_feat_lin(seq_emb)
        seq_out = self.seq_projection(seq_rep)

        graph_emb = self.GraphEncoder(graph_frag)
        graph_rep = self.graph_feat_lin(graph_emb)
        graph_out = self.graph_projection(graph_rep)

        graph_aug_emb = self.GraphEncoder(graph_frag_transform)
        graph_aug_rep = self.graph_feat_lin(graph_aug_emb)
        graph_aug_out = self.graph_projection(graph_aug_rep)

        return seq_rep, graph_rep, graph_aug_rep, seq_out, graph_out, graph_aug_out

if __name__ == '__main__':
    model = FragCLR(feat_dim=256,
                    num_TF_layers=6,
                    num_heads=8,
                    embed_dim=512,
                    ffn_hidden=512,
                    drop_rate=0.1,
                    ffn_drop_rate=0.1,
                    attention_drop_rate=0.1,
                    input_drop_rate=0.1,
                    vocab=128,
                    max_len=256,
                    num_GNN_layers=3,
                    in_channels=128,
                    hidden_channels=256,
                    out_channels=256,
                    train_eps=True,
                    pool="add")
    print(model)