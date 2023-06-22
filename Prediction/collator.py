import torch

def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_x_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros(
            [padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)

def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)

def create_frag_batch(items):
    # according to frag_node_list and atom_num, reset batch
    atom_nums = torch.zeros(len(items), dtype=torch.int64)
    for i, item in enumerate(items):
        atom_nums[i] = item.x.size(0)
    frag_nums = torch.cat([item.frag_num for item in items])
    frag_node_list = [item.frag_node_set for item in items]
    MolNum = len(atom_nums)
    mol_batch = torch.Tensor([])
    batch_frag_cnt = 0
    for i in range(MolNum):
        frag_cnt = 0
        tmp = torch.Tensor([batch_frag_cnt]).repeat(atom_nums[i].item())
        mol_frag_node_list = frag_node_list[i]
        # print(mol_frag_node_list)
        for node_set in mol_frag_node_list:
            # print(node_set)
            if len(node_set) > 0:
                for node in node_set:
                    tmp[node] = batch_frag_cnt
            frag_cnt += 1
            batch_frag_cnt += 1
        assert frag_cnt == frag_nums[i].item()
        mol_batch = torch.cat([mol_batch, tmp]).long()
    assert len(mol_batch) == torch.sum(atom_nums)
    assert batch_frag_cnt == torch.sum(frag_nums)
    return mol_batch

def batch_edge_index(items):
    """
    batch edge_index according to atom_num of each graph
    """
    atom_nums = torch.zeros(len(items), dtype=torch.int64)
    for i, item in enumerate(items):
        atom_nums[i] = item.x.size(0)
    edge_index_batch = torch.empty((2, 0), dtype=torch.int64)
    offset = 0
    for i, item in enumerate(items):
        edge_index_batch = torch.cat([edge_index_batch, item.edge_index + offset], dim=1)
        offset += atom_nums[i]
    return edge_index_batch

class Batch():
    def __init__(self, idx, x, edge_index, edge_attr, attn_bias, attn_edge_type, spatial_pos,
                 in_degree, out_degree, edge_input, frag_batch, frag_num, y):
        super(Batch, self).__init__()
        self.idx = idx
        self.in_degree, self.out_degree = in_degree, out_degree
        self.x, self.y = x, y
        self.edge_index, self.edge_attr = edge_index, edge_attr
        self.attn_bias, self.attn_edge_type, self.spatial_pos = attn_bias, attn_edge_type, spatial_pos
        self.edge_input = edge_input
        self.frag_batch = frag_batch
        self.frag_num = frag_num

    def to(self, device):
        self.idx = self.idx.to(device)
        self.in_degree, self.out_degree = self.in_degree.to(
            device), self.out_degree.to(device)
        self.x, self.y = self.x.to(device), self.y.to(device)
        self.edge_index, self.edge_attr = self.edge_index.to(device), self.edge_attr.to(device)
        self.attn_bias, self.attn_edge_type, self.spatial_pos = self.attn_bias.to(
            device), self.attn_edge_type.to(device), self.spatial_pos.to(device)
        self.edge_input = self.edge_input.to(device)
        self.frag_batch = self.frag_batch.to(device)
        self.frag_num = self.frag_num.to(device)
        return self

    def __len__(self):
        return self.in_degree.size(0)

def collator(items, max_node=256, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [
        item for item in items if item is not None and item.x.size(0) <= max_node]
    frag_batch = create_frag_batch(items)
    edge_index = batch_edge_index(items)
    items = [(item.idx, item.x, item.edge_attr, item.attn_bias, item.attn_edge_type,
              item.spatial_pos, item.in_degree, item.out_degree,
              item.edge_input[:, :, :multi_hop_max_dist, :], item.frag_num, item.y) for item in items]
    idxs, xs, edge_attrs, attn_biases, attn_edge_types, spatial_poses, \
    in_degrees, out_degrees, edge_inputs, frag_nums, ys = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
    max_dist = max(i.size(-2) for i in edge_inputs)

    y = torch.cat(ys)
    x = torch.cat([i for i in xs])
    edge_attr = torch.cat([i for i in edge_attrs])
    frag_num = torch.cat(frag_nums)
    max_frag_num = max(frag_num).item()
    edge_input = torch.cat([pad_3d_unsqueeze(
        i, max_frag_num, max_frag_num, max_dist) for i in edge_inputs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(
        i, max_frag_num + 1) for i in attn_biases])
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_frag_num) for i in attn_edge_types])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_frag_num)
                             for i in spatial_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_frag_num)
                           for i in in_degrees])
    out_degree = torch.cat([pad_1d_unsqueeze(i, max_frag_num)
                            for i in out_degrees])

    # generate batch_data
    return Batch(
        idx=torch.LongTensor(idxs),
        x = x,
        edge_index = edge_index,
        edge_attr = edge_attr,
        attn_bias = attn_bias,
        attn_edge_type = attn_edge_type,
        spatial_pos = spatial_pos,
        in_degree = in_degree,
        out_degree = out_degree,
        edge_input = edge_input,
        frag_batch = frag_batch,
        frag_num = frag_num,
        y = y
    )