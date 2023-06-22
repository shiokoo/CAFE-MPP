import torch
from rdkit import Chem
from torch_geometric.data import Data
from Pretrain.dataset import GraphFeaturizer
from Pretrain.utils import get_edgelist
from Pretrain.preprocess import GetCCSingleBonds, GetSingleBonds, GetBRICSBonds

class FragFeaturizer():
    def __init__(self, task, fragmentation):
        super(FragFeaturizer, self).__init__()
        self.task = task
        self.fragmentation = fragmentation
        self.graph_featurizer = GraphFeaturizer()

    def featurize(self, smiles, label):
        mol = Chem.MolFromSmiles(smiles)
        adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.int64)
        x, edge_attr = self.graph_featurizer.featurize(mol)
        x, edge_attr = torch.from_numpy(x), torch.from_numpy(edge_attr)
        edge_index = get_edgelist(mol, bidirection=True)
        edge_index = torch.tensor(edge_index).long()
        if self.fragmentation == 'CCSingleBond':
            single_bond = torch.tensor(GetCCSingleBonds(mol)).long()
        elif self.fragmentation == 'SingleBond':
            single_bond = torch.tensor(GetSingleBonds(mol)).long()
        elif self.fragmentation == 'BRICSBond':
            single_bond = torch.tensor(GetBRICSBonds(mol)).long()
        else:
            raise NotImplementedError
        x, edge_index, edge_attr, JT_edge_attr, adjacency_matrix = self.GetFragments(single_bond, edge_index, edge_attr, x, adjacency_matrix)
        # check whether Adj is symmetric or not.
        if not self.is_symmetric(adjacency_matrix):
            raise ValueError
        # search connected component，also means meaningful subgraph
        seen = set()
        frag_node_list = []
        for node in range(adjacency_matrix.size(0)):
            if node not in seen:
                node_set = self.SearchConnectedComponent(adjacency_matrix, node)
                # print(node_set)
                frag_node_list.append(list(node_set))
                seen.update(node_set)
        frag_num = len(frag_node_list)
        JT_edge_index = self.CreateJunctionTree(single_bond, frag_node_list)
        JT_bond_num = JT_edge_index.size(1) / 2
        if self.task == 'classification':
            y = torch.tensor(label, dtype=torch.long).view(1, -1)
        elif self.task == 'regression':
            y = torch.tensor(label, dtype=torch.float).view(1, -1)
        data = Data(x=x, edge_index = edge_index.t().contiguous(), y = y, edge_attr = edge_attr,
                    JT_edge_index = JT_edge_index, JT_edge_attr = JT_edge_attr, atom_num = torch.Tensor([x.size(0)]).long(),
                    JT_bond_num = torch.Tensor([JT_bond_num]).long(), frag_num = torch.Tensor([frag_num]).long(),
                    frag_node_set = frag_node_list)
        return data

    def SearchConnectedComponent(self, adjacency_matrix, source):
        # BFS
        queue = []
        seen = set()
        seen.add(source)
        queue.insert(0, source)
        while len(queue) > 0:
            this_node = queue.pop(0)
            for i in range(adjacency_matrix.shape[0]):
                if (adjacency_matrix[this_node][i] == 1) & (i not in seen):
                    seen.add(i)
                    queue.append(i)
        return seen

    def GetFragments(self, single_bond, edge_index, edge_attr, x, adjacency_matrix):
        # atom num
        N = x.size(0)
        dummy_num = len(single_bond) * 2
        # total atom num after adding dummy atoms
        M = N + dummy_num
        adjacency_matrix_ = torch.zeros([M, M], dtype=torch.int64)
        adjacency_matrix_[:N, :N] = adjacency_matrix
        dummy_feature = torch.tensor([[118, 1, 4, 0, 5, 0, 0, 0]]).long()
        new_edge_attr = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]]).long()
        # edge_index is not empty!
        # The molecule has more than one atom.
        if min(edge_index.shape) != 0:
            JT_edge_attr = torch.empty((0, 4), dtype=torch.int64)
            if len(single_bond) > 0:
                # remove the correlative edge_index of singlebond
                bond_id = single_bond[:, 0].tolist()
                link_id = single_bond[:, 1:].tolist()
                # update edge_index and adj
                cnt = 0
                while(len(bond_id)):
                    # find the index of singlebond in edge_index
                    assert link_id[0] in edge_index.tolist()
                    edge_idx = edge_index.tolist().index(link_id[0])
                    begin_atom_idx = link_id[0][0]
                    end_atom_idx = link_id[0][1]

                    new_index = torch.tensor([[begin_atom_idx, N+cnt], [N+cnt, begin_atom_idx]]).long()
                    edge_index = torch.cat((edge_index[:edge_idx, :], new_index, edge_index[edge_idx+1:, :]), dim=0)
                    x = torch.cat((x, dummy_feature), dim=0)

                    edge_feature = edge_attr[edge_idx]
                    edge_attr = torch.cat((edge_attr[:edge_idx, :], new_edge_attr, edge_attr[edge_idx+1:, :]), dim=0)
                    JT_edge_attr = torch.cat((JT_edge_attr, edge_feature.unsqueeze(0)), dim=0)

                    # bidirection
                    clone_bond = [end_atom_idx, begin_atom_idx]
                    assert clone_bond in edge_index.tolist()
                    edge_idx = edge_index.tolist().index(clone_bond)
                    new_index = torch.tensor([[N+cnt+1, end_atom_idx], [end_atom_idx, N+cnt+1]]).long()
                    edge_index = torch.cat((edge_index[:edge_idx, :], new_index, edge_index[edge_idx+1:, :]), dim=0)
                    x = torch.cat((x, dummy_feature), dim=0)

                    edge_feature = edge_attr[edge_idx]
                    edge_attr = torch.cat((edge_attr[:edge_idx, :], new_edge_attr, edge_attr[edge_idx+1:, :]), dim=0)
                    JT_edge_attr = torch.cat((JT_edge_attr, edge_feature.unsqueeze(0)), dim=0)

                    # update adjacency matrix
                    adjacency_matrix_[begin_atom_idx, end_atom_idx] = 0
                    adjacency_matrix_[begin_atom_idx, N+cnt] = 1
                    adjacency_matrix_[N+cnt+1, end_atom_idx] = 1

                    adjacency_matrix_[end_atom_idx, begin_atom_idx] = 0
                    adjacency_matrix_[N+cnt, begin_atom_idx] = 1
                    adjacency_matrix_[end_atom_idx, N+cnt+1] = 1

                    cnt += 2
                    bond_id.pop(0)
                    link_id.pop(0)
                assert len(link_id) == 0
                assert len(bond_id) == 0
                assert cnt == dummy_num
            else:
                pass
        else:
            # The molecule has single atom!
            JT_edge_attr = torch.empty((0, 4), dtype=torch.int64)
        return x, edge_index, edge_attr, JT_edge_attr, adjacency_matrix_

    def CreateJunctionTree(self, single_bond, frag_node_list):
        # JT_edge_index
        # There are more than one subgraph.
        if len(single_bond) > 0:
            link_ids = single_bond[:, 1:].tolist()
            JT_edge_index = []

            for link_id in link_ids:
                begin_atom = link_id[0]
                end_atom = link_id[1]
                begin_frag_idx = -1
                end_frag_idx = -1
                for node_list in frag_node_list:
                    if begin_atom in node_list:
                        begin_frag_idx = frag_node_list.index(node_list)
                    elif end_atom in node_list:
                        end_frag_idx = frag_node_list.index(node_list)

                    if (begin_frag_idx != -1) and (end_frag_idx != -1):
                        if begin_frag_idx > end_frag_idx:
                            tmp = begin_frag_idx
                            begin_frag_idx = end_frag_idx
                            end_frag_idx = tmp
                            edge = [end_frag_idx, begin_frag_idx]
                            clone_edge = [begin_frag_idx, end_frag_idx]
                        else:
                            edge = [begin_frag_idx, end_frag_idx]
                            clone_edge = [end_frag_idx, begin_frag_idx]
                assert begin_frag_idx != -1
                assert end_frag_idx != -1
                assert begin_frag_idx < end_frag_idx
                JT_edge_index.append(edge)
                JT_edge_index.append(clone_edge)
            JT_edge_index = torch.Tensor(JT_edge_index).long()
            JT_edge_index= JT_edge_index.t().contiguous()
        else:
            # only one subgraph(self), no singlebond need to cut
            JT_edge_index = torch.empty((2,0), dtype=torch.int64).contiguous()
        return JT_edge_index

    def is_symmetric(self, adjacency_matrix):
        N = adjacency_matrix.size(0)
        square = N * N
        if torch.sum(adjacency_matrix.t() == adjacency_matrix).item() != square:
            return False
        else:
            return True