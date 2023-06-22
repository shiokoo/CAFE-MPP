import math
import random
import torch
import numpy as np
from copy import deepcopy
from rdkit import Chem
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from Pretrain.tokenizer import SmilesTokenizer, build_vocub
from Pretrain.utils import *

def filter(data_path):
    """
    filter out the fragment with only one dummy node and no linkable atoms
    """
    smiles_data = []
    with open(data_path, 'r') as f:
        for line in f.readlines():
            smiles = line.strip('\n')
            # filter
            frag = Chem.MolFromSmiles(smiles)
            N, dummy_list = get_dummy_list(frag)
            link_list = get_link_list(frag, dummy_list)
            link_atom_list = get_link_atom_list(frag) # linkable atoms
            if (N == 1) and (len(link_list) == 0) and (len(link_atom_list) == 0):
                continue
            else:
                pass
            smiles_data.append(smiles)
            print(len(smiles_data))
        with open(data_path, 'w') as fp:
            [fp.write(item + '\n') for item in smiles_data]
            fp.close()
    return smiles_data

def parse_data(data_path):
    """
    return a list of SMILES
    """
    smiles_data = []
    with open(data_path, 'r') as f:
        for line in f.readlines():
            smiles = line.strip('\n')
            smiles_data.append(smiles)
    return smiles_data

def safe_index(l, e):
    try:
        return l.index(e)
    except:
        return len(l)-1

class GraphFeaturizer():
    def __init__(self):
        self.feature_list = {
            # atomic: dummy node * -> 118
            'possible_atomic_symbol': list(range(1, 119)) + ['misc'],
            'possible_degree': [0, 1, 2, 3, 4, 5, 'misc'],
            'possible_formal_charge': [-1, -2, 1, 2, 0, 'misc'],
            'possible_radical_electrons': [0, 1, 2, 3, 4, 'misc'],
            'possible_chirality': [
                'CHI_UNSPECIFIED',
                'CHI_TETRAHEDRAL_CW',
                'CHI_TETRAHEDRAL_CCW',
                'CHI_OTHER',
                'misc'
            ],
            'possible_num_Implicit_Hs': [0, 1, 2, 3, 4, 'misc'],
            'possible_hybridization': [
                'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
            ],
            'possible_is_aromatic': [False, True],
            'possible_bond_type': [
                'SINGLE',
                'DOUBLE',
                'TRIPLE',
                'AROMATIC',
                'misc'
            ],
            'possible_bond_stereo': [
                'STEREONONE',
                'STEREOANY',
                'STEREOZ',
                'STEREOE',
                'misc'
            ],
            'possible_is_conjugated': [False, True],
            'possible_is_in_ring': [False, True]
        }

    def featurize(self, mol):
        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(self.atom_feature_to_vector(atom))
        x = np.array(atom_features_list, dtype = np.int64)

        # bonds
        num_bond_features = 4  # bond type, bond stereo, is_conjugated, is_in_ring
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edge_features_list = []
            for bond in mol.GetBonds():
                edge_feature = self.bond_feature_to_vector(bond)
                # add edges in both directions
                edge_features_list.append(edge_feature)
                edge_features_list.append(edge_feature)
            # edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype = np.int64)
        else:   # mol has no bonds
            edge_attr = np.empty((0, num_bond_features), dtype = np.int64)
        return x, edge_attr

    def atom_feature_to_vector(self, atom):
        atom_feature = [
            safe_index(self.feature_list['possible_atomic_symbol'], atom.GetAtomicNum()),
            safe_index(self.feature_list['possible_degree'], atom.GetDegree()),
            safe_index(self.feature_list['possible_formal_charge'], atom.GetFormalCharge()),
            safe_index(self.feature_list['possible_radical_electrons'], atom.GetNumRadicalElectrons()),
            safe_index(self.feature_list['possible_hybridization'], str(atom.GetHybridization())),
            self.feature_list['possible_is_aromatic'].index(atom.GetIsAromatic()),
            safe_index(self.feature_list['possible_num_Implicit_Hs'], atom.GetNumImplicitHs()),
            safe_index(self.feature_list['possible_chirality'], str(atom.GetChiralTag())),
        ]
        return atom_feature

    def bond_feature_to_vector(self, bond):
        bond_feature = [
            safe_index(self.feature_list['possible_bond_type'], str(bond.GetBondType())),
            self.feature_list['possible_is_conjugated'].index(bond.GetIsConjugated()),
            self.feature_list['possible_is_in_ring'].index(bond.IsInRing()),
            safe_index(self.feature_list['possible_bond_stereo'], str(bond.GetStereo())),
        ]
        return bond_feature

    def get_atom_feature_dims(self):
        return list(map(len, [
            self.feature_list['possible_atomic_symbol'],
            self.feature_list['possible_degree'],
            self.feature_list['possible_formal_charge'],
            self.feature_list['possible_radical_electrons'],
            self.feature_list['possible_hybridization'],
            self.feature_list['possible_is_aromatic'],
            self.feature_list['possible_num_Implicit_Hs'],
            self.feature_list['possible_chirality']
        ]))

    def get_bond_feature_dims(self):
        return list(map(len, [
            self.feature_list['possible_bond_type'],
            self.feature_list['possible_is_conjugated'],
            self.feature_list['possible_is_in_ring'],
            self.feature_list['possible_bond_stereo']
        ]))

class PretrainDataset(Dataset):
    def __init__(self, data_path, max_len):
        super(PretrainDataset, self).__init__()
        self.data = parse_data(data_path)
        self.smiles_vocub = build_vocub(self.data)
        self.smiles_tokenizer = SmilesTokenizer(self.smiles_vocub, max_len)
        self.graph_featurizer = GraphFeaturizer()

    def __getitem__(self, index):
        frag = Chem.MolFromSmiles(self.data[index])
        # generate positive and negative samples
        # smiles view
        seq_ = self.smiles_tokenizer.tokenize(self.data[index])
        seq_frag = Data(x=seq_)
        # query
        # featurize
        x, edge_attr = self.graph_featurizer.featurize(frag)
        x, edge_attr = torch.from_numpy(x), torch.from_numpy(edge_attr)
        edge_index = get_edgelist(frag, bidirection=True)
        edge_index = torch.tensor(edge_index).long().t().contiguous()

        graph_frag = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        # augmentation, construct negative samples
        # N: number of dummy node *
        # dummy_list: indices of dummy node *
        # link_list: indices of the other node except dummy node * and the node linked to dummy node * (linkable node)
        # transform_edge_i: indices of bond can be transformed (link to dummy node * in the original fragments）
        # randomly link to the other atoms
        N, dummy_list = get_dummy_list(frag)
        M = frag.GetNumBonds()
        # return a list
        transform_mode = random.sample(list(range(3)), 1)
        mode_list = list(range(3))
        while(mode_list):
            try:
                if transform_mode == [0]:
                    # select several dummy node, change the position
                    # linkable nodes
                    link_list = get_link_list(frag, dummy_list)
                    if link_list:
                        num_transform_nodes = max([1, math.floor(0.25*N)])
                        if len(link_list) < num_transform_nodes:
                            num_transform_nodes = max([1, math.floor(0.10*N)])
                            if len(link_list) < num_transform_nodes:
                                num_transform_nodes = 1
                    else:
                        # There is no qualified atom to link.
                        mode_list.remove(0)
                        raise ValueError

                    # dummy node *
                    transform_nodes_i = random.sample(dummy_list, num_transform_nodes)
                    # target node
                    transform_nodes_j = random.sample(link_list, num_transform_nodes)
                    # obtain the index of bond which links to *
                    transform_edge_single_i = get_dummy_link(frag, transform_nodes_i)
                    transform_edge_i = [2*i for i in transform_edge_single_i] + [2*i+1 for i in transform_edge_single_i]
                    # new edge
                    transform_edge_j = []
                    while(transform_nodes_i and transform_nodes_j):
                        transform_edge_j.append([transform_nodes_i[0], transform_nodes_j[0]])
                        transform_edge_j.append([transform_nodes_j[0], transform_nodes_i[0]])
                        transform_nodes_i.pop(0)
                        transform_nodes_j.pop(0)
                    transform_edge_j = torch.tensor(transform_edge_j).long().t()
                    assert len(transform_edge_i) == transform_edge_j.size(-1)
                    x_transform = deepcopy(x)
                    edge_attr_transform = deepcopy(edge_attr)
                    edge_index_transform = deepcopy(edge_index)

                    for atom_idx in transform_nodes_j:
                        # modify degree and Implicit_Hs
                        # simple operation: degree + 1, Implicit_Hs -1
                        x_transform[atom_idx, :] = x_transform[atom_idx, :] + torch.tensor([0,1,0,0,0,0,-1,0])
                    # print(f"edge_index_transform:{edge_index_transform}")
                    # print(f"transform_edge_j:{transform_edge_j}")
                    count = 0
                    for bond_idx in transform_edge_i:
                        edge_index_transform[:, bond_idx] = transform_edge_j[:, count]
                        count += 1
                    graph_frag_transform = Data(x=x_transform, edge_index=edge_index_transform, edge_attr=edge_attr_transform)
                    break
                elif transform_mode == [1]:
                    # change the number of dummy node *
                    # add node *, at the same time, add one bond
                    # select one atom to link the new dummy node *
                    link_atom_list = get_link_atom_list(frag)
                    if link_atom_list:
                        add_node = random.sample(link_atom_list, 1)[0]
                        edge_index_transform = deepcopy(edge_index)
                        edge_attr_transform = deepcopy(edge_attr)
                        x_transform = deepcopy(x)
                        dummy_feature = torch.tensor([[118,1,4,0,5,0,0,0]]).long()
                        x_transform = torch.cat((x_transform, dummy_feature), dim=0)
                        bond_feature = torch.tensor([[0,0,0,0],[0,0,0,0]]).long()
                        edge_attr_transform = torch.cat((edge_attr_transform, bond_feature), dim=0)
                        new_edge_index = torch.tensor([[add_node, x.shape[0]],[x.shape[0],add_node]]).long()
                        edge_index_transform = torch.cat((edge_index_transform, new_edge_index), dim=1)
                        graph_frag_transform = Data(x=x_transform, edge_index=edge_index_transform, edge_attr=edge_attr_transform)
                        break
                    else:
                        # There is no atom to link the new dummy node *.
                        mode_list.remove(1)
                        raise ValueError
                elif transform_mode == [2]:
                    # change the number of dummy node *
                    # remove node *
                    # remove the bond between dummy node * and its neighbor
                    # There is only one dummy node*, after removing *, the fragment will become a molecule.
                    if N == 1:
                        mode_list.remove(2)
                        raise ValueError
                    # return a list
                    remove_node = random.sample(dummy_list, 1)
                    remove_edge = get_dummy_link(frag, remove_node)[0] * 2
                    remove_node = remove_node[0]
                    edge_index_transform = torch.zeros((2, 2*(M-1)), dtype=torch.long)
                    edge_attr_transform = torch.zeros((2*(M-1), 4), dtype=torch.long)
                    x_transform = deepcopy(x)
                    x_transform = x_transform[torch.arange(x_transform.size(0)) != remove_node]
                    edge_index_transform[:, :remove_edge] = edge_index[:, :remove_edge]
                    edge_index_transform[:, remove_edge:] = edge_index[:, (remove_edge+2):]
                    edge_index_transform[edge_index_transform > remove_node] = edge_index_transform[edge_index_transform > remove_node] - 1

                    edge_attr_transform[:remove_edge, :] = edge_attr[:remove_edge, :]
                    edge_attr_transform[remove_edge:, :] = edge_attr[(remove_edge+2):, :]

                    graph_frag_transform = Data(x=x_transform, edge_index=edge_index_transform, edge_attr=edge_attr_transform)
                    break
            except ValueError:
                transform_mode = random.sample(mode_list, 1)
                if len(mode_list) == 0:
                    print(self.data[index])
                    raise RuntimeError
            else:
                pass
        return seq_frag, graph_frag, graph_frag_transform

    def __len__(self):
        return len(self.data)

class PretrainDatasetWrapper(object):
    def __init__(self, dataset, batch_size, valid_rate, num_workers):
        super(PretrainDatasetWrapper, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.valid_rate = valid_rate
        self.num_workers = num_workers

    def get_data_loaders(self):
        total_num = len(self.dataset)
        indices = np.arange(total_num)
        np.random.shuffle(indices)
        valid_num = int(np.floor(self.valid_rate * total_num))
        train_idx, valid_idx = indices[valid_num:], indices[:valid_num]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        valid_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader

if __name__ == '__main__':
    filter(data_path='../Data/fragments_ccsinglebond.txt')
    dataset = PretrainDataset(data_path='../Data/fragments_ccsinglebond.txt', max_len=256)
    print(f"len(dataset):{len(dataset)}")
    data_wrapper = PretrainDatasetWrapper(dataset=dataset, batch_size=256, valid_rate=0.05, num_workers=8)
    train_loader, valid_loader = data_wrapper.get_data_loaders()
    print(f"len(train_loader):{len(train_loader)}")
    print(f"len(valid_loader):{len(valid_loader)}")
    for i, (seq_frag, graph_frag, graph_aug) in enumerate(train_loader):
        print(i)


