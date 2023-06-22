import random
import torch
import csv
import os
import numpy as np
import logging
from rdkit import Chem
from functools import partial
from collections import defaultdict
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from Prediction import algos
from Prediction.collator import collator
from Prediction.utils import FragFeaturizer
from torch.utils.data import Dataset, Subset, DataLoader

MAX_ATOM_NUM = 100

def parse_data(dataset, target, task, logger):
    data_path = os.path.join('../Data', dataset + '.csv')
    smiles_data, labels = [], []
    with open(data_path) as f:
        reader = csv.DictReader(f, delimiter=',')
        for i, row in enumerate(reader):
            if dataset == 'bace':
                smiles = row['mol']
            else:
                smiles = row['smiles']
            label = row[target]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning('(Discard) The mol object of {}th molecule is not exist.'.format(i))
                continue
            if mol.GetNumAtoms() > MAX_ATOM_NUM:
                logger.warning('(Discard) The {}th molecule has {} atoms.'.format(i, mol.GetNumAtoms()))
                continue
            if label == '':
                continue
            smiles_data.append(smiles)
            if task == 'classification':
                labels.append(int(label))
            elif task == 'regression':
                labels.append(float(label))
            else:
                raise ValueError('Task must be either classification or regression.')
    print(f'dataset: {len(smiles_data)} molecules')
    return smiles_data, labels

def get_target_name(dataset):
    data_path = os.path.join('../Data', dataset + '.csv')
    if dataset == 'bbbp':
        target_list = ['p_np']
    elif dataset == 'bace':
        target_list = ['Class']
    elif dataset == 'hiv':
        target_list = ['HIV_active']
    elif dataset == 'freesolv':
        target_list = ['expt']
    elif dataset == 'esol':
        target_list = ['measured log solubility in mols per litre']
    elif dataset == 'lipop':
        target_list = ['exp']
    elif dataset == 'malaria':
        target_list = ['activity']
    elif dataset == 'cep':
        target_list = ['PCE']
    elif dataset == 'qm7':
        target_list = ['u0_atom']
    elif dataset in ['sider', 'toxcast', 'clintox', 'qm8']:
        with open(data_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            header.remove('smiles')
            target_list = header
            f.close()
    elif dataset in ['muv', 'tox21']:
        with open(data_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            header.remove('smiles')
            header.remove('mol_id')
            target_list = header
            f.close()
    else:
        raise ValueError(f'There is no dataset named {dataset} here.')
    return target_list

def get_task_name(dataset):
    if dataset in ['bbbp', 'bace', 'hiv', 'clintox', 'tox21', 'sider', 'muv', 'toxcast']:
        task = 'classification'
    elif dataset in['freesolv', 'esol', 'lipop', 'cep', 'malaria', 'qm7', 'qm8']:
        task = 'regression'
    else:
        raise ValueError(f'There is no dataset named {dataset} here.')
    return task

def generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

def scaffold_to_ids(dataset):
    scaffolds = defaultdict(set)
    for i, smiles in enumerate(dataset):
        scaffold = generate_scaffold(smiles)
        scaffolds[scaffold].add(i)
    return scaffolds

def scaffold_split(dataset, valid_rate, test_rate, seed, balanced=False, logger=None):
    train_ids, valid_ids, test_ids = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    total_num = len(dataset)
    train_size = (1 - valid_rate - test_rate) * total_num
    valid_size = valid_rate * total_num
    test_size = test_rate * total_num

    scaffold_to_indices = scaffold_to_ids(dataset)
    if balanced:
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > valid_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:
        # sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)
    for index_set in index_sets:
        if len(train_ids) + len(index_set) <= train_size:
            train_ids += index_set
            train_scaffold_count += 1
        elif len(valid_ids) + len(index_set) <= valid_size:
            valid_ids += index_set
            val_scaffold_count += 1
        else:
            test_ids += index_set
            test_scaffold_count += 1

    if logger is not None:
        logger.info(f'Total scaffolds = {len(scaffold_to_indices):,} | '
                     f'train scaffolds = {train_scaffold_count:,} | '
                     f'val scaffolds = {val_scaffold_count:,} | '
                     f'test scaffolds = {test_scaffold_count:,}')

    return train_ids, valid_ids, test_ids

def split(dataset, split_type, valid_rate, test_rate, seed, logger):
    if split_type == 'random':
        total_num = len(dataset)
        indices = np.arange(total_num)

        np.random.seed(seed)
        np.random.shuffle(indices)

        valid_size = int(np.floor(valid_rate * total_num))
        test_size = int(np.floor(test_rate * total_num))
        test_idx, valid_idx, train_idx = indices[:test_size], indices[test_size:test_size + valid_size], indices[test_size + valid_size:]

    elif split_type == 'scaffold':
        train_idx, valid_idx, test_idx = scaffold_split(dataset, valid_rate, test_rate, seed, balanced=True, logger=logger)
    else:
        raise ValueError(f'split_type"{split_type} not supported."')

    return train_idx, valid_idx, test_idx

class PredictionDataset(Dataset):
    def __init__(self, dataset, target, task, fragmentation, logger):
        super(PredictionDataset, self).__init__()
        self.logger = logger
        self.data, self.labels = parse_data(dataset, target, task, self.logger)
        self.frag_featurizer = FragFeaturizer(task, fragmentation)

    def __getitem__(self, index):
        item = self.frag_featurizer.featurize(self.data[index], self.labels[index])
        JT_edge_index, JT_edge_attr = item.JT_edge_index, item.JT_edge_attr
        frag_num = item.frag_num.item()
        N = frag_num
        # adjacency matrix [N, N] bool for JT
        JT_adj = torch.zeros([N, N], dtype=torch.bool)
        JT_adj[JT_edge_index[0, :], JT_edge_index[1, :]] = True
        JT_edge_attr = JT_edge_attr.long()
        # edge feature
        if len(JT_edge_attr.size()) == 1:
            JT_edge_attr = JT_edge_attr[:, None]
        # [n_frag, n_frag, n_edge_features]
        # edge_feature for whole graph
        attn_edge_type = torch.zeros([N, N, JT_edge_attr.size(-1)], dtype=torch.int64)
        # Other [i,j], for no edge between i,j, the attr remains zeros.
        attn_edge_type[JT_edge_index[0, :], JT_edge_index[1, :]
        ] = JT_edge_attr + 1
        #
        # floyd algorithm, calculate the shortest path between any frag i and j in the graph.
        shortest_path_result, path = algos.floyd_warshall(JT_adj.numpy())
        # the max dist between i and j in whole graph
        max_dist = np.amax(shortest_path_result)
        # feature of the n-edge in shortest path
        edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
        # length of shortest path between frag i and j in the graph [N,N]
        spatial_pos = torch.from_numpy((shortest_path_result)).long()
        attn_bias = torch.zeros(
            [N + 1, N + 1], dtype=torch.float)  # with graph token
        # combine
        item.idx = index
        item.attn_bias = attn_bias
        item.attn_edge_type = attn_edge_type
        item.spatial_pos = spatial_pos
        item.in_degree = JT_adj.long().sum(dim=1).view(-1)
        item.out_degree = JT_adj.long().sum(dim=0).view(-1)
        item.edge_input = torch.from_numpy(edge_input).long()
        return item

    def __len__(self):
        return len(self.data)

class PredictionDatasetWrapper(object):
    def __init__(self, max_node, spatial_pos_max, multi_hop_max_dist,
                 dataset, batch_size, valid_rate, test_rate, split_type, num_workers, split_seed):
        super(PredictionDatasetWrapper, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.valid_rate = valid_rate
        self.test_rate = test_rate
        self.split_type = split_type
        self.seed = split_seed
        self.num_workers = num_workers
        self.max_node = max_node
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

    def get_data_loaders(self):

        train_idx, valid_idx, test_idx = split(self.dataset.data, self.split_type, self.valid_rate, self.test_rate,
                                               self.seed, self.dataset.logger)

        train_set = Subset(self.dataset, train_idx)
        valid_set = Subset(self.dataset, valid_idx)
        test_set = Subset(self.dataset, test_idx)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=True,
                                  pin_memory=True, collate_fn=partial(collator, max_node=self.max_node,
                                                                      multi_hop_max_dist=self.multi_hop_max_dist,
                                                                      spatial_pos_max=self.spatial_pos_max))
        valid_loader = DataLoader(valid_set, batch_size=self.batch_size, shuffle=False, num_workers=8, drop_last=False,
                                  pin_memory=True, collate_fn=partial(collator, max_node=self.max_node,
                                                                      multi_hop_max_dist=self.multi_hop_max_dist,
                                                                      spatial_pos_max=self.spatial_pos_max))
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=8, drop_last=False,
                                  pin_memory=True, collate_fn=partial(collator, max_node=self.max_node,
                                                                      multi_hop_max_dist=self.multi_hop_max_dist,
                                                                      spatial_pos_max=self.spatial_pos_max))
        return train_loader, valid_loader, test_loader

if __name__ == '__main__':
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    dataset = 'bbbp'
    logging.basicConfig(filename='load_dataset.log', level=logging.INFO)
    logger = logging.getLogger(dataset)
    target_list = get_target_name(dataset)
    task = get_task_name(dataset)
    for target in target_list:
        dataset = PredictionDataset(dataset=dataset, target=target, task=task, fragmentation='CCSingleBond',logger=logger)
        print(f"len(dataset):{len(dataset)}")
        data_wrapper = PredictionDatasetWrapper(max_node=512, spatial_pos_max=20, multi_hop_max_dist=20,
                                                dataset=dataset, batch_size=256, valid_rate=0.1, test_rate=0.1,
                                                split_type='scaffold', num_workers=8, split_seed=8)
        train_loader, valid_loader, test_loader = data_wrapper.get_data_loaders()
        print(f"len(train_loader):{len(train_loader)}")
        print(f"len(valid_loader):{len(valid_loader)}")
        print(f"len(test_loader):{len(test_loader)}")
        for i, data in enumerate(train_loader):
            print(i)