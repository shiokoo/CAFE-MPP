import numpy as np
import rdkit.Chem as Chem
from typing import List


def get_edgelist(mol, bidirection=False, offset = 0):
    bondlist = mol.GetBonds()
    edge_list = []
    bond_cnt = 0
    for bond in bondlist:
        bond_idx = bond.GetIdx()
        assert bond_cnt == bond_idx
        start_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        edge_list.append([start_atom+offset, end_atom+offset])
        if bidirection:
            edge_list.append([end_atom+offset, start_atom+offset])
        bond_cnt += 1
    if len(edge_list) == 0:
        edge_list = np.empty((0,2), dtype=np.int64)
    return edge_list

def is_fragment(mol):
    """
    distinguish molecule and fragment according to atomic number.
    dummy atom [*] -> 0
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            return True
    return False

def get_dummy_list(mol):
    """
    :param mol: fragment
    :return: number of dummy node * and indices of dummy node *
    """
    count = 0
    dummy_list = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            dummy_list.append(atom.GetIdx())
            count += 1
    return count, dummy_list

def get_dummy_link(mol: Chem.Mol, dummy_list:List) -> List:
    """
    :param mol: fragment
    :param dummy_list: indices of dummy node *
    :return: indices of bond which links to dummy node*
    """
    edge_list = get_edgelist(mol, bidirection=False)
    dummy_link_list = list()
    for idx in dummy_list:
        atom = mol.GetAtomWithIdx(idx)
        nei = atom.GetNeighbors()
        assert len(nei) == 1
        nei_idx = nei[0].GetIdx()
        try:
            bond_id = edge_list.index([idx, nei_idx])
        except ValueError:
            bond_id = edge_list.index([nei_idx, idx])
        else:
            pass
        dummy_link_list.append(bond_id)
    return dummy_link_list

def get_link_list(mol: Chem.Mol, dummy_list) -> List:
    """
    obtain indices of linkable nodes (unlinked to dummy node *)
    for changing the link of dummy node * to a linkable node
    """
    N = mol.GetNumAtoms()
    total_node_list = set(range(N))
    # atom idx of dummy node and the node linked to dummy node
    link_dummy_list = set()
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        if (begin_atom in dummy_list) or (end_atom in dummy_list):
            link_dummy_list.add(begin_atom)
            link_dummy_list.add(end_atom)
    # obtain the node unlinked to dummy node *
    link_list = total_node_list - link_dummy_list
    # check whether the number of Implicit Hs is larger than zero.
    remove_node = set()
    for idx in link_list:
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetNumImplicitHs() == 0) or (atom.GetDegree() == 5):
            remove_node.add(idx)
    link_list = link_list - remove_node
    link_list = list(link_list)
    return link_list

def get_link_atom_list(mol:Chem.Mol) -> List:
    """
    obtain indices of linkable nodes 
    for adding dummy nodes
    """
    link_atom_list = []
    for atom in mol.GetAtoms():
        if (atom.GetNumImplicitHs() > 0) and (atom.GetDegree() < 5):
            link_atom_list.append(atom.GetIdx())
    return link_atom_list

def count_mol(data_path):
    """
    count the number of data in the dataset
    """
    with open(data_path,'r') as f:
        smiles = []
        for line in f.readlines():
            smi = line.strip('\n')
            smiles.append(smi)
        print(len(smiles))
