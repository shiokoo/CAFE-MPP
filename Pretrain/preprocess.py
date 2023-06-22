import logging
import rdkit
import torch
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import BRICS
from typing import List

def GetCCSingleBonds(mol):
    """
    Generates a list of singlebond between C and C, which singlebond is not in a ring.
    :param mol: A molecule (RDKit molecule).
    :return: A list of CC singlebond id and a list of atom pairs of CC singlebond
    """
    CCSingle_bonds = list()
    for bond in mol.GetBonds():
        if bond.GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE:
            if not bond.IsInRing():
                begin_atom_idx = bond.GetBeginAtomIdx()
                end_atom_idx = bond.GetEndAtomIdx()
                begin_atom = mol.GetAtomWithIdx(begin_atom_idx).GetSymbol()
                end_atom = mol.GetAtomWithIdx(end_atom_idx).GetSymbol()
                if (begin_atom == 'C') & (end_atom == 'C'):
                    bond_idx = bond.GetIdx()
                    CCSingle_bonds.append([bond_idx, begin_atom_idx, end_atom_idx])
    return CCSingle_bonds

def GetSingleBonds(mol):
    """
    Reference: FraGAT: a fragment-oriented multi-scale graph attention model for molecular property prediction
    """
    Single_bonds = list()
    for bond in mol.GetBonds():
        if bond.GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE:
            if not bond.IsInRing():
                begin_atom_idx = bond.GetBeginAtomIdx()
                end_atom_idx = bond.GetEndAtomIdx()
                bond_idx = bond.GetIdx()
                Single_bonds.append([bond_idx, begin_atom_idx, end_atom_idx])
    return Single_bonds

def GetBRICSBonds(mol):
    """
    Reference: Motif-based Graph Self-Supervised Learning for molecular Property Prediction
    """
    global_bonds = []
    breaks = []
    for bond in mol.GetBonds():
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        begin_atom = mol.GetAtomWithIdx(begin_atom_idx)
        end_atom = mol.GetAtomWithIdx(end_atom_idx)

        # break bonds between rings and non-ring atoms
        if begin_atom.IsInRing() and not end_atom.IsInRing():
            bond_idx = bond.GetIdx()
            breaks.append([bond_idx, begin_atom_idx, end_atom_idx])
        if end_atom.IsInRing() and not begin_atom.IsInRing():
            bond_idx = bond.GetIdx()
            breaks.append([bond_idx, begin_atom_idx, end_atom_idx])
        global_bonds.append([begin_atom_idx, end_atom_idx])
    try:
        # select atoms at intersections as break point
        for atom in mol.GetAtoms():
            if len(atom.GetNeighbors()) > 2 and not atom.IsInRing():
                for nei in atom.GetNeighbors():
                    if [nei.GetIdx(), atom.GetIdx()] in global_bonds:
                        bond_idx = global_bonds.index([nei.GetIdx(), atom.GetIdx()])
                        if bond_idx not in np.array(breaks)[:, 0]:
                            breaks.append([bond_idx, nei.GetIdx(), atom.GetIdx()])
                    elif [atom.GetIdx(), nei.GetIdx()] in global_bonds:
                        bond_idx = global_bonds.index([atom.GetIdx(), nei.GetIdx()])
                        if bond_idx not in np.array(breaks)[:, 0]:
                            breaks.append([bond_idx, atom.GetIdx(), nei.GetIdx()])
    except:
        pass
    else:
        pass
    try:
        # find brics bonds
        res = list(BRICS.FindBRICSBonds(mol))
        if len(res) != 0:
            for bond in res:
                if [bond[0][0], bond[0][1]] in global_bonds:
                    bond_idx = global_bonds.index([bond[0][0], bond[0][1]])
                    if bond_idx not in np.array(breaks)[:, 0]:
                        breaks.append([bond_idx, bond[0][0], bond[0][1]])
                else:
                    bond_idx = global_bonds.index([bond[0][1], bond[0][0]])
                    if bond_idx not in np.array(breaks)[:, 0]:
                        breaks.append([bond_idx, bond[0][1], bond[0][0]])
        else:
            pass
    except:
        pass
    else:
        pass

    return sorted(breaks)
    """
    Generates a list of singlebond between C and C, which singlebond is not in a ring.
    :param mol: A molecule (RDKit molecule).
    :return: A list of CC singlebond id and a list of atom pairs of CC singlebond
    """
    bond_id = list()
    link_id = list()
    for bond in mol.GetBonds():
        if bond.GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE:
            if not bond.IsInRing():
                begin_atom = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol()
                end_atom = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol()
                if (begin_atom == 'C') & (end_atom == 'C'):
                    bond_id.append(bond.GetIdx())
                    link_id.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    return bond_id, link_id

def fragment_generator(mol: Chem.Mol, bonds_id: List) -> List:
    """
    Generates fragments of a molecule given a list of bond ids.
    :param mol: A molecule (RDKit molecule).
    :param bonds_id: A list of CC singlebond id.
    :return: A list of fragments (RDKit molecule).
    """
    if bonds_id:
        dummyLabels = [(0, 0) for i in range(len(bonds_id))]
        fragments = Chem.FragmentOnBonds(mol, bonds_id, addDummies=True, dummyLabels=dummyLabels)
        smiles = Chem.MolToSmiles(fragments)
        smiles = smiles.split('.')
        frags = []
        for smi in smiles:
            subgraph = Chem.MolFromSmiles(smi)
            if subgraph: # not NoneType
                frag = Chem.MolToSmiles(subgraph, isomericSmiles=True)
            else:
                logging.info(f'The following SMILES needs to be checked: {Chem.MolToSmiles(mol)}')
                # discard the molecule whose fragments are invalid.
                frags = []
                break
            frags.append(frag)
    else:
        # bonds id is empty (no bond to cut).
        # canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        # discard the molecule
        frags = []
    return frags

def break_on_bond(mol:Chem.Mol, bond) -> Chem.Mol:
    """
    Given bond id, cut it, generate fragments.
    :param mol: A molecule or a atom group to be fragmented (RDKit molecule).
    :param bond: id of bond to be cut
    :return: frags tuple
    """
    broken = Chem.FragmentOnBonds(mol, bondIndices=[bond], dummyLabels=[(0,0)])
    res = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=False)
    return res

def build_fragment_vocub(src_path, res_path, cut_method='CCSingleBond'):
    with open(src_path, 'r') as f:
        fragments = list()
        count = 0 # count the number of molecules
        for line in f.readlines():
            smiles = line.strip('\n')
            try:
                mol = Chem.MolFromSmiles(smiles)
                if cut_method == 'CCSingleBond':
                    bonds = GetCCSingleBonds(mol)
                elif cut_method == 'SingleBond':
                    bonds = GetSingleBonds(mol)
                elif cut_method == 'BRICSBond':
                    bonds = GetBRICSBonds(mol)
                else:
                    raise RuntimeError('The method of fragmentation is not specified.')
                
                # bonds_id = [item[0] for item in bond]
                if bonds:
                    bonds_id = torch.tensor(bonds, dtype=torch.int64)[:,0].tolist()
                else:
                    bonds_id = []

                frags = fragment_generator(mol,bonds_id)
                for item in frags:
                    if item not in fragments:
                        fragments.append(item)
                    else:
                        pass
                count+=1
                if count % 10000 == 0:
                    print(f'{count} molecules have been preprocessed.')
            except:
                logging.info(f'The following SMILES needs to be checked: {smiles}')
                count+=1
            else:
                pass
        print(f'{count} molecules have been fragmented.')
        print(f"The total number of fragments is {len(fragments)}")
        with open(res_path,'w') as fp:
            [fp.write(item + '\n') for item in fragments]
            fp.close()

if __name__ == '__main__':
    # example
    mol = Chem.MolFromSmiles('CCCOc1csc2cc(O)c([S-])cc12')
    canonical_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    print(canonical_smi)
    bonds = GetCCSingleBonds(mol)
    bonds_id = [item[0] for item in bonds]
    frags = fragment_generator(mol, bonds_id)
    print(frags)

    build_fragment_vocub('../Data/chembl_30.txt', '../Data/chembl_fragment.txt', cut_method='CCSingleBond')
