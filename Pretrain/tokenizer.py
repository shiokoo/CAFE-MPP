import torch

def build_vocub(dataset):
    """
    construct a vocabulary for SMILES
    """
    char_set = set()
    for smiles in dataset:
        for c in smiles:
            char_set.add(c)
    return ''.join(char_set)

class SmilesTokenizer():
    """
    max_len: the max length of SMILES
    """
    def __init__(self, smiles_vocab, max_len=256):
        self.smiles_vocab = smiles_vocab
        self.vocab_size = len(smiles_vocab) + 3  #[SOS], [EOS], [PAD]
        self.max_len = max_len
        self.SOS = self.vocab_size - 2
        self.EOS = self.vocab_size - 1
        self.PAD = 0

    def token2id(self, token):
        return self.smiles_vocab.find(token) + 1  #[PAD] -> 0

    def id2token(self, id):
        if id == self.SOS: return '[SOS]'
        if id == self.EOS: return '[EOS]'
        if id == self.PAD: return '[PAD]'
        return self.smiles_vocab[id-1]

    def tokenize(self, smiles):
        tensor = torch.zeros(1, self.max_len, dtype=torch.int64)
        tensor[0, 0] = self.SOS
        for ii, token in enumerate(smiles):
            tensor[0, ii+1] = self.token2id(token)
            if ii+3 == self.max_len: break
        tensor[0, ii+2] = self.EOS
        return tensor
