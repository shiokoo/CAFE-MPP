import os
import random
import shutil
import torch
import yaml
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from Pretrain.metric import NTXent

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def _save_config_file(model_ckpt_folder):
    if not os.path.exists(model_ckpt_folder):
        os.makedirs(model_ckpt_folder)
        shutil.copy('../Config/config_pretrain.yaml', os.path.join(model_ckpt_folder, 'config_pretrain.yaml'))

class FragCLRTrainer(object):
    def __init__(self, dataset, config):
        self.config = config
        self.dataset = dataset
        self.device = torch.device(self.config['gpu'] if torch.cuda.is_available() else 'cpu')
        self.log_dir = os.path.join('ckpt', config['model_folder'])

        self.NTXent = NTXent(self.device, **config['loss'])

    def _step(self, model, seq_frag, graph_frag, graph_frag_aug, n_iter):
        # get the representations and the projections
        seq_rep, graph_rep, graph_aug_rep, \
        seq_out, graph_out, graph_aug_out = model(seq_frag, graph_frag, graph_frag_aug)  # [N,D]

        graph_aug_out = graph_aug_out.unsqueeze(1) # [N,1,D]
        # calculate loss
        loss = self.NTXent(query=graph_out, positive_key=seq_out, negative_keys=graph_aug_out)

        return loss

    def train(self):
        train_loader, valid_loader = self.dataset.get_data_loaders()

        from Pretrain.model import FragCLR
        model = FragCLR(**self.config['model']).to(self.device)
        model = self._load_pre_trained_weights(model)
        print(model)

        optimizer = torch.optim.Adam(model.parameters(), self.config['init_lr'],
                                     weight_decay=eval(self.config['weight_decay']))
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.config['epochs']-self.config['warm_up'],
            eta_min=0, last_epoch=-1
        )

        model_ckpt_folder = os.path.join(self.log_dir, 'checkpoints')
        _save_config_file(model_ckpt_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        for epoch_counter in range(self.config['epochs']):
            for bn, (seq_frag, graph_frag, graph_frag_aug) in enumerate(train_loader):
                optimizer.zero_grad()

                seq_frag = seq_frag.to(self.device)
                graph_frag = graph_frag.to(self.device)
                graph_frag_aug = graph_frag_aug.to(self.device)

                loss = self._step(model, seq_frag.x, graph_frag, graph_frag_aug, n_iter)
 
                if n_iter % self.config['log_every_n_steps'] == 0:
                    print(f"epoch:{epoch_counter}, n_iter:{bn}, loss:{loss.item()}")

                loss.backward()
                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                print(f"epoch:{epoch_counter}, n_iter:{bn}, valid_loss:{valid_loss} (validation)")
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_ckpt_folder, 'model.pth'))

                valid_n_iter += 1

            if (epoch_counter+1) % self.config['save_every_n_epochs'] == 0:
                torch.save(model.state_dict(), os.path.join(model_ckpt_folder, 'model_{}.pth'.format(str(epoch_counter))))

            # warmup for the first few epochs
            if epoch_counter >= self.config['warm_up']:
                scheduler.step()

    def _load_pre_trained_weights(self, model):
        try:
            ckpt_folder = os.path.join('./ckpt', self.config['load_model'], 'checkpoints')
            state_dict = torch.load(os.path.join(ckpt_folder), 'model.pth')
            model.load_state_dict(state_dict)
            print('Loaded pre-trained model with success.')
        except FileNotFoundError:
            print('Pre-trained weights not found. Training from scratch.')
        return model

    def _validate(self, model, valid_loader):
        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (seq_frag, graph_frag, graph_frag_aug) in valid_loader:
                seq_frag = seq_frag.to(self.device)
                graph_frag = graph_frag.to(self.device)
                graph_frag_aug = graph_frag_aug.to(self.device)

                loss = self._step(model, seq_frag.x, graph_frag, graph_frag_aug, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter

        model.train()
        return valid_loss

def main():
    config = yaml.load(open('../Config/config_pretrain.yaml','r'), Loader=yaml.FullLoader)
    print(config)
    setup_seed(config['seed'])
    from Pretrain.dataset import PretrainDataset, PretrainDatasetWrapper
    dataset = PretrainDatasetWrapper(PretrainDataset(**config['dataset']), config['batch_size'], **config['wrapper'])
    fragclr = FragCLRTrainer(dataset, config)
    fragclr.train()

if __name__ == '__main__':
    main()