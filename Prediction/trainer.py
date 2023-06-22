import os
import shutil
import random
import logging
import numpy as np
import pandas as pd
import torch
from ruamel import yaml
from torch import nn
import torch.nn.functional as F
from collections import Iterable
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from Prediction.model import FragTransformer
from Prediction.dataset import get_target_name, get_task_name

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def _save_config_file(model_ckpt_folder, config):
    if not os.path.exists(model_ckpt_folder):
        os.makedirs(model_ckpt_folder)
        with open(os.path.join(model_ckpt_folder, 'config_prediction.yaml'), 'w', encoding="utf-8") as f:
            yaml.dump(config, f, Dumper=yaml.RoundTripDumper)
        # shutil.copy('../Config/config_prediction.yaml', os.path.join(model_ckpt_folder, 'config_prediction.yaml'))

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

class FragTransformerTrainer(object):
    def __init__(self, dataset, config):
        self.config = config
        self.dataset = dataset
        self.device = torch.device(self.config['gpu'] if torch.cuda.is_available() else 'cpu')

        self.log_dir = os.path.join('ckpt', config['model_folder'], config['dataset']['dataset_name'])

        if config['dataset']['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif config['dataset']['task'] == 'regression':
            if self.config['dataset']['dataset_name'] in ['qm7', 'qm8']:
                self.criterion = nn.L1Loss()
            else:
                self.criterion = nn.MSELoss()

    def _step(self, model, data, n_iter):
        # get the prediction
        pred = model(data)  # [N,C]

        if self.config['dataset']['task'] == 'classification':
            loss = self.criterion(pred, data.y.flatten())
        elif self.config['dataset']['task'] == 'regression':
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(data.y))
            else:
                loss = self.criterion(pred, data.y)
        else:
            raise ValueError('Unknown task')
        
        return loss

    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()
        print(f"len(train_loader):{len(train_loader)}")
        print(f"len(valid_loader):{len(valid_loader)}")
        print(f"len(test_loader):{len(test_loader)}")
        self.normalizer = None

        if self.config['dataset']['dataset_name'] in ['qm7']:
            labels = []
            for data in train_loader:
                labels.append(data.y)
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels)
            print(f"mean:{self.normalizer.mean}, std:{self.normalizer.std}, label shape:{labels.shape}")

        model = FragTransformer(self.config['dataset']['task'], self.config['edge_type'], self.config['multi_hop_max_dist'],
                                **self.config["model"]).to(self.device)

        model = self._load_pre_trained_weights(model)
        print(model)

        if self.config['finetune']:
            layer_list = []
            for name, child in model.named_children():
                if name in self.config['module']:
                    for child_name, param in child.named_parameters():
                        layer_list.append(self.config['module'] + '.' + child_name)
                        # print(layer_list)
                else:
                    pass
            params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
            downsteam_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

            optimizer = torch.optim.Adam(
                [{'params': params, 'lr': self.config['init_lr']}, {'params': downsteam_params}],
                self.config['downstream_lr'], weight_decay=eval(str(self.config['weight_decay']))
            )
        else:
            #freeze the pre-trained model -> extract the embeddings of fragments
            self.freeze_by_names(model, layer_names=self.config['module'])

            optimizer = torch.optim.Adam(model.parameters(), self.config['downstream_lr'],
                                         weight_decay=eval(str(self.config['weight_decay'])))

        model_ckpt_folder = os.path.join(self.log_dir, 'checkpoints', 'exp' + str(self.config['exp']),
                                         str(self.config['seed']))
        _save_config_file(model_ckpt_folder, self.config)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_cls = 0

        for epoch_counter in range(self.config['epochs']):
            for bn, data in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(self.device)
                loss = self._step(model, data, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    print(f"epoch:{epoch_counter}, n_iter:{bn}, loss:{loss.item()}")
                loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification':
                    valid_loss, valid_cls = self._validate(model, valid_loader)
                    if valid_cls > best_valid_cls:
                        # save the model weights
                        best_valid_cls = valid_cls
                        torch.save(model.state_dict(), os.path.join(model_ckpt_folder, 'model_{}.pth'.format(self.config['dataset']['target'])))
                elif self.config['dataset']['task'] == 'regression':
                    valid_loss, valid_rgr = self._validate(model, valid_loader)
                    if valid_rgr < best_valid_rgr:
                        # save the model weights
                        best_valid_rgr = valid_rgr
                        torch.save(model.state_dict(), os.path.join(model_ckpt_folder, 'model_{}.pth'.format(self.config['dataset']['target'])))

                valid_n_iter += 1

        self._test(model, test_loader)

    def set_freeze_by_names(self, model, layer_names, freeze=True):
        if not isinstance(layer_names, Iterable):
            layer_names = [layer_names]
        for name, child in model.named_children():
            if name not in layer_names:
                continue
            for param in child.parameters():
                param.requires_grad = not freeze

    def freeze_by_names(self, model, layer_names):
        self.set_freeze_by_names(model, layer_names, freeze=True)

    def unfreeze_by_names(self, model, layer_names):
        self.set_freeze_by_names(model, layer_names, freeze=False)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join(self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)

            model.load_my_state_dict(state_dict, self.config['module'])
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                pred = model(data)
                loss = self._step(model, data, bn)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            valid_loss /= num_data

        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['dataset']['dataset_name'] in ['qm7', 'qm8']:
                mae = mean_absolute_error(labels, predictions)
                print('Validation loss:', valid_loss, 'MAE:', mae)
                return valid_loss, mae
            else:
                rmse = mean_squared_error(labels, predictions, squared=False)
                print('Validation loss:', valid_loss, 'RMSE:', rmse)
                return valid_loss, rmse

        elif self.config['dataset']['task'] == 'classification':
            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions[:, 1])
            print('Validation loss:', valid_loss, 'ROC AUC:', roc_auc)
            return valid_loss, roc_auc

    def _test(self, model, test_loader):
        model_path = os.path.join(self.log_dir, 'checkpoints', 'exp' + str(self.config['exp']),
                                  str(self.config['seed']), 'model_{}.pth'.format(self.config['dataset']['target']))
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                pred = model(data)
                loss = self._step(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data

        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['dataset']['dataset_name'] in ['qm7', 'qm8']:
                self.mae = mean_absolute_error(labels, predictions)
                print('Test loss:', test_loss, 'Test MAE:', self.mae)
            else:
                self.rmse = mean_squared_error(labels, predictions, squared=False)
                print('Test loss:', test_loss, 'Test RMSE:', self.rmse)

        elif self.config['dataset']['task'] == 'classification':
            predictions = np.array(predictions)
            labels = np.array(labels)
            self.roc_auc = roc_auc_score(labels, predictions[:, 1])
            print('Test loss:', test_loss, 'Test ROC AUC:', self.roc_auc)

def main(config):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    logging.basicConfig(filename='load_dataset.log', level=logging.INFO)
    logger = logging.getLogger(config['dataset']['dataset_name'])

    from Prediction.dataset import PredictionDataset, PredictionDatasetWrapper
    dataset = PredictionDataset(config['dataset']['dataset_name'], config['dataset']['target'],
                                task=config['dataset']['task'], fragmentation=config['fragmentation'],logger=logger)
    dataset = PredictionDatasetWrapper(config['max_node'], config['spatial_pos_max'], config['multi_hop_max_dist'],
                                       dataset, config['batch_size'], **config['wrapper'])

    FT_Trainer = FragTransformerTrainer(dataset, config)
    FT_Trainer.train()

    if config['dataset']['task'] == 'classification':
        return FT_Trainer.roc_auc
    if config['dataset']['task'] == 'regression':
        if config['dataset']['dataset_name'] in ['qm7', 'qm8']:
            return FT_Trainer.mae
        else:
            return FT_Trainer.rmse

if __name__ == '__main__':
    with open('../Config/config_prediction.yaml', encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.RoundTripLoader)

    target_list = get_target_name(config['dataset']['dataset_name'])
    task = get_task_name(config['dataset']['dataset_name'])
    config['dataset']['task'] = task

    config['exp'] = 0
    
    seed = config['seed']
    setup_seed(seed)

    results_list = []
    results = []
    for target in target_list:
        config['dataset']['target'] = target
        print(config)
        result = main(config)
        results_list.append([target, result])
        results.append(result)
    results = torch.tensor(results)
    average = torch.mean(results).item()
    results_list.append(['exp', 0, 'seed', seed, 'average', average])
    results_dir = os.path.join('ckpt', config['model_folder'], config['dataset']['dataset_name'], 'results')
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(results_list)
    df.to_csv(
        results_dir + '/{}_{}.csv'.format('FragTransformer', config['dataset']['dataset_name']),
        mode='a', index=False, header=False
    )
