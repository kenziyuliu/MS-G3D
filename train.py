#!/usr/bin/env python
from __future__ import print_function
import os
import time
import yaml
import pprint
import random
import pickle
import shutil
import inspect
import argparse
from collections import OrderedDict, defaultdict

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import apex
from mmcv.runner import DistSamplerSeedHook, Runner, obj_from_dict

from utils.tools import count_params, import_class, get_parser
from utils.distributed import MMDistributedDataParallel
from utils.distributed import DistributedSampler
from utils.distributed import DistOptimizerHook
from utils.eval_hooks import DistEvalTopKAccuracyHook


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class msg3d_with_loss(nn.Module):
    def __init__(self, backbone, loss):
        super(msg3d_with_loss, self).__init__()
        self.network = backbone
        self.loss = loss

    def forward(self, batchdata, label, return_loss=True):
        output = self.network(batchdata)
        if not return_loss:
            return output
        label = label.view(-1)
        losses = self.loss(output, label)
        return {"loss": losses}


class Processor():
    """Processor for Skeleton-based Action Recgnition"""

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        self._init_dist_pytorch(backend='nccl', world_size=torch.cuda.device_count())
        self.load_model()
        self.load_param_groups()
        self.load_optimizer()
        self.load_lr_scheduler()
        self.load_data()

        self.global_step = 0
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        if self.arg.half:
            self.print_log('*************************************')
            self.print_log('*** Using Half Precision Training ***')
            self.print_log('*************************************')
            self.model, self.optimizer = apex.amp.initialize(
                self.model,
                self.optimizer,
                opt_level=f'O{self.arg.amp_opt_level}'
            )
            if self.arg.amp_opt_level != 1:
                self.print_log('[WARN] nn.DataParallel is not yet supported by amp_opt_level != "O1"')
        self.runner()

    def _init_dist_pytorch(self, backend, **kwargs):
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        dist.init_process_group(backend=backend, **kwargs)

    def load_model(self):
        Model = import_class(self.arg.model)

        # Copy model file and main
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        shutil.copy2(os.path.join('.', __file__), self.arg.work_dir)

        self._model = Model(**self.arg.model_args).cuda()
        self.loss = nn.CrossEntropyLoss().cuda()
        self.print_log(f'Model total number of params: {count_params(self._model)}')

        if self.arg.weights:
            try:
                self.global_step = int(arg.weights[:-3].split('-')[-1])
            except:
                print('Cannot parse global_step from model weights filename')
                self.global_step = 0

            self.print_log(f'Loading weights from {self.arg.weights}')
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            elif '.pth' in self.arg.weights:
                weights = torch.load(self.arg.weights)["state_dict"]
                weights = OrderedDict(
                    [[k.split('network.')[-1],
                      v.cuda()] for k, v in weights.items()])
            else:
                weights = torch.load(self.arg.weights)
                weights = OrderedDict(
                    [[k.split('module.')[-1],
                      v.cuda()] for k, v in weights.items()])


            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log(f'Sucessfully Remove Weights: {w}')
                else:
                    self.print_log(f'Can Not Remove Weights: {w}')
           
            if '.pth' in self.arg.weights:
                try:
                    self._model.load_state_dict(weights)
                except:
                    state = self._model.state_dict()
                    diff = list(set(state.keys()).difference(set(weights.keys())))
                    self.print_log('Can not find these weights:')
                    for d in diff:
                        self.print_log('  ' + d)
                    state.update(weights)
                    self._model.load_state_dict(state)
            elif self.arg.weights.endswith(".pt") or self.arg.weights.endswith(".pkl"):
                model_params = self._model.state_dict()
                # model_params.update(weights)
                self._model.load_state_dict(model_params, strict=False)
            else:
                raise "Support *.pth or *.pkl or *.pt pretrain"


        self._model_full = msg3d_with_loss(self._model, self.loss)
        rank = int(os.environ['RANK'])
        # self._model.to(rank)
        self._model_full.to(rank)
        self.model = MMDistributedDataParallel(self._model_full.cuda())

    def load_param_groups(self):
        """
        Template function for setting different learning behaviour
        (e.g. LR, weight decay) of different groups of parameters
        """
        self.param_groups = defaultdict(list)

        for name, params in self.model.named_parameters():
            self.param_groups['other'].append(params)

        self.optim_param_groups = {
            'other': {'params': self.param_groups['other']}
        }

    def load_optimizer(self):
        params = list(self.optim_param_groups.values())
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                params,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                params,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError('Unsupported optimizer: {}'.format(self.arg.optimizer))

        # Load optimizer states if any
        if self.arg.checkpoint is not None:
            self.print_log(f'Loading optimizer states from: {self.arg.checkpoint}')
            self.optimizer.load_state_dict(torch.load(self.arg.checkpoint)['optimizer_states'])
            current_lr = self.optimizer.param_groups[0]['lr']
            self.print_log(f'Starting LR: {current_lr}')
            self.print_log(f'Starting WD1: {self.optimizer.param_groups[0]["weight_decay"]}')
            if len(self.optimizer.param_groups) >= 2:
                self.print_log(f'Starting WD2: {self.optimizer.param_groups[1]["weight_decay"]}')

    def load_lr_scheduler(self):
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=self.arg.step, gamma=0.1)
        if self.arg.checkpoint is not None:
            scheduler_states = torch.load(self.arg.checkpoint)['lr_scheduler_states']
            self.print_log(f'Loading LR scheduler states from: {self.arg.checkpoint}')
            self.lr_scheduler.load_state_dict(scheduler_states)
            self.print_log(f'Starting last epoch: {scheduler_states["last_epoch"]}')
            self.print_log(f'Loaded milestones: {scheduler_states["last_epoch"]}')

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()

        def worker_seed_fn(worker_id):
            # give workers different seeds
            return init_seed(self.arg.seed + worker_id + 1)

        rank = int(os.environ['RANK'])
        world_size = torch.cuda.device_count()
        if self.arg.phase == 'train':
            dataset_train = Feeder(is_test=False, **self.arg.train_feeder_args)
            sampler_train = DistributedSampler(dataset_train, world_size, rank, shuffle=True)
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=dataset_train,
                batch_size=self.arg.batch_size // world_size,
                sampler=sampler_train,
                shuffle=False,
                num_workers=self.arg.num_worker // world_size,
                drop_last=True,
                worker_init_fn=worker_seed_fn)

        dataset_test = Feeder(is_test=False, **self.arg.test_feeder_args)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=dataset_test,
            batch_size=self.arg.test_batch_size // world_size,
            shuffle=False,
            num_workers=self.arg.num_worker // world_size,
            drop_last=False,
            worker_init_fn=worker_seed_fn)

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir, exist_ok=True)
        with open(os.path.join(self.arg.work_dir, 'config.yaml'), 'w') as f:
            yaml.dump(arg_dict, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log(f'Local current time: {localtime}')

    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = f'[ {localtime} ] {s}'
        print(s)
        if self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a') as f:
                print(s, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def save_states(self, epoch, states, out_folder, out_name):
        out_folder_path = os.path.join(self.arg.work_dir, out_folder)
        out_path = os.path.join(out_folder_path, out_name)
        os.makedirs(out_folder_path, exist_ok=True)
        torch.save(states, out_path)

    def save_checkpoint(self, epoch, out_folder='checkpoints'):
        state_dict = {
            'epoch': epoch,
            'optimizer_states': self.optimizer.state_dict(),
            'lr_scheduler_states': self.lr_scheduler.state_dict(),
        }

        checkpoint_name = f'checkpoint-{epoch}-fwbz{self.arg.forward_batch_size}-{int(self.global_step)}.pt'
        self.save_states(epoch, state_dict, out_folder, checkpoint_name)

    def save_weights(self, epoch, out_folder='weights'):
        state_dict = self.model.state_dict()
        weights = OrderedDict([
            [k.split('module.')[-1], v.cpu()]
            for k, v in state_dict.items()
        ])

        weights_name = f'weights-{epoch}-{int(self.global_step)}.pt'
        self.save_states(epoch, weights, out_folder, weights_name)

    def runner(self):
        def parse_losses(losses):
            log_vars = OrderedDict()
            for loss_name, loss_value in losses.items():
                if isinstance(loss_value, torch.Tensor):
                    log_vars[loss_name] = loss_value.mean()
                elif isinstance(loss_value, list):
                    log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
                else:
                    raise TypeError(
                        '{} is not a tensor or list of tensors'.format(loss_name))

            loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

            log_vars['loss'] = loss
            for name in log_vars:
                log_vars[name] = log_vars[name].item()

            return loss, log_vars

        def batch_processor(model, data, train_mode):
            losses = model(**data)
            # losses = model(data)
            loss, log_vars = parse_losses(losses)
            outputs = dict(
                loss=loss, log_vars=log_vars,
                num_samples=len(data['batchdata'].data))
            return outputs
        self.runner = Runner(self.model, batch_processor, self.optimizer, self.arg.work_dir)
        optimizer_config = DistOptimizerHook(grad_clip=dict(max_norm=20, norm_type=2))
        if not "policy" in self.arg.policy:
            lr_config = dict(policy='step', step=self.arg.step)
        else:
            lr_config = dict(**self.arg.policy)
        checkpoint_config = dict(interval=5)
        log_config = dict(
            interval=20,
            hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
        self.runner.register_training_hooks(lr_config, optimizer_config, checkpoint_config, log_config)
        self.runner.register_hook(DistSamplerSeedHook())
        Feeder = import_class(self.arg.feeder)
        self.runner.register_hook(DistEvalTopKAccuracyHook(
            Feeder(is_test=False, **self.arg.test_feeder_args),
            interval=self.arg.test_interval, k=(1, 5)))

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        # Skip evaluation if too early
        if epoch + 1 < self.arg.eval_start:
            return

        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        with torch.no_grad():
            self.model = self.model.cuda()
            self.model.eval()
            self.print_log(f'Eval epoch: {epoch + 1}')
            for ln in loader_name:
                loss_values = []
                score_batches = []
                step = 0
                process = tqdm(self.data_loader[ln], dynamic_ncols=True)
                for batch_idx, (data, label, index) in enumerate(process):
                    data = data.float().cuda()
                    label = label.long().cuda()
                    output = self.model(data)
                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    loss = self.loss(output, label)
                    score_batches.append(output.data.cpu().numpy())
                    loss_values.append(loss.item())

                    _, predict_label = torch.max(output.data, 1)
                    step += 1

                    if wrong_file is not None or result_file is not None:
                        predict = list(predict_label.cpu().numpy())
                        true = list(label.data.cpu().numpy())
                        for i, x in enumerate(predict):
                            if result_file is not None:
                                f_r.write(str(x) + ',' + str(true[i]) + '\n')
                            if x != true[i] and wrong_file is not None:
                                f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

            score = np.concatenate(score_batches)
            loss = np.mean(loss_values)
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            print('Accuracy: ', accuracy, ' model: ', self.arg.work_dir)
            if self.arg.phase == 'train' and not self.arg.debug:
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('loss_l1', l1, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log(f'\tMean {ln} loss of {len(self.data_loader[ln])} batches: {np.mean(loss_values)}.')
            for k in self.arg.show_topk:
                self.print_log(f'\tTop {k}: {100 * self.data_loader[ln].dataset.top_k(score, k):.2f}%')

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

        # Empty cache after evaluation
        torch.cuda.empty_cache()

    def start(self):
        if self.arg.checkpoint is not None:
            self.runner.resume(self.arg.checkpoint)
        if self.arg.phase == 'train':
            self.runner.run([self.data_loader['train']], workflow=[('train', 1)], max_epochs=self.arg.num_epoch)
        elif self.arg.phase == 'eval':
            self.runner.run([self.data_loader['test']], workflow=[('train', 1)], max_epochs=self.arg.num_epoch)
            self.print_log('Done.\n')


def main():
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    os.environ['RANK'] = str(arg.local_rank)
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()


if __name__ == '__main__':
    main()
