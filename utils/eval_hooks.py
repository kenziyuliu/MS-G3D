import os
import os.path as osp
import numpy as np

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import Hook, obj_from_dict
from torch.utils.data import Dataset

from .parallel_eval import collate
from .distributed import scatter


def top_k_hit(score, lb_set, k=3):
    idx = np.argsort(score)[-k:]
    return len(lb_set.intersection(idx)) > 0, 1


def top_k_accuracy(scores, labels, k=(1, )):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = []
    for kk in k:
        hits = []
        for x, y in zip(scores, labels):
            y = int(y)
            y = [y] if isinstance(y, int) else y
            hits.append(top_k_hit(x, set(y), k=kk)[0])
        res.append(np.mean(hits))
    return res


class DistEvalHook(Hook):

    def __init__(self, dataset, interval=1, distributed=True):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval
        self.dist = distributed

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        # if runner.rank == 0:
        #     prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                result = runner.model(return_loss=False, **data_gpu)
            results[idx] = result.cpu().detach().numpy()

            batch_size = runner.world_size
            # if runner.rank == 0:
            #     for _ in range(batch_size):
            #         prog_bar.update()

        tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(runner.rank))
        mmcv.dump(results, tmp_file)
        dist.barrier()

        if runner.rank == 0:
            print('\n')
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
            os.remove(osp.join(runner.work_dir, 'temp_0.pkl'))

        return

    def evaluate(self):
        raise NotImplementedError


class DistEvalTopKAccuracyHook(DistEvalHook):

    def __init__(self, dataset, interval=1, k=(1, ), dist=True):
        super(DistEvalTopKAccuracyHook, self).__init__(dataset, interval, dist)
        self.k = k

    def evaluate(self, runner, results):
        gt_labels = []
        for i in range(len(self.dataset)):
            # ann = self.dataset.video_infos[i]
            # gt_labels.append(ann['label'])
            gt_labels.append(self.dataset.label[i])

        results = [res.squeeze() for res in results]
        print(len(results), " VS ", len(gt_labels))
        top1, top5 = top_k_accuracy(results, gt_labels, k=self.k)
        runner.mode = 'val'
        runner.log_buffer.output['top1 acc'] = top1
        runner.log_buffer.output['top5 acc'] = top5
        runner.log_buffer.ready = True
