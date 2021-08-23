"""
Modified from ML-GCN/util.py:
* adding calculations: OP, OR, OF1, CP, CR, CF1, EP, ER, EF1
* removing args: difficult_examples
"""

import math
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import torch.nn.functional as F

class AveragePrecisionMeter(object):

    def __init__(self):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        concatenate samples of the new batch and previous batches
        Args:
            output: predicted multiple labels, should be an NxK tensor, postive/negative means presence/absence
            target: ground truth multiple labels, should be an NxK binary tensors, each is multi-hot
        Notes:
            N: the number of samples
            K: the number of classes
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        self.scores_nonzero = self.scores[:, self.targets.sum(axis=0)>0]
        self.targets_nonzero = self.targets[:, self.targets.sum(axis=0)>0]
        ap = torch.zeros(self.scores_nonzero.size(1))
        rg = torch.arange(1, self.scores_nonzero.size(0)).float()
        # compute average precision for each class
        for k in range(self.scores_nonzero.size(1)):
            # sort scores
            scores = self.scores_nonzero[:, k]
            targets = self.targets_nonzero[:, k]
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets)
        return ap

    @staticmethod
    def average_precision(output, target):
        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        if pos_count==0:
            precision_at_i = 0
        else:
            precision_at_i /= pos_count
        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
        return self.evaluation(scores, targets)

    def evaluation(self, scores_, targets_):
        """Returns the model's OP, OR, OF1, CP, CR, CF1, EP, ER, EF1
            Return:
            OP, OR, OF1, CP, CR, CF1, EP, ER, EF1: 9 Float tensors
        """
        eps = 1e-10
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores >= 0)
            Nc[k] = np.sum(targets * (scores >= 0))

        OP = np.sum(Nc) / (np.sum(Np) + eps)
        OR = np.sum(Nc) / (np.sum(Ng) + eps)
        OF1 = (2 * OP * OR) / (OP + OR + eps)

        CP = Nc / (Np + eps)
        CR = Nc / (Ng + eps)
        CF1 = (2 * CP * CR) / ( CP + CR + eps)

        CP = np.mean(CP)
        CR = np.mean(CR)
        CF1 = np.mean(CF1)

        # calculate example-based
        pred = np.int8(np.round(1/(1+np.exp(-scores_))))
        gt = np.int8(np.round(targets_))
        TP_e = np.float32(np.sum(((pred+gt) == 2), 1))
        FP_e = np.float32(np.sum(((pred-gt) == 1), 1))
        FN_e = np.float32(np.sum(((pred-gt) == -1), 1))
        TN_e = np.float32(np.sum(((pred+gt) == 0), 1))

        # clear TP_e is 0, assign it some value and latter assign zero
        Nc = TP_e
        Np = TP_e + FP_e
        Ng = TP_e + FN_e

        EP = Nc / (Np + eps)
        ER = Nc / (Ng + eps)
        EF1 = (2 * EP * ER) / (EP + ER + eps)

        EP = np.mean(EP)
        ER = np.mean(ER)
        EF1 = np.mean(EF1)

        return OP, OR, OF1, CP, CR, CF1, EP, ER, EF1 


