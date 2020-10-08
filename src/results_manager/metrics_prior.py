import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from torch.distributions import Dirichlet
from sklearn import metrics


def accuracy(Y, alpha):
    corrects = (Y.squeeze() == alpha.max(-1)[1]).type(torch.DoubleTensor)
    accuracy = corrects.sum() / corrects.size(0)
    return accuracy.cpu().detach().numpy()


def confidence(Y, alpha, score_type='AUROC', uncertainty_type='aleatoric'):
    corrects = (Y.squeeze() == alpha.max(-1)[1]).cpu().detach().numpy()
    if uncertainty_type == 'epistemic':
        scores = alpha.max(-1)[0].cpu().detach().numpy()
    elif uncertainty_type == 'aleatoric':
        p = torch.nn.functional.normalize(alpha, p=1, dim=-1)
        scores = p.max(-1)[0].cpu().detach().numpy()

    if score_type == 'AUROC':
        fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
        return metrics.auc(fpr, tpr)
    elif score_type == 'APR':
        return metrics.average_precision_score(corrects, scores)
    else:
        raise NotImplementedError


def brier_score(Y, alpha):
    batch_size = alpha.size(0)

    p = torch.nn.functional.normalize(alpha, p=1, dim=-1)
    indices = torch.arange(batch_size)
    p[indices, Y.squeeze()] -= 1
    brier_score = p.norm(dim=-1).mean().cpu().detach().numpy()
    return brier_score


# OOD detection metrics
def anomaly_detection(alpha, ood_alpha, score_type='AUROC', uncertainty_type='aleatoric'):
    if uncertainty_type == 'epistemic':
        scores = alpha.sum(-1).cpu().detach().numpy()
    elif uncertainty_type == 'aleatoric':
        p = torch.nn.functional.normalize(alpha, p=1, dim=-1)
        scores = p.max(-1)[0].cpu().detach().numpy()

    if uncertainty_type == 'epistemic':
        ood_scores = ood_alpha.sum(-1).cpu().detach().numpy()
    elif uncertainty_type == 'aleatoric':
        p = torch.nn.functional.normalize(ood_alpha, p=1, dim=-1)
        ood_scores = p.max(-1)[0].cpu().detach().numpy()

    corrects = np.concatenate([np.ones(alpha.size(0)), np.zeros(ood_alpha.size(0))], axis=0)
    scores = np.concatenate([scores, ood_scores], axis=0)

    if score_type == 'AUROC':
        fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
        return metrics.auc(fpr, tpr)
    elif score_type == 'APR':
        return metrics.average_precision_score(corrects, scores)
    else:
        raise NotImplementedError


# Entropy
def entropy(alpha, uncertainty_type, n_bins=10, plot=True):
    entropy = []

    if uncertainty_type == 'aleatoric':
        p = torch.nn.functional.normalize(alpha, p=1, dim=-1)
        entropy.append(Categorical(p).entropy().squeeze().cpu().detach().numpy())
    elif uncertainty_type == 'epistemic':
        entropy.append(Dirichlet(alpha).entropy().squeeze().cpu().detach().numpy())

    if plot:
        plt.hist(entropy, n_bins)
        plt.show()
    return entropy
