import torch.nn as nn
from torch import Tensor

import torch
import torch.nn.functional as F

def tp_fp_tn_fn(input, target):
    
    input = input.int()
    target = target.int()

    true_pred, false_pred = target == input, target != input
    pos_pred, neg_pred = input == 1, input == 0

    tp = (true_pred * pos_pred).sum()
    fp = (false_pred * pos_pred).sum()

    tn = (true_pred * neg_pred).sum()
    fn = (false_pred * neg_pred).sum()

    return tp, fp, tn, fn

def compute_metrics(input, target):
        assert input.size() == target.size()
        input = input[:, 0].contiguous().view(-1)
        target = target[:, 0].contiguous().view(-1)

        intersection = (input * target).sum()
        dsc = (2. * intersection) / (input.sum() + target.sum())
    
        tp, fp, fn, _ = tp_fp_tn_fn(input, target)
        return dsc.item(), tp.item(), fp.item(), fn.item()


class LogMetrics(nn.Module):

    def __init__(self, epsilon=1e-5)-> None:
        super(LogMetrics, self).__init__()
        self.epsilon = epsilon

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.size() == target.size()
        input = input[:, 0].contiguous().view(-1)
        target = target[:, 0].contiguous().view(-1)

        intersection = (input * target).sum()
        dsc = (2. * intersection + self.epsilon) / (
            input.sum() + target.sum() + self.epsilon
        )
        return dsc, tp_fp_tn_fn(input, target)

class DiceScore(nn.Module):

    def __init__(self, epsilon=1e-5)-> None:
        super(DiceScore, self).__init__()
        self.epsilon = epsilon

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.size() == target.size()
        input = input[:, 0].contiguous().view(-1)
        target = target[:, 0].contiguous().view(-1)

        intersection = (input * target).sum()
        dsc = (2. * intersection + self.epsilon) / (
            input.sum() + target.sum() + self.epsilon
        )
        return dsc

class DiceLoss(nn.Module):

    def __init__(self, epsilon=1)-> None:
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        #Hao's code start
        
        # target = torch.squeeze(target, 1)
        
        # target = torch.permute(target, (1, 0, 2, 3))
        # print("Input Size: "+str(input.size()))
        # print("Target Size: "+str(target.size()))
        # target =target.float()
        # Hao's code end
        assert input.size() == target.size()
        input = input[:, 0].contiguous().view(-1)
        target = target[:, 0].contiguous().view(-1)

        intersection = (input * target).sum()
        dsc = (2. * intersection + self.epsilon) / (
            input.sum() + target.sum() + self.epsilon
        )
        return 1. - dsc


class TverskyLoss(nn.Module):
    """
    https://towardsdatascience.com/dealing-with-class-imbalanced-image-datasets-1cbd17de76b5
    """
    
    def __init__(self, beta=0.3, epsilon=1e-5)-> None:
        super(TverskyLoss, self).__init__()
        self.epsilon=epsilon
        self.beta=beta

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.size() == target.size()
        input = input[:, 0].contiguous().view(-1)
        target = target[:, 0].contiguous().view(-1)

        intersection = (input * target).sum()

        tp = intersection
        fp = input.sum() - intersection
        fn = target.sum() - intersection

        tversky_index = (tp + self.epsilon)/(
            tp + (1-self.beta) * fn + self.beta * fp + self.epsilon
            )


        return 1. - tversky_index

class TverskyFocalLoss(nn.Module):
    # https://towardsdatascience.com/dealing-with-class-imbalanced-image-datasets-1cbd17de76b5
    
    def __init__(self, beta=0.3, gamma=1., epsilon=1e-5)-> None:
        super(TverskyFocalLoss, self).__init__()
        #self.loss_fn = FBeta(num_classes=None, beta=beta)
        self.epsilon=epsilon
        self.beta=beta
        self.gamma =gamma


    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.size() == target.size()
        input = input[:, 0].contiguous().view(-1)
        target = target[:, 0].contiguous().view(-1)

        intersection = (input * target).sum()

        tp = intersection
        fp = input.sum() - intersection
        fn = target.sum() - intersection

        tversky_index = (tp + self.epsilon)/(
            tp + (1-self.beta) * fn + self.beta * fp + self.epsilon
            )


        return (1. - tversky_index) ** self.gamma


class FBetaLoss(nn.Module):

    def __init__(self, beta=0.5)-> None:
        super(FBetaLoss, self).__init__()
        self.epsilon=1e-5
        self.beta=beta


    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.size() == target.size()
        input = input[:, 0].contiguous().view(-1)
        target = target[:, 0].contiguous().view(-1)

        intersection = (input * target).sum()

        tp = intersection
        fp = input.sum() - intersection
        fn = target.sum() - intersection

        fbeta_score = (1.0+self.beta**2)*(tp + self.epsilon)/(
            self.beta**2 * (fn + fp)+(1.0-self.beta**2) * tp +self.epsilon
            )


        return 1. - fbeta_score



class WeightedDiceLoss(nn.Module):

    def __init__(self, weight=0.9):
        super(WeightedDiceLoss, self).__init__()
        self.epsilon = 1.0
        self.weight = weight

    def forward(self, input, target):
        assert input.size() == target.size()
        input = input[:, 0].contiguous().view(-1)
        target = target[:, 0].contiguous().view(-1)

        intersection = (input * target).sum()
        dsc = (2. * intersection * self.weight + self.epsilon) / (
            input.sum() * (self.weight**2) + target.sum() + self.epsilon
        )
        return 1. - dsc

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):

        assert input.size() == target.size()
        input = input[:, 0].contiguous().view(-1)
        target = target[:, 0].contiguous().view(-1)

        bce_loss = F.binary_cross_entropy(input.squeeze(),  target.float())
        loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        return loss

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, input, target):

        assert input.size() == target.size()
        input = input[:, 0].contiguous().view(-1)
        target = target[:, 0].contiguous().view(-1)

        return F.binary_cross_entropy(input.squeeze(),  target.float())

class BCEDiceLoss(nn.Module):
    def __init__(self,epsilon=1e-5):
        super(BCEDiceLoss, self).__init__()
        self.epsilon=epsilon

    def forward(self, input, target):

        assert input.size() == target.size()
        input = input[:, 0].contiguous().view(-1)
        target = target[:, 0].contiguous().view(-1)

        intersection = (input * target).sum()
        dsc = (2. * intersection + self.epsilon) / (
            input.sum() + target.sum() + self.epsilon
        )
        dsc_loss =  1. - dsc

        bce_loss = F.binary_cross_entropy(input.squeeze(),  target.float())

        return (dsc_loss + bce_loss) / 2.0
