import paddle.nn as nn
import paddle.nn.functional as F


class CustomLoss(nn.Layer):
    def __init__(self, weight=None, reduction='mean'):
        super(CustomLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, label):
        # 计算 softmax 之后的概率
        log_probs = F.log_softmax(input, axis=-1)

        # 计算 NLLLoss
        loss = F.nll_loss(log_probs, label, weight=self.weight, reduction=self.reduction)

        return loss
