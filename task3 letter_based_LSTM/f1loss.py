import torch.nn as nn

class F1_Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(F1_Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, output, target):
        probas = nn.Sigmoid()(output)
        TP = (probas * target).sum(dim=1)
        precision = TP / (probas.sum(dim=1) + self.epsilon)
        recall = TP / (target.sum(dim=1) + self.epsilon)
        f1 = 2 * precision * recall / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()