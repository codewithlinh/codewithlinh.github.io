import torch


def form_triplet(x, y):
    """Triplet loss from two embedding tensors. Embeddings with same batch are similar and otherwise"""
    b, embedding_size = x.shape
    perms = b ** 2

    labels = [0] * perms
    similar_idxs = [(0 + i * b) + i for i in range(b)]
    for idx in similar_idxs:
        labels[idx] = 1
    labels = torch.Tensor(labels).type(torch.BoolTensor)
    anchors = x.repeat(b, 1)[~labels]
    negatives = torch.cat([y[i:].repeat(b, 1) for i in range(b)])[~labels]
    positives = y.repeat(b, 1)[~labels]

    return anchors, positives, negatives


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, distance=lambda x, y: torch.pow(x - y, 2).sum(1),
                 margin=1.0, mode='pair', batch_size=None, temperature=0.5):
        super(ContrastiveLoss, self).__init__()

        self.margin = margin
        self.mode = mode
        self.distance = distance

    def forward_triplet(self, x, y):
        a, p, n = form_triplet(x, y)
