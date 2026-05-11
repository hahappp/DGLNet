# Loss/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLabelSmooth(nn.Module):
    """
    Standard CrossEntropy Loss with Label Smoothing regularization.
    Used as the primary classification loss for robust facial expression recognition.
    """

    def __init__(self, num_classes: int, epsilon: float = 0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = self.logsoftmax(inputs)

        # Convert target labels to one-hot encoding
        targets_onehot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)

        # Apply label smoothing formula
        smooth_targets = (1 - self.epsilon) * targets_onehot + self.epsilon / self.num_classes

        # Calculate cross entropy loss
        loss = (-smooth_targets * log_probs).mean(0).sum()
        return loss


class SampleWeightedFocalContrastiveLoss(nn.Module):
    """
    Advanced auxiliary contrastive loss mentioned in the manuscript.

    NOTE: To strictly comply with the double-blind peer review policy, the proprietary
    sample weight calculation and contrastive margin mechanisms have been temporarily
    omitted in this review-version code. The implementation currently falls back to
    the standard CrossEntropyLoss and will be fully released upon paper acceptance.
    """

    def __init__(self):
        super(SampleWeightedFocalContrastiveLoss, self).__init__()
        self.base_loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Fallback to base cross entropy during the peer-review phase
        return self.base_loss(logits, labels)


class SoftHGRLoss(nn.Module):
    """
    Soft-HGR Loss for maximizing multi-modal and cross-stage feature correlation.

    NOTE: The core feature-correlation optimization and matrix projection logic are
    temporarily hidden to preserve author anonymity and protect intellectual property
    during the double-blind peer-review process. It currently returns a zero-valued
    differentiable tensor as a placeholder.
    """

    def __init__(self):
        super(SoftHGRLoss, self).__init__()

    def forward(self, *args) -> torch.Tensor:
        # Return a zero-gradient placeholder during the peer-review phase
        return torch.tensor(0.0, requires_grad=True).to(args[0].device)