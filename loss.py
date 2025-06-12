import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output


class JointLoss(nn.Module):

    def __init__(self, rule_weight):
        super().__init__()

        self.atlop_criterion = ATLoss()
        self.rule_weight = rule_weight
        self.current_epoch = 0

    def forward(self, relation_logits, labels, rule_loss=None,
                generated_rules=None, mode='train'):

        labels_tensor = self._prepare_labels(labels, relation_logits.device)
        atlop_loss = self.atlop_criterion(relation_logits, labels_tensor)
        current_rule_weight = self.rule_weight
        if rule_loss is not None and mode == 'train':
            total_loss = atlop_loss + current_rule_weight * rule_loss
        else:
            total_loss = atlop_loss
            if rule_loss is None:
                rule_loss = torch.tensor(0.0, device=atlop_loss.device)
        loss_dict = {
            'total_loss': total_loss,
            'atlop_loss': atlop_loss,
            'rule_loss': rule_loss,
            'rule_weight': current_rule_weight,
            'rule_contribution': current_rule_weight * rule_loss if rule_loss is not None else 0.0
        }

        return loss_dict

    def _prepare_labels(self, labels, device):
        if isinstance(labels, list):
            batch_labels = []
            for doc_labels in labels:
                if isinstance(doc_labels, list):
                    doc_tensor = torch.tensor(doc_labels, dtype=torch.float)
                    batch_labels.append(doc_tensor)
                else:
                    batch_labels.append(doc_labels)
            labels_tensor = torch.cat(batch_labels, dim=0).to(device)
        else:
            labels_tensor = labels.to(device)

        return labels_tensor.float()

    def _get_current_rule_weight(self):
        return self.rule_weight

    def set_rule_weight(self, weight):
        self.rule_weight = weight

    def _compute_rule_statistics(self, generated_rules):
        if not generated_rules:
            return {
                'num_rules': 0,
                'avg_rule_score': 0.0,
                'avg_rule_length': 0.0
            }

        num_rules = len(generated_rules)
        rule_scores = [rule.get('score', 0.0) for rule in generated_rules]
        avg_rule_score = sum(rule_scores) / num_rules if rule_scores else 0.0
        rule_lengths = [len(rule.get('body', [])) for rule in generated_rules]
        avg_rule_length = sum(rule_lengths) / num_rules if rule_lengths else 0.0

        return {
            'num_rules': num_rules,
            'avg_rule_score': avg_rule_score,
            'avg_rule_length': avg_rule_length
        }
