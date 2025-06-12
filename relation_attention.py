import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DocumentRelationAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, smoothing_factor=1e-6):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.smoothing_factor = smoothing_factor

        self.w_q_doc = nn.Linear(d_model, n_head * d_k, bias=False)  # W_Q^i
        self.w_k_doc = nn.Linear(d_model, n_head * d_k, bias=False)  # W_K^j
        self.w_v_doc = nn.Linear(d_model, n_head * d_v, bias=False)  # W_V^j

        self.w_k_rel = nn.Linear(d_model, n_head * d_k, bias=False)  # W_K^r
        self.w_v_rel = nn.Linear(d_model, n_head * d_v, bias=False)  # W_V^r

        self.fc_doc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.fc_rel = nn.Linear(n_head * d_v, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm_doc = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_rel = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, doc_input, rel_input, mask=None):

        residual_doc = doc_input
        residual_rel = rel_input

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_doc, n_rel = doc_input.size(0), doc_input.size(1), rel_input.size(1)
        doc_q = self.w_q_doc(doc_input).view(sz_b, len_doc, n_head, d_k).transpose(1, 2)  # [B, H, L, d_k]
        doc_k = self.w_k_doc(doc_input).view(sz_b, len_doc, n_head, d_k).transpose(1, 2)  # [B, H, L, d_k]
        doc_v = self.w_v_doc(doc_input).view(sz_b, len_doc, n_head, d_v).transpose(1, 2)  # [B, H, L, d_v]
        rel_k = self.w_k_rel(rel_input).view(sz_b, n_rel, n_head, d_k).transpose(1, 2)  # [B, H, R, d_k]
        rel_v = self.w_v_rel(rel_input).view(sz_b, n_rel, n_head, d_v).transpose(1, 2)  # [B, H, R, d_v]
        sum_rel_k = rel_k.sum(dim=2, keepdim=True)  # [B, H, 1, d_k]

        fused_k = doc_k + sum_rel_k.squeeze(2).unsqueeze(2).expand(-1, -1, len_doc, -1)  # [B, H, L, d_k]

        doc_doc_scores = torch.matmul(doc_q, fused_k.transpose(2, 3)) / math.sqrt(d_k) + self.smoothing_factor

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            doc_doc_scores = doc_doc_scores.masked_fill(mask_expanded == 0, -1e9)

        alpha_doc_doc = F.softmax(doc_doc_scores, dim=-1)  # [B, H, L, L]
        alpha_doc_doc = self.dropout(alpha_doc_doc)
        rel_encoding_list = []

        for r_idx in range(n_rel):
            rel_k_single = rel_k[:, :, r_idx:r_idx + 1, :]  # [B, H, 1, d_k]
            rel_v_single = rel_v[:, :, r_idx:r_idx + 1, :]  # [B, H, 1, d_v]
            scores = torch.matmul(rel_k_single, doc_k.transpose(2, 3)) / math.sqrt(d_k) + self.smoothing_factor
            # scores: [B, H, 1, L]

            if mask is not None:
                mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
                scores = scores.masked_fill(mask_expanded == 0, -1e9)
            alpha = F.softmax(scores, dim=-1)  # [B, H, 1, L]
            alpha = self.dropout(alpha)
            doc_contribution = torch.matmul(alpha, doc_v)  # [B, H, 1, d_v]

            rel_contribution = rel_v_single  # [B, H, 1, d_v]
            psi_r_single = doc_contribution + rel_contribution  # [B, H, 1, d_v]
            rel_encoding_list.append(psi_r_single)
        psi_r = torch.cat(rel_encoding_list, dim=2)  # [B, H, R, d_v]
        psi_r = psi_r.transpose(1, 2).contiguous().view(sz_b, n_rel, -1)  # [B, R, H*d_v]
        rel_output = self.dropout(self.fc_rel(psi_r))
        rel_output = self.layer_norm_rel(rel_output + residual_rel)

        sum_rel_v = rel_v.sum(dim=2, keepdim=True).squeeze(2)  # [B, H, d_v]
        sum_rel_v_expanded = sum_rel_v.unsqueeze(2).expand(-1, -1, len_doc, -1)  # [B, H, L, d_v]
        fused_v = doc_v + sum_rel_v_expanded  # [B, H, L, d_v]
        psi_i = torch.matmul(alpha_doc_doc, fused_v)  # [B, H, L, d_v]
        psi_i = psi_i.transpose(1, 2).contiguous().view(sz_b, len_doc, -1)  # [B, L, H*d_v]
        doc_output = self.dropout(self.fc_doc(psi_i))
        doc_output = self.layer_norm_doc(doc_output + residual_doc)

        return doc_output, rel_output
