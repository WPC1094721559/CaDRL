import torch.nn as nn

from doc_rel_attention.RelationAttention import DocumentRelationAttention
from doc_rel_attention.utils import PositionedFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.dual_attn = DocumentRelationAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.doc_ffn = PositionedFeedForward(d_model, d_inner, dropout=dropout)
        self.rel_ffn = PositionedFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, doc_input, rel_input, slf_attn_mask=None):
        doc_output, rel_output = self.dual_attn(doc_input, rel_input, slf_attn_mask)
        doc_output = self.doc_ffn(doc_output)
        rel_output = self.rel_ffn(rel_output)

        return doc_output, rel_output
