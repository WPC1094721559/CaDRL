import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, n_src_vocab, n_relations, d_word_vec, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, pad_idx, dropout=0.1, n_position=1024, scale_emb=False):
        super().__init__()

        self.d_model = d_model
        self.n_relations = n_relations
        self.src_pad_idx = pad_idx
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.relation_emb = nn.Embedding(n_relations, d_word_vec)

        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb

    def forward(self, src_seq, src_mask=None, relation_ids=None, return_attns=False, length=None):
        batch_size = src_seq.size(0)
        doc_output = self.src_word_emb(src_seq)

        if self.scale_emb:
            doc_output *= self.d_model ** 0.5
        doc_output = self.dropout(self.position_enc(doc_output, length))
        doc_output = self.layer_norm(doc_output)
        if relation_ids is None:
            relation_ids = torch.arange(self.n_relations, device=src_seq.device)
            relation_ids = relation_ids.unsqueeze(0).expand(batch_size, -1)

        rel_output = self.relation_emb(relation_ids)

        if self.scale_emb:
            rel_output *= self.d_model ** 0.5

        rel_output = self.dropout(rel_output)
        rel_output = self.layer_norm(rel_output)

        if src_mask is None:
            src_mask = get_pad_mask(src_seq, self.src_pad_idx).squeeze(-2)

        enc_slf_attn_list = []

        for enc_layer in self.layer_stack:
            doc_output, rel_output = enc_layer(
                doc_output, rel_output, slf_attn_mask=src_mask
            )
            if return_attns:
                enc_slf_attn_list.append(None)

        if return_attns:
            return doc_output, rel_output, enc_slf_attn_list
        return doc_output, rel_output
