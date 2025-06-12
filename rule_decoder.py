import torch
import torch.nn as nn
import torch.nn.functional as F


class RuleDecoder(nn.Module):
    def __init__(self, d_model, n_relations, max_rule_length=2,
                 n_head=8, dropout=0.1, gamma=1e-6):
        super().__init__()

        self.d_model = d_model
        self.n_relations = n_relations
        self.max_rule_length = max_rule_length  # L
        self.gamma = gamma

        self.na_embedding = nn.Parameter(torch.randn(d_model))

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )

        self.relation_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_relations)
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, rel_encoding, entity_adjacency_matrices=None,
                target_triples=None, mode='inference'):

        batch_size, n_relations, d_model = rel_encoding.shape

        generated_rules = []
        all_rule_scores = []
        total_rule_loss = 0.0

        for batch_idx in range(batch_size):
            for rel_idx in range(n_relations):
                rule_head = rel_idx
                rel_encoding_single = rel_encoding[batch_idx, rel_idx:rel_idx + 1]

                rule_body, rule_score = self._generate_rule_sequence(
                    rel_encoding_single, rule_head
                )

                if len(rule_body) > 0:
                    rule = {
                        'head': rule_head,
                        'body': rule_body,
                        'score': rule_score.item(),
                        'batch_idx': batch_idx
                    }
                    generated_rules.append(rule)
                    all_rule_scores.append(rule_score)

        if mode == 'training' and target_triples is not None:
            rule_loss = self._compute_differentiable_rule_loss(
                generated_rules, entity_adjacency_matrices, target_triples
            )
            total_rule_loss = rule_loss

        return generated_rules, all_rule_scores, total_rule_loss

    def _generate_rule_sequence(self, rel_encoding_single, rule_head):
        device = rel_encoding_single.device

        current_state = self.na_embedding.unsqueeze(0).unsqueeze(0)

        rule_body = []
        step_probs = []

        for step in range(self.max_rule_length):
            attn_output, _ = self.cross_attention(
                query=current_state,
                key=rel_encoding_single.unsqueeze(0),
                value=rel_encoding_single.unsqueeze(0)
            )

            attn_output = self.layer_norm(attn_output + current_state)

            relation_logits = self.relation_mlp(attn_output.squeeze(1))
            relation_probs = F.softmax(relation_logits, dim=-1)

            max_prob, max_idx = torch.max(relation_probs, dim=-1)
            selected_relation = max_idx.item()

            if selected_relation == 0 or max_prob.item() < 0.1:
                break

            rule_body.append(selected_relation)
            step_probs.append(max_prob)

            relation_emb = rel_encoding_single
            current_state = torch.cat([current_state, relation_emb.unsqueeze(0)], dim=1)

        if step_probs:
            rule_score = torch.prod(torch.stack(step_probs))
        else:
            rule_score = torch.tensor(0.0, device=device)

        return rule_body, rule_score

    def _compute_differentiable_rule_loss(self, generated_rules,
                                          entity_adjacency_matrices, target_triples):
        if not generated_rules or entity_adjacency_matrices is None:
            return torch.tensor(0.0)

        device = entity_adjacency_matrices.device
        n_entities = entity_adjacency_matrices.size(1)
        total_loss = 0.0
        valid_triples = 0

        for h, r, t in target_triples:
            if h >= n_entities or t >= n_entities:
                continue
            relevant_rules = [rule for rule in generated_rules if rule['head'] == r]
            if not relevant_rules:
                continue
            triple_scores = []

            for rule in relevant_rules:
                rule_body = rule['body']
                if not rule_body:
                    continue

                xi = torch.zeros(n_entities, device=device)
                xi[h] = 1.0

                for step, rel_id in enumerate(rule_body):
                    if rel_id < entity_adjacency_matrices.size(0):
                        adjacency_matrix = entity_adjacency_matrices[rel_id]
                        xi = torch.matmul(xi.unsqueeze(0), adjacency_matrix.float()).squeeze(0)

                target_vector = torch.zeros(n_entities, device=device)
                target_vector[t] = 1.0

                xi_stabilized = torch.clamp(xi, min=self.gamma)
                log_xi = torch.log(xi_stabilized)
                score = torch.dot(target_vector, log_xi)
                triple_scores.append(score)

            if triple_scores:
                max_score = torch.max(torch.stack(triple_scores))
                loss = 1.0 - torch.sigmoid(max_score)
                total_loss += loss
                valid_triples += 1

        if valid_triples > 0:
            return total_loss / valid_triples
        else:
            return torch.tensor(0.0, device=device)
