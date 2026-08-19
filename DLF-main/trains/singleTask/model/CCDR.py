import torch
import torch.nn as nn


class CREDDisentangler(nn.Module):
    """Consensus-residual effect-guided disentanglement."""

    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.shared_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(3)
        ])
        self.specific_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(3)
        ])
        self.leave_one_out_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=False,
            )
            for _ in range(3)
        ])
        self.consensus_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(3)
        ])
        self.residual_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(3)
        ])
        self.residual_routers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
            for _ in range(3)
        ])

    def forward(self, shared_features, specific_features):
        shared_features = [
            norm(feature)
            for norm, feature in zip(self.shared_norms, shared_features)
        ]
        specific_features = [
            norm(feature)
            for norm, feature in zip(self.specific_norms, specific_features)
        ]

        consensus_parts = []
        support_parts = []
        conflict_parts = []
        route_gates = []
        reconstructed_parts = []

        for index in range(3):
            other_shared = torch.cat([
                feature
                for other_index, feature in enumerate(shared_features)
                if other_index != index
            ], dim=0)
            consensus, _ = self.leave_one_out_attentions[index](
                shared_features[index],
                other_shared,
                other_shared,
                need_weights=False,
            )
            consensus = self.consensus_norms[index](consensus)

            shared_deviation = shared_features[index] - consensus
            residual = self.residual_projections[index](torch.cat([
                specific_features[index],
                shared_deviation,
            ], dim=-1))
            route_gate = self.residual_routers[index](torch.cat([
                residual,
                consensus,
            ], dim=-1))
            support = route_gate * residual
            conflict = (1.0 - route_gate) * residual

            consensus_parts.append(consensus)
            support_parts.append(support)
            conflict_parts.append(conflict)
            route_gates.append(route_gate)
            reconstructed_parts.append(consensus + support + conflict)

        evidence_targets = [
            shared + specific
            for shared, specific in zip(shared_features, specific_features)
        ]
        return {
            'consensus_parts': consensus_parts,
            'support_parts': support_parts,
            'conflict_parts': conflict_parts,
            'route_gates': route_gates,
            'reconstructed_parts': reconstructed_parts,
            'evidence_targets': evidence_targets,
        }


class CCDR(nn.Module):
    """CRED decomposition followed by language-anchored conflict adjudication."""

    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.disentangler = CREDDisentangler(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.support_slot = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.reverse_slot = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.normal_(self.support_slot, std=0.02)
        nn.init.normal_(self.reverse_slot, std=0.02)

        self.evidence_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.evidence_norm = nn.LayerNorm(hidden_dim)
        self.consensus_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.support_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.adjudication_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(self, shared_features, specific_features):
        decomposition = self.disentangler(shared_features, specific_features)
        consensus_parts = decomposition['consensus_parts']
        support_parts = decomposition['support_parts']
        conflict_parts = decomposition['conflict_parts']

        consensus_summary = torch.stack(consensus_parts, dim=0).mean(dim=(0, 1))
        support_summary = torch.stack(support_parts, dim=0).mean(dim=(0, 1))
        language_consensus = consensus_parts[0].mean(dim=0)
        language_anchor = (
            consensus_parts[0] + support_parts[0]
        ).mean(dim=0)

        batch_size = shared_features[0].size(1)
        evidence_query = torch.cat([
            self.support_slot.expand(-1, batch_size, -1),
            self.reverse_slot.expand(-1, batch_size, -1),
        ], dim=0)
        evidence_query = evidence_query + (
            language_anchor + consensus_summary
        ).unsqueeze(0)
        nonverbal_conflict = torch.cat(conflict_parts[1:], dim=0)
        evidence, evidence_attention = self.evidence_attention(
            evidence_query,
            nonverbal_conflict,
            nonverbal_conflict,
            need_weights=True,
            average_attn_weights=False,
        )
        evidence = self.evidence_norm(evidence + evidence_query)
        support_evidence, reverse_evidence = evidence[0], evidence[1]

        consensus_logit = self.consensus_head(torch.cat([
            consensus_summary,
            language_consensus,
        ], dim=-1))
        support_delta = self.support_head(torch.cat([
            consensus_summary,
            support_summary,
        ], dim=-1))
        conflict_delta = self.adjudication_head(torch.cat([
            consensus_summary,
            language_anchor,
            support_evidence,
            reverse_evidence,
        ], dim=-1))
        conflict_energy = torch.stack([
            conflict.pow(2).mean(dim=(0, 2))
            for conflict in conflict_parts
        ], dim=-1)

        return {
            'output_logit': consensus_logit + support_delta + conflict_delta,
            'consensus_logit': consensus_logit,
            'support_delta': support_delta,
            'conflict_delta': conflict_delta,
            'conflict_energy': conflict_energy,
            'consensus_parts': consensus_parts,
            'support_parts': support_parts,
            'conflict_parts': conflict_parts,
            'route_gates': decomposition['route_gates'],
            'reconstructed_parts': decomposition['reconstructed_parts'],
            'evidence_targets': decomposition['evidence_targets'],
            'evidence_attention': evidence_attention,
        }