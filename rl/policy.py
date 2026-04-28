"""Custom feature extractor for the Clash Royale RL policy.

The default `CombinedExtractor` from SB3 flattens every non-image Box
into raw floats. That's fine for `elixir` and `hand_affordable` (already
continuous-ish), but it's pathological for `hand` (4 int32 card IDs in
[0, 77]) and `det_class` (16 int32 troop class IDs): card 1 (Skeletons)
and card 50 (Mega Knight) become "near" each other simply because their
ids are close, which they are not. The policy can never learn per-card
behavior under that representation.

This extractor:

  * embeds `hand` through nn.Embedding(NUM_CARD_IDS, EMB_DIM);
  * embeds `det_class` through the SAME embedding table — so a Knight
    in hand and a Knight on the board share representation;
  * embeds `det_side` through a small categorical embedding;
  * concats class-emb + pos + side-emb + conf into a per-detection
    vector, runs it through a small MLP, then mean+max-pools across the
    detection slots weighted by confidence (permutation-invariant);
  * keeps NatureCNN on `pixels` and `troops` (image keys);
  * flattens `hand_affordable` and `elixir`.

Frame-stacking note: SB3's VecFrameStack concatenates non-image Box
keys along the last axis, multiplying their size by n_stack. We use
the original (unstacked) sizes from rl.bridge as constants and slice
the *last segment* of each obs (= most recent frame) for the embedding
branches; the policy still sees historical pixels through NatureCNN.
This keeps the extractor robust to whatever n_stack the env uses.
"""
from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    NatureCNN,
)
from torch import nn

from rl.bridge import NUM_CARD_IDS, NUM_DETECTION_SLOTS, NUM_SIDES

# Embedding dimensions. Small (16-d) — the card vocab is only 78 entries,
# and a tiny embedding regularizes well with limited training data.
CARD_EMB_DIM = 16
SIDE_EMB_DIM = 4

# Per-detection MLP output (after class+pos+side+conf concat).
DET_FEATURE_DIM = 32

# NatureCNN output for each image key (matches SB3 default).
CNN_OUT_DIM = 256


class CardAwareExtractor(BaseFeaturesExtractor):
    """Dict-obs feature extractor with shared card / troop embeddings."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_output_dim: int = CNN_OUT_DIM,
    ) -> None:
        # We set features_dim correctly at the end after computing all
        # branch sizes; pass a placeholder of 1 to satisfy super().
        super().__init__(observation_space, features_dim=1)

        self.card_embedding = nn.Embedding(NUM_CARD_IDS, CARD_EMB_DIM)
        self.side_embedding = nn.Embedding(NUM_SIDES, SIDE_EMB_DIM)

        # Per-detection MLP: in = card_emb + pos(2) + side_emb + conf(1)
        det_in = CARD_EMB_DIM + 2 + SIDE_EMB_DIM + 1
        self.det_mlp = nn.Sequential(
            nn.Linear(det_in, DET_FEATURE_DIM),
            nn.ReLU(),
            nn.Linear(DET_FEATURE_DIM, DET_FEATURE_DIM),
            nn.ReLU(),
        )

        # Image branches (per-key NatureCNN). After SB3's VecTransposeImage
        # these are CHW, with C = original_channels * n_stack — exactly
        # what NatureCNN wants.
        self.pixels_cnn = NatureCNN(
            observation_space.spaces["pixels"],
            features_dim=cnn_output_dim,
            normalized_image=False,
        )
        self.troops_cnn = NatureCNN(
            observation_space.spaces["troops"],
            features_dim=cnn_output_dim,
            normalized_image=False,
        )

        # Sizes contributed by each branch in the final concat:
        hand_size = 4 * CARD_EMB_DIM
        # Detection summary = mean-pool + max-pool of per-det features.
        det_summary_size = 2 * DET_FEATURE_DIM
        misc_size = 4 + 1  # hand_affordable (4) + elixir (1) — current frame only
        total = (
            cnn_output_dim  # pixels
            + cnn_output_dim  # troops
            + hand_size
            + det_summary_size
            + misc_size
        )
        self._features_dim = total

        # Cache the unstacked widths for slicing in forward(). These are
        # the per-frame sizes; the actual obs is n_stack * width along
        # its last axis, so [:, -width:] is "current frame".
        self._hand_width = 4
        self._aff_width = 4
        self._elixir_width = 1
        self._det_count = NUM_DETECTION_SLOTS
        self._det_pos_width = 2

    def forward(
        self, observations: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # Image branches: NatureCNN expects float CHW input; SB3 has
        # already transposed to CHW, we just need to cast.
        pix_feat = self.pixels_cnn(observations["pixels"].float())
        trp_feat = self.troops_cnn(observations["troops"].float())

        # Hand embedding — slice last frame from the stacked obs.
        hand_ids = observations["hand"][:, -self._hand_width:].long()
        hand_emb = self.card_embedding(hand_ids)
        hand_feat = hand_emb.flatten(start_dim=1)  # (B, 4*emb)

        # Per-detection encoding — last frame of each stacked key.
        det_class = observations["det_class"][:, -self._det_count:].long()
        det_pos = observations["det_pos"][:, :, -self._det_pos_width:].float()
        det_side = observations["det_side"][:, -self._det_count:].long()
        det_conf = observations["det_conf"][:, -self._det_count:].float()

        det_class_emb = self.card_embedding(det_class)
        det_side_emb = self.side_embedding(det_side)
        det_input = torch.cat(
            [
                det_class_emb,
                det_pos,
                det_side_emb,
                det_conf.unsqueeze(-1),
            ],
            dim=-1,
        )  # (B, K, det_in)

        B, K, _ = det_input.shape
        det_h = self.det_mlp(det_input.view(B * K, -1)).view(B, K, -1)

        # Confidence-weighted mean + masked max pool. Empty slots have
        # conf=0 so they contribute zero to either pool.
        weights = det_conf.unsqueeze(-1)
        weight_norm = weights.sum(dim=1).clamp(min=1e-6)
        det_mean = (det_h * weights).sum(dim=1) / weight_norm

        det_h_masked = det_h * (det_conf > 0).unsqueeze(-1).float()
        det_max = det_h_masked.max(dim=1).values
        det_feat = torch.cat([det_mean, det_max], dim=-1)

        # Misc scalars — last frame of each.
        affordable = observations["hand_affordable"][:, -self._aff_width:].float()
        elixir = observations["elixir"][:, -self._elixir_width:].float()
        misc = torch.cat([affordable, elixir], dim=-1)

        return torch.cat(
            [pix_feat, trp_feat, hand_feat, det_feat, misc], dim=-1
        )
