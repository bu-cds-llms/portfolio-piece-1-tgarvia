"""Attention helpers.

This file implements the core ideas from the Transformer attention mechanism:
1) Scaled dot-product attention
2) Multi-head attention

Mask convention:
- `mask` is a boolean tensor where True means "ignore this position".
- The mask must be broadcastable to the attention score shape.

Typical use:
- Self-attention: Q=K=V=x
- Cross-attention: Q comes from one sequence, K/V from another
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention.

    The formula is:
        Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k)) V

    Args:
        q: Query tensor with shape (..., q_len, d_k).
        k: Key tensor with shape (..., k_len, d_k).
        v: Value tensor with shape (..., k_len, d_v).
        mask: Optional boolean mask broadcastable to (..., q_len, k_len).
            True entries are masked out by setting their score to -inf.

    Returns:
        Tuple (out, attn):
            out: Attention output with shape (..., q_len, d_v).
            attn: Attention weights with shape (..., q_len, k_len).

    Notes:
        - For padding masks, a common shape is (batch, 1, 1, k_len) so it
          broadcasts across heads and query length.
    """
    if q.size(-1) != k.size(-1):
        raise ValueError("q and k must have the same last dimension (d_k).")

    d_k = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)  # (..., q_len, k_len)

    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("mask must be a boolean tensor.")
        scores = scores.masked_fill(mask, float("-inf"))

    attn = F.softmax(scores, dim=-1)  # (..., q_len, k_len)
    out = attn @ v                    # (..., q_len, d_v)
    return out, attn


class MultiHeadAttention(nn.Module):
    """Multi-head attention.

    Args:
        d_model: Embedding/model dimension.
        num_heads: Number of attention heads.

    Input:
        x_q: (batch, q_len, d_model)
        x_k: (batch, k_len, d_model)
        x_v: (batch, k_len, d_model)

    Returns:
        out: (batch, q_len, d_model)
        attn: (batch, num_heads, q_len, k_len)
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers to create Q, K, V (and final output projection)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split (B, T, d_model) into (B, H, T, d_k)."""
        b, t, _ = x.shape
        x = x.view(b, t, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine (B, H, T, d_k) back into (B, T, d_model)."""
        b, h, t, dk = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * dk)

    def forward(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
        x_v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run attention.

        Args:
            x_q: Query inputs, shape (batch, q_len, d_model).
            x_k: Key inputs, shape (batch, k_len, d_model).
            x_v: Value inputs, shape (batch, k_len, d_model).
            mask: Optional boolean mask broadcastable to
                (batch, num_heads, q_len, k_len). True = masked out.

        Returns:
            Tuple (out, attn):
                out: (batch, q_len, d_model)
                attn: (batch, num_heads, q_len, k_len)
        """
        # Project to Q, K, V
        q = self._split_heads(self.w_q(x_q))  # (B, H, q_len, d_k)
        k = self._split_heads(self.w_k(x_k))  # (B, H, k_len, d_k)
        v = self._split_heads(self.w_v(x_v))  # (B, H, k_len, d_k)

        # Apply attention in each head
        out, attn = scaled_dot_product_attention(q, k, v, mask=mask)  # out: (B,H,q_len,d_k)

        # Merge heads + final linear layer
        out = self._combine_heads(out)  # (B, q_len, d_model)
        out = self.w_o(out)
        return out, attn

    def self_attention(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience wrapper for self-attention (Q=K=V=x)."""
        return self.forward(x, x, x, mask=mask)
