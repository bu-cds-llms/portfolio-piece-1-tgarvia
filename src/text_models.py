"""Text utilities and simple neural baselines.

This module contains:
- A minimal tokenizer (no external NLP models required)
- Vocabulary building (word -> id mapping)
- Encoding text into fixed-length tensors with padding
- Two baseline classifiers in PyTorch:
  1) Average-of-embeddings classifier
  2) Self-attention pooling classifier (interpretable token weights)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections import Counter
from typing import List, Dict, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Vocab:
    """A simple vocabulary container.

    Attributes:
        stoi: Mapping from token string to integer id (string-to-index).
        itos: List mapping from integer id to token string (index-to-string).
        pad_id: Integer id used for padding tokens ("<pad>").
        unk_id: Integer id used for unknown tokens ("<unk>").
    """

    stoi: Dict[str, int]
    itos: List[str]
    pad_id: int
    unk_id: int


def basic_tokenize(text: str) -> List[str]:
    """Tokenize text into a list of whitespace-separated tokens.

    This tokenizer is intentionally simple and robust:
    - Lowercases text
    - Removes HTML tags
    - Replaces most punctuation with spaces (keeps apostrophes)
    - Collapses repeated whitespace

    Args:
        text: Raw input string.

    Returns:
        List of token strings.

    Notes:
        This is not a linguistically perfect tokenizer. For this portfolio
        project, the goal is reproducibility and simplicity.
    """
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # remove HTML tags
    text = re.sub(r"[^a-z0-9'\s]", " ", text)     # keep letters, digits, apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def build_vocab(texts: List[str], max_size: int = 20000, min_freq: int = 2) -> Vocab:
    """Build a vocabulary from a list of texts.

    Args:
        texts: List of raw documents.
        max_size: Maximum vocabulary size (including special tokens).
        min_freq: Minimum frequency required to include a token.

    Returns:
        A Vocab object with stoi/itos and special token ids.

    Notes:
        We always include two special tokens:
        - "<pad>" at index 0
        - "<unk>" at index 1

        Tokens are added in descending frequency order.
    """
    counter = Counter()
    for t in texts:
        counter.update(basic_tokenize(t))

    specials = ["<pad>", "<unk>"]

    # Keep only tokens meeting min_freq and not in specials
    words = [w for w, f in counter.most_common() if f >= min_freq and w not in specials]

    # Reserve room for specials
    words = words[: max(0, max_size - len(specials))]

    itos = specials + words
    stoi = {w: i for i, w in enumerate(itos)}

    return Vocab(stoi=stoi, itos=itos, pad_id=stoi["<pad>"], unk_id=stoi["<unk>"])


def encode(text: str, vocab: Vocab, max_len: int = 256) -> torch.Tensor:
    """Convert one document into a fixed-length tensor of token ids.

    Steps:
    1) Tokenize text
    2) Truncate to max_len
    3) Map tokens to ids (unknown tokens -> unk_id)
    4) Pad with pad_id up to max_len

    Args:
        text: Raw document string.
        vocab: Vocab containing token->id mapping.
        max_len: Output sequence length.

    Returns:
        A 1D LongTensor of shape (max_len,).

    Example:
        ids = encode("This movie was great!", vocab, max_len=8)
        # tensor([..., ..., ..., 0, 0, 0, 0, 0])  # padded
    """
    tokens = basic_tokenize(text)[:max_len]
    ids = [vocab.stoi.get(tok, vocab.unk_id) for tok in tokens]

    if len(ids) < max_len:
        ids += [vocab.pad_id] * (max_len - len(ids))

    return torch.tensor(ids, dtype=torch.long)


class AvgEmbClassifier(nn.Module):
    """Average-of-embeddings classifier.

    This is a very common neural baseline:
    - Embed each token -> (batch_size, seq_len, embedding_dim)
    - Average embeddings over non-padding tokens -> (batch_size, embedding_dim)
    - Linear layer -> logits -> (batch_size, num_classes)

    Args:
        vocab_size: Size of vocabulary.
        d_model: Embedding dimension.
        num_classes: Number of output classes (2 for sentiment).
        pad_id: Token id used for padding.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_classes: int = 2,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Token id tensor of shape (batch, seq_len).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        emb = self.emb(x)  # (batch_size, seq_len, embedding_dim)

        # Mask out padding positions so they do not affect the average.
        mask = (x != self.pad_id).float().unsqueeze(-1)  # (batch_size, seq_len, 1)
        summed = (emb * mask).sum(dim=1)                 # (batch_size, embedding_dim)
        denom = mask.sum(dim=1).clamp_min(1.0)           # (batch_size, 1)

        avg = summed / denom                              # (batch_size, embedding_dim)
        return self.fc(avg)


class SelfAttentionPoolingClassifier(nn.Module):
    """Self-attention pooling classifier (interpretable baseline).

    Idea:
    - Embed tokens -> (batch_size, seq_len, embedding_dim)
    - Learn a single query vector q (embedding_dim,)
    - Compute token scores: score_i = (e_i · q) / sqrt(embedding_dim)
    - Softmax over tokens -> attention weights (batch_size, seq_len)
    - Weighted sum of embeddings -> pooled vector (batch_size, embedding_dim)
    - Linear classifier -> logits

    This is not a full Transformer. It is a lightweight way to:
    - give the model flexibility beyond simple averaging
    - expose token-level importance weights for interpretation

    Args:
        vocab_size: Vocabulary size.
        d_model: Embedding dimension.
        num_classes: Number of output classes.
        pad_id: Padding token id.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_classes: int = 2,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # Learned query for scoring each token embedding.
        self.query = nn.Parameter(torch.randn(d_model))  # (embedding_dim,)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Token id tensor of shape (batch, seq_len).
            return_attn: If True, also return attention weights.

        Returns:
            If return_attn is False:
                logits: (batch, num_classes)
            If return_attn is True:
                (logits, attn)
                logits: (batch, num_classes)
                attn: (batch, seq_len) attention weights over tokens

        Notes:
            Padding positions are masked out so they receive ~0 attention.
        """
        emb = self.emb(x)  # (batch_size, seq_len, embedding_dim)

        # Score each token embedding against the learned query.
        scores = (emb @ self.query) / (emb.size(-1) ** 0.5)  # (batch_size, seq_len)

        # Mask out padding tokens so they don't get attention mass.
        scores = scores.masked_fill(x == self.pad_id, float("-inf"))

        attn = F.softmax(scores, dim=1)  # (batch_size, seq_len)

        # Weighted pooling
        pooled = (attn.unsqueeze(-1) * emb).sum(dim=1)  # (batch_size, embedding_dim)
        logits = self.fc(pooled)                        # (batch_size, num_classes)

        if return_attn:
            return logits, attn
        return logits
