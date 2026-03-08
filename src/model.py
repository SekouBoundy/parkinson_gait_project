"""
model.py — LSTM classifier for Parkinson's gait detection.

Architecture
------------
  Input     : (batch, window_size, features)   e.g. (32, 128, 19)
  LSTM      : 1–3 stacked layers, configurable hidden size
  Dropout   : applied between LSTM layers and before the head
  FC head   : hidden_size → 1  (single logit for BCEWithLogitsLoss)
  Output    : raw logit — apply sigmoid OUTSIDE training for probabilities

Why a single logit output?
  BCEWithLogitsLoss combines sigmoid + BCE in one numerically stable
  operation. Never apply sigmoid BEFORE the loss during training.
"""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    Stacked LSTM binary classifier.

    Parameters
    ----------
    input_size  : int   number of input features per time step  (19 for PhysioNet)
    hidden_size : int   number of hidden units in each LSTM layer
    num_layers  : int   number of stacked LSTM layers (1–3 recommended)
    dropout     : float dropout probability applied between layers and before head
    bidirectional: bool if True, use a bidirectional LSTM (doubles effective hidden size)
    """

    def __init__(
        self,
        input_size:   int   = 19,
        hidden_size:  int   = 128,
        num_layers:   int   = 2,
        dropout:      float = 0.3,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.directions    = 2 if bidirectional else 1

        # ── LSTM stack ────────────────────────────────────────────────────
        # dropout inside nn.LSTM only applies *between* layers (not after last)
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,          # input shape: (batch, seq, features)
            dropout     = lstm_dropout,
            bidirectional = bidirectional,
        )

        # ── Dropout before the classification head ────────────────────────
        self.dropout = nn.Dropout(dropout)

        # ── Classification head ───────────────────────────────────────────
        # Takes the last time-step hidden state → single logit
        self.fc = nn.Linear(hidden_size * self.directions, 1)

    # ─────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, input_size)

        Returns
        -------
        logits : torch.Tensor, shape (batch,)
            Raw unnormalised scores.  Apply sigmoid for probabilities.
        """
        # out: (batch, seq_len, hidden * directions)
        # _  : (h_n, c_n) — we only need the final hidden state
        out, _ = self.lstm(x)

        # Take only the last time step
        last = out[:, -1, :]                  # (batch, hidden * directions)
        last = self.dropout(last)

        logits = self.fc(last).squeeze(-1)    # (batch,)
        return logits

    # ─────────────────────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> None:
        """Print a concise model summary to stdout."""
        d = "Bi-LSTM" if self.bidirectional else "LSTM"
        print(f"\n  Model : {d}  ×{self.num_layers} layers")
        print(f"  Hidden size       : {self.hidden_size}")
        print(f"  Directions        : {self.directions}")
        print(f"  Trainable params  : {self.count_parameters():,}")
        print()