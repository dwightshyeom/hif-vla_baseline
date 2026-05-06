"""
LSTM Encoder for History Encoding

This module provides an LSTM-based encoder that processes sequential history
of observations (keypoints + agent positions) and outputs a latent representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMHistoryEncoder(nn.Module):
    """
    LSTM-based encoder for processing history sequences.
    
    Takes a sequence of observations (keypoints + agent position) and outputs
    a fixed-size latent representation that captures temporal dependencies.
    
    Args:
        input_dim: Dimension of each observation (e.g., 20 for 9*2 keypoints + 2 agent_pos)
        hidden_dim: Hidden dimension of LSTM
        num_layers: Number of LSTM layers
        latent_dim: Dimension of output latent representation
        dropout: Dropout probability for LSTM (applied between layers)
        bidirectional: Whether to use bidirectional LSTM
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 64,
        dropout: float = 0.1,
        bidirectional: bool = False,
        max_memory_length: int = None  # NEW: Control memory length for debugging
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.bidirectional = bidirectional
        self.max_memory_length = max_memory_length  # None/-1 = infinite, 1 = no memory, 2 = 2 steps, etc.
        
        # Input projection (optional, helps stabilize training)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True  # (batch, seq, feature)
        )
        
        # Output projection to latent space
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_output_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
    def forward(
        self,
        history_obs: torch.Tensor,
        history_mask: torch.Tensor = None,
        hidden_state: tuple = None,
        return_all_hidden: bool = True
    ):
        """
        Forward pass through LSTM encoder.
        
        Note: max_memory_length should be handled BEFORE calling this function
        (by the caller slicing the input). This function processes whatever
        sequence it receives.
        
        Args:
            history_obs: (B, H, D) - batch of history sequences
                B = batch size
                H = history length (should already be limited to max_memory_length by caller)
                D = observation dimension (e.g., 20)
            history_mask: (B, H) - mask indicating valid timesteps (1=valid, 0=padding)
            hidden_state: Optional (h, c) tuple of LSTM hidden states
                h: (num_layers * num_directions, B, hidden_dim)
                c: (num_layers * num_directions, B, hidden_dim)
                If None, initialized to zeros
            return_all_hidden: If True, return hidden states for all timesteps
        
        Returns:
            If return_all_hidden is False:
                latent: (B, latent_dim) - latent representation of history
                hidden_state: (h, c) tuple of updated LSTM hidden states
            If return_all_hidden is True:
                latent: (B, latent_dim) - latent from final timestep
                all_hidden: (B, H, latent_dim) - latent for all timesteps
                hidden_state: (h, c) tuple of updated LSTM hidden states
        """
        B, H, D = history_obs.shape
        
        # Project input
        x = self.input_proj(history_obs)  # (B, H, hidden_dim)
        
        # Pack sequence if mask is provided (for efficiency with variable lengths)
        if history_mask is not None:
            # Get actual lengths from mask
            lengths = history_mask.sum(dim=1).long().cpu()  # (B,)
            # Clamp to avoid zero-length sequences
            lengths = torch.clamp(lengths, min=1)
            
            # Pack padded sequence
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            
            # Pass through LSTM with optional hidden state
            lstm_out_packed, (h_n, c_n) = self.lstm(x_packed, hidden_state)
            
            # Unpack sequence - don't specify total_length to avoid issues with slicing
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out_packed, batch_first=True
            )  # (B, actual_H, hidden_dim * num_directions)
            
            # If the unpacked length is less than H (due to slicing), pad to H
            if lstm_out.shape[1] < H:
                padding = torch.zeros(
                    B, H - lstm_out.shape[1], lstm_out.shape[2],
                    device=lstm_out.device, dtype=lstm_out.dtype
                )
                lstm_out = torch.cat([lstm_out, padding], dim=1)
        else:
            # No packing, just pass through LSTM
            lstm_out, (h_n, c_n) = self.lstm(x, hidden_state)  # lstm_out: (B, H, hidden_dim * num_directions)
        
        # Extract final hidden state
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            h_final = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, hidden_dim * 2)
        else:
            h_final = h_n[-1]  # (B, hidden_dim)
        
        # Project to latent space
        latent = self.output_proj(h_final)  # (B, latent_dim)
        
        if return_all_hidden:
            # Also project all timesteps to latent space
            all_hidden = self.output_proj(lstm_out)  # (B, H, latent_dim)
            return latent, all_hidden, (h_n, c_n)
        
        return latent, (h_n, c_n)
    
    def reset(self, batch_size: int = 1, device: torch.device = None):
        """
        Initialize hidden states to zeros.
        
        Args:
            batch_size: Batch size for hidden state initialization
            device: Device to create hidden states on
            
        Returns:
            (h_0, c_0): Tuple of initialized hidden states
                Each has shape (num_layers * num_directions, batch_size, hidden_dim)
        """
        if device is None:
            device = next(self.parameters()).device
        
        num_directions = 2 if self.bidirectional else 1
        h_0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim, device=device)
        
        return (h_0, c_0)


# class LSTMHistoryEncoderWithAttention(nn.Module):
#     """
#     LSTM encoder with attention mechanism.
    
#     Instead of just using the final hidden state, this computes an attention-weighted
#     sum over all hidden states, allowing the model to focus on relevant parts of history.
#     """
    
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int = 128,
#         num_layers: int = 2,
#         latent_dim: int = 64,
#         dropout: float = 0.1,
#         bidirectional: bool = False,
#         attention_heads: int = 4
#     ):
#         super().__init__()
        
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.latent_dim = latent_dim
#         self.bidirectional = bidirectional
        
#         # Input projection
#         self.input_proj = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU()
#         )
        
#         # LSTM layers
#         self.lstm = nn.LSTM(
#             input_size=hidden_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             dropout=dropout if num_layers > 1 else 0.0,
#             bidirectional=bidirectional,
#             batch_first=True
#         )
        
#         lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
#         # Multi-head attention for pooling
#         self.attention = nn.MultiheadAttention(
#             embed_dim=lstm_output_dim,
#             num_heads=attention_heads,
#             dropout=dropout,
#             batch_first=True
#         )
        
#         # Learnable query for attention pooling
#         self.query = nn.Parameter(torch.randn(1, 1, lstm_output_dim))
        
#         # Output projection
#         self.output_proj = nn.Sequential(
#             nn.Linear(lstm_output_dim, latent_dim),
#             nn.LayerNorm(latent_dim),
#             nn.ReLU(),
#             nn.Linear(latent_dim, latent_dim)
#         )
        
#     def forward(
#         self,
#         history_obs: torch.Tensor,
#         history_mask: torch.Tensor = None,
#         return_attention_weights: bool = False
#     ):
#         """
#         Forward pass with attention pooling.
        
#         Args:
#             history_obs: (B, H, D) - batch of history sequences
#             history_mask: (B, H) - mask for padding (1=valid, 0=padding)
#             return_attention_weights: If True, also return attention weights
        
#         Returns:
#             latent: (B, latent_dim)
#             attention_weights: (B, 1, H) - optional, if return_attention_weights=True
#         """
#         B, H, D = history_obs.shape
        
#         # Project input
#         x = self.input_proj(history_obs)  # (B, H, hidden_dim)
        
#         # Pass through LSTM
#         if history_mask is not None:
#             lengths = history_mask.sum(dim=1).long().cpu()
#             lengths = torch.clamp(lengths, min=1)
#             x_packed = nn.utils.rnn.pack_padded_sequence(
#                 x, lengths, batch_first=True, enforce_sorted=False
#             )
#             lstm_out_packed, _ = self.lstm(x_packed)
#             lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
#                 lstm_out_packed, batch_first=True, total_length=H
#             )
#         else:
#             lstm_out, _ = self.lstm(x)  # (B, H, hidden_dim * num_directions)
        
#         # Expand query for batch
#         query = self.query.expand(B, -1, -1)  # (B, 1, hidden_dim * num_directions)
        
#         # Apply attention
#         # Create attention mask: True means "ignore this position"
#         attn_mask = None
#         if history_mask is not None:
#             # Convert to boolean mask (True = ignore, False = attend)
#             attn_mask = (history_mask == 0).unsqueeze(1)  # (B, 1, H)
        
#         attended_output, attn_weights = self.attention(
#             query, lstm_out, lstm_out,
#             key_padding_mask=(history_mask == 0) if history_mask is not None else None
#         )  # attended_output: (B, 1, hidden_dim * num_directions)
        
#         # Squeeze and project to latent
#         attended_output = attended_output.squeeze(1)  # (B, hidden_dim * num_directions)
#         latent = self.output_proj(attended_output)  # (B, latent_dim)
        
#         if return_attention_weights:
#             return latent, attn_weights
        
#         return latent
    
#     def reset(self):
#         """Reset any internal state"""
#         pass
