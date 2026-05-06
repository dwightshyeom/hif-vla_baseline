"""
LSTM with Observation Prediction Head + Raw Action Chunk Heads.

Unlike ObsFASTActionLSTM, this model directly regresses LP-normalised action
values instead of DCT coefficients.  No tokenizer is needed.

At each timestep t the shared LSTM processes:
    input_t = concat(obs_{t-1}, action_{t-1})   [default]
    input_t = action_{t-1}                       [action_only=True]

Up to three heads decode the hidden state h_t:

    obs_head          →  obs_t                               (obs_dim,)          [opt]
    future_chunk_head →  norm(a_t, a_{t+1}, …, a_{t+H-1})   (H × action_dim)   [opt]
    past_chunk_head   →  norm(a_{t-1}, a_{t-2}, …,           (pastH × action_dim) [opt]
                              a_{t-pastH})     [reversed, recent→older;
                                                includes a_{t-1} — the action
                                                just seen as LSTM input]

All chunk predictions are in LP-normalised action space; denormalise at
inference time.

Training objectives
-------------------
    obs_mse    = masked MSE(obs_pred,    obs_norm)             [all valid t]
    future_mse = masked MSE(future_pred, action_norm[t:t+H])  [t+H ≤ T]
    past_mse   = masked MSE(past_pred,   rev_past_chunk)       [t ≥ 1 for any valid slot;
                                                                  t ≥ pastH for all slots valid]
    l1_loss    = optional L1 sparsity on LSTM hidden states

    total = obs_loss_weight    * obs_mse    (only if use_obs_head=True)
          + future_loss_weight * future_mse  (only if use_future_head=True)
          + past_loss_weight   * past_mse    (only if use_past_head=True)
          + l1_weight          * l1_loss
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Vector Quantization layer (EMA codebook, straight-through gradient)
# ---------------------------------------------------------------------------

class VectorQuantize(nn.Module):
    """
    Vector Quantization with Exponential Moving Average (EMA) codebook updates.

    Given continuous input z ∈ R^D, finds the nearest codebook entry e_k and
    returns the quantised version via straight-through estimator:
        z_q = z + (e_k - z).detach()

    Two auxiliary losses:
      - commitment_loss = ||z - e_k.detach()||^2  (push encoder towards codes)
      - codebook_loss is handled implicitly via EMA updates (no gradient needed)

    Args:
        dim:        Embedding dimension (must match hidden_size).
        n_codes:    Number of codebook entries.
        ema_decay:  Decay rate for EMA codebook updates (default 0.99).
        eps:        Epsilon for Laplace smoothing in EMA count updates.
    """
    def __init__(self, dim: int, n_codes: int, ema_decay: float = 0.99,
                 eps: float = 1e-5):
        super().__init__()
        self.dim      = dim
        self.n_codes  = n_codes
        self.ema_decay = ema_decay
        self.eps      = eps

        # Codebook embeddings (initialised from N(0,1), will be overwritten by first batch)
        self.register_buffer('embeddings', torch.randn(n_codes, dim))
        # EMA cluster counts and embedding sums
        self.register_buffer('_ema_count', torch.zeros(n_codes))
        self.register_buffer('_ema_embed_sum', self.embeddings.clone())
        self._initialised = False

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (..., dim)  continuous input vectors.

        Returns:
            z_q:             (..., dim) quantised output (straight-through).
            commitment_loss: scalar, mean ||z - e_k.detach()||^2.
            encoding_indices: (...,) long tensor of codebook indices.
        """
        flat = z.reshape(-1, self.dim)   # (N, D)

        # ---- Nearest-neighbour lookup ------------------------------------
        # ||z - e||^2 = ||z||^2 - 2 z·e + ||e||^2
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ self.embeddings.t()
            + self.embeddings.pow(2).sum(dim=1, keepdim=True).t()
        )  # (N, n_codes)

        indices = dist.argmin(dim=1)                      # (N,)
        z_q_flat = self.embeddings[indices]                # (N, D)

        # ---- EMA codebook update (training only) -------------------------
        if self.training:
            if not self._initialised:
                # First forward: init codebook from data
                n_init = min(flat.shape[0], self.n_codes)
                perm = torch.randperm(flat.shape[0], device=flat.device)[:n_init]
                self.embeddings[:n_init] = flat[perm].detach()
                self._ema_embed_sum[:n_init] = flat[perm].detach()
                self._ema_count[:n_init] = 1.0
                self._initialised = True

            one_hot = F.one_hot(indices, self.n_codes).float()  # (N, n_codes)
            counts  = one_hot.sum(dim=0)                        # (n_codes,)
            embed_sum = one_hot.t() @ flat.detach()             # (n_codes, D)

            self._ema_count.mul_(self.ema_decay).add_(counts, alpha=1 - self.ema_decay)
            self._ema_embed_sum.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)

            # Laplace smoothing
            n = self._ema_count.sum()
            count_smooth = (
                (self._ema_count + self.eps) / (n + self.n_codes * self.eps) * n
            )
            self.embeddings.copy_(self._ema_embed_sum / count_smooth.unsqueeze(1))

        # ---- Losses & straight-through -----------------------------------
        commitment_loss = (flat - z_q_flat.detach()).pow(2).mean()

        # Straight-through: gradient flows as if z_q = z
        z_q_flat = flat + (z_q_flat - flat).detach()

        z_q = z_q_flat.view_as(z)
        encoding_indices = indices.view(z.shape[:-1])
        return z_q, commitment_loss, encoding_indices


# ---------------------------------------------------------------------------
# DCT helpers
# ---------------------------------------------------------------------------

def build_dct_basis(N: int, K: int) -> torch.Tensor:
    """
    Build the Type-II DCT synthesis (inverse) basis matrix.

    Returns B of shape (N, K) such that:
        trajectory_approx = B @ coefficients

    where coefficients has shape (K, D).
    """
    n = torch.arange(N, dtype=torch.float64).unsqueeze(1)   # (N, 1)
    k = torch.arange(K, dtype=torch.float64).unsqueeze(0)   # (1, K)
    B = torch.cos(torch.pi * k * (2 * n + 1) / (2 * N))    # (N, K)
    # Normalisation so that B^T B ≈ I  (orthonormal columns)
    B[:, 0] *= 1.0 / np.sqrt(N)
    B[:, 1:] *= np.sqrt(2.0 / N)
    return B.float()


def build_segment_pool_basis(N: int, K: int) -> torch.Tensor:
    """
    Build an orthonormal segment-pooling basis matrix of shape (N, K).

    Divides N time steps into K segments of roughly equal length (the last
    segment absorbs the remainder).  Let S_k be the length of segment k.

        B[n, k] = 1 / sqrt(S_k)   if n belongs to segment k
                  0                otherwise

    Columns are orthonormal (B^T B = I_K), so this basis is interchangeable
    with the DCT basis in dct_forward / dct_inverse.

    Reconstruction (B @ B^T @ x) is the piecewise-constant segment-mean
    approximation of x — each segment is replaced by its mean.
    """
    basis = torch.zeros(N, K, dtype=torch.float32)
    seg_size = N // K
    for k in range(K):
        start = k * seg_size
        end   = (k + 1) * seg_size if k < K - 1 else N
        seg_len = end - start
        basis[start:end, k] = 1.0 / (seg_len ** 0.5)
    return basis


def build_dft_basis(N: int, K_f: int) -> torch.Tensor:
    """
    Build an orthonormal real DFT (Fourier) synthesis basis of shape (N, 2*K_f - 1).

    K_f frequency bins are retained (k = 0, 1, …, K_f-1):
      - k = 0 (DC):     1 column: 1/√N × [1, 1, …, 1]
      - k = 1…K_f-1:    2 columns each:
            cosine col: √(2/N) × cos(2π k n / N)
            sine   col: √(2/N) × sin(2π k n / N)
    Total columns: 1 + 2×(K_f-1) = 2×K_f - 1.

    The basis is orthonormal (B^T B = I_{2K_f-1}), so ``dct_forward`` /
    ``dct_inverse`` work unchanged.  Unlike DCT, DFT captures both even and
    odd (sine) components, making it more expressive for asymmetric or
    non-zero-mean trajectories.

    Args:
        N:   signal length (number of timesteps).
        K_f: number of frequency bins to retain.
             Valid range: 1 ≤ K_f ≤ N//2 + 1.
    """
    assert 1 <= K_f <= N // 2 + 1, \
        f"K_f={K_f} out of valid range [1, {N // 2 + 1}] for N={N}"
    n = torch.arange(N, dtype=torch.float64)   # (N,)
    cols: list = []
    # DC component (k=0, cosine only — sine of 0 frequency is identically 0)
    cols.append(torch.ones(N, dtype=torch.float64) / np.sqrt(N))
    for k in range(1, K_f):
        angle = 2.0 * np.pi * k * n / N
        cols.append(torch.cos(angle) * np.sqrt(2.0 / N))  # cosine column
        cols.append(torch.sin(angle) * np.sqrt(2.0 / N))  # sine column
    return torch.stack(cols, dim=1).float()   # (N, 2*K_f - 1)


def build_haar_basis(N: int, K: int) -> torch.Tensor:
    """
    Build an orthonormal Haar wavelet synthesis basis of shape (N, K).

    Basis functions are ordered coarsest-to-finest (best approximation first):
      col 0:          scaling function — constant 1/√N over [0, N)
      col 1:          level-0 detail — +/− split over the full signal
      col 2, 3:       level-1 details — +/− splits over each half
      col 4…7:        level-2 details — +/− splits over each quarter
      …

    Works for **arbitrary** N (non-power-of-2) by computing integer segment
    boundaries with floor division.  Normalization is adjusted per segment so
    all columns are exactly unit-norm and mutually orthogonal.

    Unlike DCT, Haar wavelets have **compact temporal support** — each basis
    function is non-zero only within its own time segment.  This gives
    better localization for trajectories with abrupt direction changes.

    Args:
        N: signal length.
        K: number of basis functions to retain (1 ≤ K ≤ N).
    """
    assert 1 <= K <= N, f"K={K} must be in [1, N={N}]"
    basis = torch.zeros(N, K, dtype=torch.float32)

    # Column 0: scaling function (DC)
    basis[:, 0] = 1.0 / np.sqrt(N)

    col = 1
    level = 0
    while col < K:
        n_wavelets_at_level = 1 << level   # 2^level wavelets at this level
        for p in range(n_wavelets_at_level):
            if col >= K:
                break
            # Segment boundaries for wavelet at (level, position p)
            seg_start = (p * N) // n_wavelets_at_level
            seg_end   = ((p + 1) * N) // n_wavelets_at_level
            seg_mid   = (seg_start + seg_end) // 2

            n_pos = seg_mid - seg_start   # positive part length
            n_neg = seg_end - seg_mid     # negative part length

            if n_pos > 0 and n_neg > 0:
                # Normalize so ||col||_2 = 1 and col ⊥ DC (zero-mean)
                # v_pos = sqrt(n_neg / (n_pos * N_seg)),
                # v_neg = sqrt(n_pos / (n_neg * N_seg))
                N_seg = n_pos + n_neg
                v_pos = np.sqrt(n_neg / (n_pos * N_seg))
                v_neg = np.sqrt(n_pos / (n_neg * N_seg))
                basis[seg_start:seg_mid, col] =  float(v_pos)
                basis[seg_mid:seg_end,   col] = -float(v_neg)
            # If a half-segment is empty (N very small), the column stays zero —
            # the model will simply not use it (effectively fewer useful bases).
            col += 1
        level += 1
    return basis


def _basis_n_coeffs(abstraction: str, n_bases: int) -> int:
    """Return the number of basis coefficients for the given abstraction and n_bases.

    For all abstractions except 'dft', this equals n_bases.
    For 'dft', K_f frequency bins produce 2*K_f - 1 real coefficients.
    """
    if abstraction == 'dft':
        return 2 * n_bases - 1
    return n_bases


def dct_forward(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project time-domain signal x (..., N, D) onto orthonormal basis → (..., K, D)."""
    return torch.einsum('...nd, nk -> ...kd', x, basis)     # B^T @ x


def dct_inverse(coeffs: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Reconstruct time-domain signal from orthonormal-basis coefficients (..., K, D) → (..., N, D)."""
    return torch.einsum('...kd, nk -> ...nd', coeffs, basis)  # B @ coeffs


# ---------------------------------------------------------------------------
# Spatial keypoint helpers
# ---------------------------------------------------------------------------

def greedy_spatial_grouping(
    trajectory: np.ndarray,  # (N, D) — ordered past actions
    K: int,                  # target number of groups
    threshold: float = 0.1,  # Euclidean distance threshold for splitting a time segment
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Group consecutive past actions into exactly K waypoints.

    **Time has higher priority than spatial location.**  The algorithm starts
    from K roughly equal-duration temporal segments (time-first), then
    refines boundaries based on spatial proximity (spatial-second).

    Phase 1 — uniform temporal segmentation:
      Divide N timesteps into K equal-length segments.

    Phase 2 — spatial refinement:
      Within each segment, check if the actions span more than *threshold*
      (max intra-segment distance to centroid). If so, split the worst
      segment at the point of largest spatial jump and merge the two
      temporally nearest remaining segments to keep the count at K.
      Repeat for a bounded number of iterations.

    This ensures temporally contiguous groups that also respect spatial
    boundaries — actions that are near each other in time AND space are
    grouped together, but temporal contiguity is never violated.

    Args:
        trajectory: (N, D) past actions in normalised space, ordered recent→older.
        K: desired number of output groups.
        threshold: distance threshold; segments with intra-spread > threshold
                   are candidates for splitting.

    Returns:
        centroids: (K, D) group centroids.
        durations: (K,)   normalised durations (sum to 1).
    """
    N, D = trajectory.shape

    if N == 0:
        return np.zeros((K, D), dtype=np.float32), np.ones(K, dtype=np.float32) / K

    if N <= K:
        centroids = np.zeros((K, D), dtype=np.float32)
        durations = np.zeros(K, dtype=np.float32)
        centroids[:N] = trajectory
        durations[:N] = 1.0 / N
        return centroids, durations

    # Phase 1: uniform temporal segmentation → K groups
    seg_size = N / K
    groups = []  # list of (start_idx, end_idx)
    for k in range(K):
        s = int(round(k * seg_size))
        e = int(round((k + 1) * seg_size))
        e = max(e, s + 1)  # ensure at least 1 element
        groups.append((s, min(e, N)))

    # Phase 2: spatial refinement — split high-spread segments and merge
    # the two temporally nearest remaining segments to keep count == K.
    max_iters = K  # bounded refinement
    for _ in range(max_iters):
        # Find the segment with the worst intra-spread
        worst_spread = 0.0
        worst_idx = -1
        worst_split_at = -1  # index within segment to split
        for i, (s, e) in enumerate(groups):
            if e - s < 2:
                continue
            seg_data = trajectory[s:e]
            centroid = seg_data.mean(axis=0)
            dists = np.linalg.norm(seg_data - centroid, axis=1)
            spread = float(dists.max())
            if spread > threshold and spread > worst_spread:
                # Find the biggest spatial jump within this segment
                jumps = np.linalg.norm(np.diff(seg_data, axis=0), axis=1)
                split_local = int(jumps.argmax()) + 1  # split after this index
                if split_local > 0 and split_local < (e - s):
                    worst_spread = spread
                    worst_idx = i
                    worst_split_at = s + split_local

        if worst_idx < 0:
            break  # all segments are within threshold

        # Split the worst segment
        s, e = groups[worst_idx]
        groups[worst_idx] = (s, worst_split_at)
        groups.insert(worst_idx + 1, (worst_split_at, e))

        # Merge the two adjacent groups with smallest combined duration
        # to restore count back to K
        min_cost = float('inf')
        min_idx = 0
        for i in range(len(groups) - 1):
            combined = (groups[i][1] - groups[i][0]) + (groups[i+1][1] - groups[i+1][0])
            if combined < min_cost:
                min_cost = combined
                min_idx = i
        groups[min_idx] = (groups[min_idx][0], groups[min_idx + 1][1])
        groups.pop(min_idx + 1)

    # Compute centroids and normalised durations
    centroids = np.zeros((K, D), dtype=np.float32)
    durations = np.zeros(K, dtype=np.float32)
    for k, (s, e) in enumerate(groups):
        centroids[k] = trajectory[s:e].mean(axis=0)
        durations[k] = (e - s) / N

    return centroids, durations


def reconstruct_from_keypoints(
    centroids: torch.Tensor,   # (..., K, D)
    durations: torch.Tensor,   # (..., K)
    H: int,
    _chunk: int = 32,
) -> torch.Tensor:
    """
    Differentiable soft reconstruction from spatial keypoints.

    Each centroid is placed at the midpoint of its temporal segment (determined
    by the cumulative duration).  Every output timestep is a Gaussian-weighted
    mixture of the K centroids, where the kernel width of each centroid equals
    half its duration.  This is fully differentiable w.r.t. both centroids and
    durations.

    The H dimension is processed in chunks to avoid OOM on large (B, T, K, H)
    intermediate tensors.

    Returns: (..., H, D)
    """
    device = centroids.device

    # Segment boundaries
    cum_dur = torch.cumsum(durations, dim=-1)
    seg_start = torch.cat([
        torch.zeros_like(cum_dur[..., :1]),
        cum_dur[..., :-1]
    ], dim=-1)  # (..., K)
    positions = (seg_start + cum_dur) / 2  # centroid midpoint in [0, 1]

    pos = positions.unsqueeze(-1)                                   # (..., K, 1)
    sigma = (durations.unsqueeze(-1) / 2).clamp(min=0.01)           # (..., K, 1)
    n_leading = centroids.dim() - 2

    t_idx = torch.linspace(0, 1, H, device=device)  # (H,)

    recon_chunks: List[torch.Tensor] = []
    for start in range(0, H, _chunk):
        t_chunk = t_idx[start : start + _chunk]                     # (chunk,)
        t_shape = [1] * n_leading + [1, t_chunk.shape[0]]
        t_exp = t_chunk.view(*t_shape)                              # (1..., 1, chunk)

        log_w = -0.5 * ((t_exp - pos) / sigma).pow(2)              # (..., K, chunk)
        weights = F.softmax(log_w, dim=-2)                          # softmax over K
        recon_chunks.append(
            torch.matmul(weights.transpose(-2, -1), centroids)      # (..., chunk, D)
        )

    return torch.cat(recon_chunks, dim=-2)                          # (..., H, D)


class ObsActionChunkLSTM(nn.Module):
    """
    LSTM encoder with:
      1. Observation prediction head  → obs_t                    (obs_dim,)        [opt]
      2. Future action chunk head     → norm(a_{t:t+H})          (H × action_dim)  [opt]
      3. Past action chunk head (opt) → norm(rev past H actions) (pastH × action_dim) [opt]

    Args:
        obs_dim:        Observation dimension (default 20 for PushT).
        action_dim:     Action dimension (default 2 for PushT).
        hidden_size:    LSTM hidden units.
        num_layers:     Stacked LSTM layers.
        dropout:        Dropout between LSTM layers and in MLP heads.
        chunk_H:        Future action chunk length H.
        use_obs_head:   If True (default), add the observation prediction head.
        use_future_head: If True (default), add the future action chunk head.
        num_future_modes: Number of future prediction modes (M). When M > 1,
                        the future head predicts M distinct action chunks and
                        a mode probability head outputs M logits.  Training
                        uses Winner-Takes-All loss (only best mode gets grad).
        use_past_head:  If True, add a past chunk prediction head.
        past_chunk_H:   Past chunk length (defaults to chunk_H when None).
        past_abstraction: 'raw', 'dct', 'segment_pool', or 'spatial_keypoint'.
                        'spatial_keypoint' groups consecutive actions by spatial
                        proximity into K waypoints (centroids + durations).
        past_n_bases:   Number of basis functions / segments / keypoints.
        action_only:    If True, LSTM input is action_{t-1} only (no obs).
                        Useful for learning pure action-trajectory memory.
        use_vq:         If True, apply Vector Quantization to LSTM hidden states
                        before feeding them to heads.
        vq_n_codes:     Number of VQ codebook entries.
        vq_commitment_weight: Weight for VQ commitment loss.
    """

    def __init__(
        self,
        obs_dim:         int   = 20,
        action_dim:      int   = 2,
        hidden_size:     int   = 256,
        num_layers:      int   = 2,
        dropout:         float = 0.1,
        chunk_H:         int   = 16,
        use_obs_head:    bool  = True,
        use_future_head: bool  = True,
        num_future_modes: int  = 1,
        future_abstraction: str = 'raw',
        future_n_bases:  int   = 32,
        use_past_head:   bool  = False,
        past_chunk_H:    Optional[int] = None,
        past_abstraction: str  = 'raw',
        past_n_bases:    int   = 32,
        action_only:     bool  = False,
        use_vq:          bool  = False,
        vq_n_codes:      int   = 512,
        vq_commitment_weight: float = 0.25,
    ):
        super().__init__()

        self.obs_dim         = obs_dim
        self.action_dim      = action_dim
        self.hidden_size     = hidden_size
        self.num_layers      = num_layers
        self.action_only     = action_only
        self.input_dim       = action_dim if action_only else obs_dim + action_dim
        self.chunk_H         = chunk_H
        self.use_obs_head    = use_obs_head
        self.use_future_head   = use_future_head
        self.future_abstraction = future_abstraction
        self.future_n_bases    = future_n_bases
        self.use_past_head     = use_past_head
        self.past_chunk_H    = past_chunk_H if past_chunk_H is not None else chunk_H
        self.past_abstraction = past_abstraction
        self.past_n_bases    = past_n_bases
        self.num_future_modes = num_future_modes
        self.use_vq          = use_vq
        self.vq_n_codes      = vq_n_codes
        self.vq_commitment_weight = vq_commitment_weight

        # Persistent hidden state for stateful (step-by-step) inference
        self._hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # ---- Validate past_abstraction mode is recognized ---------------
        _VALID_ABSTRACTIONS = ('raw', 'dct', 'segment_pool', 'dft', 'dwt',
                               'spatial_keypoint')
        if past_abstraction not in _VALID_ABSTRACTIONS:
            raise ValueError(
                f"Unknown past_abstraction={past_abstraction!r}. "
                f"Must be one of {_VALID_ABSTRACTIONS}.")

        # ---- Shared LSTM encoder ----------------------------------------
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ---- Observation prediction head: hidden_size → obs_dim ---------
        if use_obs_head:
            obs_h = max(hidden_size, obs_dim * 2)
            self.obs_head = nn.Sequential(
                nn.Linear(hidden_size, obs_h),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(obs_h, obs_h // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(obs_h // 2, obs_dim),
            )
        else:
            self.obs_head = None

        # ---- Future action chunk head --------------------------------
        if use_future_head:
            M = num_future_modes
            if future_abstraction == 'dct':
                # Predict DCT coefficients: M * K * action_dim
                fut_out = M * future_n_bases * action_dim
                B_fut = build_dct_basis(chunk_H, future_n_bases)
                self.register_buffer('_future_dct_basis', B_fut)
            else:
                # Raw: M * H * action_dim
                fut_out = M * chunk_H * action_dim
            fut_h   = max(hidden_size * 2, fut_out * 2)
            fut_mid = max(fut_h // 2, fut_out)  # never compress below output dim
            self.future_chunk_head = nn.Sequential(
                nn.Linear(hidden_size, fut_h),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fut_h, fut_mid),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fut_mid, fut_out),
                # No final activation – predictions are in LP-normalised space
            )
            # Mode probability head (only when M > 1)
            if M > 1:
                self.future_mode_head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, M),
                )
            else:
                self.future_mode_head = None
        else:
            self.future_chunk_head = None
            self.future_mode_head = None

        # ---- Past action chunk head (optional) --------------------------
        if use_past_head:
            if past_abstraction in ('dct', 'segment_pool', 'dft', 'dwt'):
                # Predict basis coefficients per action dim.
                # DFT produces 2*K_f-1 coefficients; all others produce K.
                n_coeffs = _basis_n_coeffs(past_abstraction, past_n_bases)
                past_out = n_coeffs * action_dim
                if past_abstraction == 'dct':
                    B = build_dct_basis(self.past_chunk_H, past_n_bases)
                elif past_abstraction == 'segment_pool':
                    B = build_segment_pool_basis(self.past_chunk_H, past_n_bases)
                elif past_abstraction == 'dft':
                    B = build_dft_basis(self.past_chunk_H, past_n_bases)
                else:  # 'dwt'
                    B = build_haar_basis(self.past_chunk_H, past_n_bases)
                # Buffer name kept as _past_dct_basis for checkpoint compatibility
                self.register_buffer('_past_dct_basis', B)
            elif past_abstraction == 'spatial_keypoint':
                # Predict K centroids (K×D) + K duration logits (K×1)
                past_out = past_n_bases * (action_dim + 1)
            else:
                past_out = self.past_chunk_H * action_dim
            past_h = max(hidden_size * 4, past_out * 2)
            past_mid = max(past_h // 2, past_out)  # never compress below output dim
            self.past_chunk_head = nn.Sequential(
                nn.Linear(hidden_size, past_h),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(past_h, past_mid),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(past_mid, past_out),
            )
        else:
            self.past_chunk_head = None

        # ---- Optional VQ bottleneck on LSTM hidden states ---------------
        if use_vq:
            self.vq_layer = VectorQuantize(dim=hidden_size, n_codes=vq_n_codes)
        else:
            self.vq_layer = None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset persistent hidden state (call at episode start)."""
        self._hidden_state = None

    # ------------------------------------------------------------------
    def forward(
        self,
        obs:          torch.Tensor,
        actions:      torch.Tensor,
        lengths:      Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple:
        """
        Forward pass.

        The input at each step t is [obs_{t-1}, action_{t-1}] — the shift is
        applied inside this method.

        Args:
            obs:     (B, T, obs_dim)    LP-normalised observations.
            actions: (B, T, action_dim) LP-normalised actions (LSTM input).
            lengths: (B,) episode lengths for packed-sequence LSTM.
            hidden_state: initial (h_n, c_n) or None.

        Returns:
            obs_pred:          (B, T, obs_dim)  or None      predicted next obs.
            future_chunk_pred: (B, T, chunk_H,   action_dim)  normalised future chunk.
            past_chunk_pred:   (B, T, past_chunk_H, action_dim) or None.
            new_hidden:        (h_n, c_n)
            lstm_output:       (B, T, hidden_size)
        """
        B, T, _ = obs.shape

        zero_action = torch.zeros(B, 1, self.action_dim, device=actions.device, dtype=actions.dtype)
        act_sh  = torch.cat([zero_action, actions[:, :-1, :]], dim=1)

        if self.action_only:
            x = act_sh                                              # (B, T, action_dim)
        else:
            zero_obs = torch.zeros(B, 1, self.obs_dim, device=obs.device, dtype=obs.dtype)
            obs_sh   = torch.cat([zero_obs, obs[:, :-1, :]], dim=1)
            x        = torch.cat([obs_sh, act_sh], dim=-1)          # (B, T, input_dim)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, (h_n, c_n) = self.lstm(packed, hidden_state)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=T)
        else:
            lstm_output, (h_n, c_n) = self.lstm(x, hidden_state)

        # ---- Optional VQ bottleneck on hidden states --------------------
        vq_loss = torch.tensor(0.0, device=obs.device)
        if self.use_vq and self.vq_layer is not None:
            lstm_output, commitment_loss, _ = self.vq_layer(lstm_output)
            vq_loss = self.vq_commitment_weight * commitment_loss

        obs_pred: Optional[torch.Tensor] = None
        if self.use_obs_head and self.obs_head is not None:
            obs_pred = self.obs_head(lstm_output)                   # (B, T, obs_dim)

        future_chunk_pred: Optional[torch.Tensor] = None
        future_mode_logits: Optional[torch.Tensor] = None
        future_chunk_coeffs: Optional[torch.Tensor] = None   # DCT coefficients (B,T,[M,]K,D)
        if self.use_future_head and self.future_chunk_head is not None:
            M = self.num_future_modes
            fut_flat = self.future_chunk_head(lstm_output)
            if self.future_abstraction == 'dct':
                K_f = self.future_n_bases
                if M > 1:
                    coeffs = fut_flat.view(B, T, M, K_f, self.action_dim)
                    future_chunk_coeffs = coeffs
                    future_chunk_pred = dct_inverse(coeffs, self._future_dct_basis)  # (B,T,M,H,D)
                    future_mode_logits = self.future_mode_head(lstm_output)
                else:
                    coeffs = fut_flat.view(B, T, K_f, self.action_dim)
                    future_chunk_coeffs = coeffs
                    future_chunk_pred = dct_inverse(coeffs, self._future_dct_basis)  # (B,T,H,D)
            else:
                if M > 1:
                    future_chunk_pred = fut_flat.view(B, T, M, self.chunk_H, self.action_dim)
                    future_mode_logits = self.future_mode_head(lstm_output)
                else:
                    future_chunk_pred = fut_flat.view(B, T, self.chunk_H, self.action_dim)

        past_chunk_pred: Optional[torch.Tensor] = None
        past_chunk_coeffs = None  # Tensor for DCT/segment_pool/dft/dwt, tuple for spatial_keypoint
        if self.use_past_head and self.past_chunk_head is not None:
            past_flat = self.past_chunk_head(lstm_output)
            if self.past_abstraction in ('dct', 'segment_pool', 'dft', 'dwt'):
                # past_flat: (B, T, n_coeffs*D) — basis coefficients
                n_coeffs = _basis_n_coeffs(self.past_abstraction, self.past_n_bases)
                coeffs = past_flat.view(B, T, n_coeffs, self.action_dim)
                past_chunk_coeffs = coeffs
                # Reconstruct full trajectory via basis inverse: (B,T,K,D) → (B,T,pastH,D)
                past_chunk_pred = dct_inverse(coeffs, self._past_dct_basis)
            elif self.past_abstraction == 'spatial_keypoint':
                # past_flat: (B, T, K*(D+1))
                raw = past_flat.view(B, T, self.past_n_bases, self.action_dim + 1)
                centroids = raw[..., :self.action_dim]          # (B, T, K, D)
                durations = F.softmax(raw[..., -1], dim=-1)     # (B, T, K)
                past_chunk_coeffs = (centroids, durations)
                # Skip expensive (B,T,K,H) reconstruction here — the loss
                # operates directly on centroids+durations.  Reconstruction
                # is performed lazily in predict() when actually needed.
                past_chunk_pred = None
            else:
                past_chunk_pred = past_flat.view(B, T, self.past_chunk_H, self.action_dim)

        return obs_pred, future_chunk_pred, past_chunk_pred, (h_n, c_n), lstm_output, vq_loss, past_chunk_coeffs, future_mode_logits, future_chunk_coeffs

    # ------------------------------------------------------------------
    def predict(
        self,
        obs:          torch.Tensor,
        actions:      torch.Tensor,
        lengths:      Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """No-gradient forward pass."""
        with torch.no_grad():
            (obs_pred, future_chunk_pred, past_chunk_pred,
             new_hidden, _, _, past_coeffs, future_mode_logits, _) = self.forward(
                obs, actions, lengths=lengths, hidden_state=hidden_state)
            # Lazy reconstruction for spatial_keypoint (skipped in forward to save memory)
            if past_chunk_pred is None and isinstance(past_coeffs, tuple):
                centroids, durations = past_coeffs
                past_chunk_pred = reconstruct_from_keypoints(
                    centroids, durations, self.past_chunk_H)
        return obs_pred, future_chunk_pred, past_chunk_pred, new_hidden, future_mode_logits

    def extract_hidden_states(self, obs, actions, lengths=None, hidden_state=None):
        """Extract per-step LSTM hidden states without applying heads."""
        _, _, _, new_hidden, lstm_output, _, _, _, _ = self.forward(
            obs, actions, lengths=lengths, hidden_state=hidden_state)
        return lstm_output, new_hidden


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def compute_obs_action_chunk_loss(
    obs_pred:           Optional[torch.Tensor],
    obs_target:         Optional[torch.Tensor],
    obs_mask:           torch.Tensor,
    future_pred:        Optional[torch.Tensor] = None,
    future_target:      Optional[torch.Tensor] = None,
    future_mask:        Optional[torch.Tensor] = None,
    past_pred:          Optional[torch.Tensor] = None,
    past_target:        Optional[torch.Tensor] = None,
    past_mask:          Optional[torch.Tensor] = None,
    lstm_output:        Optional[torch.Tensor] = None,
    l1_on_hidden_state: bool  = False,
    l1_weight:          float = 1e-3,
    obs_loss_weight:    float = 1.0,
    future_loss_weight: float = 1.0,
    past_loss_weight:   float = 1.0,
    # ── Option 1: temporal reweighting for past chunk ────────────────
    past_temporal_weight: str   = 'uniform',   # 'uniform' | 'linear' | 'exponential'
    past_temporal_alpha:  float = 2.0,         # strength of reweighting
    # ── Option 2 (DCT mode): frequency-weighted coefficient loss ─────
    past_dct_basis:       Optional[torch.Tensor] = None,  # (H_past, K) basis matrix
    past_dct_freq_decay:  float = 0.0,   # >0: downweight high-freq coeffs (exp decay)
    past_pred_coeffs:     Optional[torch.Tensor] = None,  # model's raw DCT/basis coefficients
    # ── Temporal consistency regularizer (works in raw and DCT mode) ──
    past_consistency_weight: float = 0.0,  # penalise disagreement between N-step-offset predictions
    past_consistency_n_steps: Optional[List[int]] = None,  # list of N values; default [1]
    # ── VQ commitment loss (passed through from model forward) ──────
    vq_loss: Optional[torch.Tensor] = None,
    # ── Multi-modal future: WTA loss + mode probability ──────────────
    future_mode_logits: Optional[torch.Tensor] = None,  # (B, T, M) raw logits
    future_mode_cls_weight: float = 0.0,   # cross-entropy loss weight for mode selection
    future_entropy_weight:  float = 0.0,   # entropy regularization (prevent mode collapse)
    # ── Future DCT: frequency-weighted coefficient loss ──────────────
    future_dct_basis:       Optional[torch.Tensor] = None,  # (H, K_f) basis matrix
    future_dct_freq_decay:  float = 0.0,   # >0: downweight high-freq coeffs
    future_pred_coeffs:     Optional[torch.Tensor] = None,  # (B, T, [M,] K_f, D) model coefficients
) -> Dict[str, torch.Tensor]:
    """
    Combined loss for obs head, future chunk head, and optional past chunk head.

    Args:
        obs_pred:      (B, T, obs_dim) or None  predicted observations.
        obs_target:    (B, T, obs_dim) or None  ground-truth observations.
        obs_mask:      (B, T)                   validity mask.
        future_pred:   (B, T, [M,] H, action_dim) predicted future chunk (normalised).
                       When M > 1, the Winner-Takes-All (WTA) loss is used: only
                       the mode closest to GT contributes to the gradient.
        future_target: (B, T, H, action_dim)    target future chunk (normalised).
        future_mask:   (B, T)                   1 where t + H ≤ T_b.
        future_mode_logits: (B, T, M) or None   mode probability logits.
        past_pred:     (B, T, pastH, action_dim) predicted past chunk or None.
        past_target:   (B, T, pastH, action_dim) target past chunk or None.
        past_mask:     (B, T, pastH) or (B, T)  per-position or per-timestep validity.
        lstm_output:   (B, T, hidden_size)      for optional L1 regularisation.

    Returns:
        dict with scalar tensors: 'loss', 'obs_mse_loss', 'future_mse_loss',
        'past_mse_loss', 'l1_loss', 'obs_mse_per_dim', 'future_mse_per_step',
        'future_mode_cls_loss', 'future_entropy_loss'.
    """
    device = obs_mask.device

    # ---- Obs MSE --------------------------------------------------------
    obs_mse     = torch.tensor(0.0, device=device)
    obs_per_dim = None
    if obs_pred is not None and obs_target is not None:
        m_obs   = obs_mask.unsqueeze(-1).expand_as(obs_pred)
        sq_obs  = (obs_pred - obs_target).pow(2) * m_obs
        obs_mse = sq_obs.sum() / (m_obs.sum() + 1e-8)
        obs_per_dim = sq_obs.sum(dim=(0, 1)) / (obs_mask.sum() + 1e-8)

    # ---- Future chunk MSE (WTA when multi-modal) -------------------------
    future_mse      = torch.tensor(0.0, device=device)
    future_per_step = None
    future_mode_cls = torch.tensor(0.0, device=device)
    future_entropy  = torch.tensor(0.0, device=device)
    if future_pred is not None and future_target is not None and future_mask is not None:
        # Optional: build frequency weights for DCT coefficient-space loss
        _use_coeff_loss = (future_dct_basis is not None
                           and future_pred_coeffs is not None
                           and future_dct_freq_decay > 0.0)
        _freq_w = None
        if _use_coeff_loss:
            basis = future_dct_basis.to(device)  # (H, K_f)
            K_f = future_pred_coeffs.shape[-2]
            k_idx = torch.arange(K_f, device=device, dtype=future_pred.dtype)
            _freq_w = torch.exp(-future_dct_freq_decay * k_idx / max(K_f - 1, 1))
            _freq_w = _freq_w * K_f / _freq_w.sum()  # normalise so mean weight = 1

        if future_pred.dim() == 5:
            # Multi-modal: future_pred is (B, T, M, H, D)
            M = future_pred.shape[2]

            if _use_coeff_loss:
                # Compute loss in coefficient space with frequency weighting
                # Project target: (B,T,H,D) → (B,T,K_f,D) → (B,T,1,K_f,D)
                coeffs_target = dct_forward(future_target, basis).unsqueeze(2)
                # future_pred_coeffs: (B,T,M,K_f,D)
                diff_sq = (future_pred_coeffs - coeffs_target) ** 2  # (B,T,M,K_f,D)
                diff_sq = diff_sq * _freq_w.view(1, 1, 1, K_f, 1)
                per_mode_sq = diff_sq.mean(dim=(-2, -1))  # (B,T,M)
            else:
                # Time-domain MSE
                target_exp = future_target.unsqueeze(2)
                per_mode_sq = ((future_pred - target_exp) ** 2).mean(dim=(-2, -1))

            # Mask invalid timesteps
            fm = future_mask.unsqueeze(-1)  # (B, T, 1)
            per_mode_sq_masked = per_mode_sq * fm
            # Winner-Takes-All: select mode with lowest MSE at each (b, t)
            best_mode = per_mode_sq_masked.detach().argmin(dim=-1)  # (B, T)
            # Gather winner MSE
            winner_mse = per_mode_sq.gather(2, best_mode.unsqueeze(-1)).squeeze(-1)  # (B, T)
            future_mse = (winner_mse * future_mask).sum() / (future_mask.sum() + 1e-8)

            # Per-step error of the best mode (for logging — always in time-domain)
            bm_idx = best_mode.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            bm_idx = bm_idx.expand(-1, -1, -1, future_pred.shape[3], future_pred.shape[4])
            best_pred = future_pred.gather(2, bm_idx).squeeze(2)  # (B,T,H,D)
            m_fut = future_mask.unsqueeze(-1).unsqueeze(-1).expand_as(best_pred)
            sq_fut = (best_pred - future_target) ** 2 * m_fut
            future_per_step = sq_fut.sum(dim=(0, 1)) / (future_mask.sum() + 1e-8).unsqueeze(-1).unsqueeze(-1)

            # Mode classification loss: teach mode_head to predict the winner
            if future_mode_logits is not None and future_mode_cls_weight > 0.0:
                # future_mode_logits: (B, T, M), best_mode: (B, T) long
                logits_flat = future_mode_logits.reshape(-1, M)
                target_flat = best_mode.reshape(-1)
                mask_flat = future_mask.reshape(-1)
                # Masked cross-entropy
                ce = F.cross_entropy(logits_flat, target_flat, reduction='none')
                future_mode_cls = (ce * mask_flat).sum() / (mask_flat.sum() + 1e-8)

            # Entropy regularization: encourage diverse mode usage
            if future_mode_logits is not None and future_entropy_weight > 0.0:
                probs = F.softmax(future_mode_logits, dim=-1)  # (B, T, M)
                log_probs = F.log_softmax(future_mode_logits, dim=-1)
                ent = -(probs * log_probs).sum(dim=-1)  # (B, T)
                # Maximize entropy → minimize negative entropy
                future_entropy = -(ent * future_mask).sum() / (future_mask.sum() + 1e-8)
        else:
            # Single-mode: (B, T, H, D)
            if _use_coeff_loss:
                # Coefficient-space loss with frequency weighting
                coeffs_target = dct_forward(future_target, basis)  # (B,T,K_f,D)
                diff_sq = (future_pred_coeffs - coeffs_target) ** 2  # (B,T,K_f,D)
                diff_sq = diff_sq * _freq_w.view(1, 1, K_f, 1)
                m_coeff = future_mask.unsqueeze(-1).unsqueeze(-1)  # (B,T,1,1)
                future_mse = (diff_sq * m_coeff).sum() / (m_coeff.sum() * K_f * future_pred.shape[-1] + 1e-8)
            else:
                m_fut      = future_mask.unsqueeze(-1).unsqueeze(-1).expand_as(future_pred)
                sq_fut     = (future_pred - future_target).pow(2) * m_fut
                future_mse = sq_fut.sum() / (m_fut.sum() + 1e-8)
            # Per-step error (always time-domain for logging)
            m_fut_log = future_mask.unsqueeze(-1).unsqueeze(-1).expand_as(future_pred)
            sq_fut_log = (future_pred - future_target).pow(2) * m_fut_log
            future_per_step = sq_fut_log.sum(dim=(0, 1)) / (future_mask.sum() + 1e-8).unsqueeze(-1).unsqueeze(-1)

    # ---- Past chunk MSE (optional) --------------------------------------
    past_mse = torch.tensor(0.0, device=device)
    if past_target is not None and past_mask is not None and (
            past_pred is not None or isinstance(past_pred_coeffs, tuple)):

        if isinstance(past_pred_coeffs, tuple):
            # ── spatial_keypoint mode ──────────────────────────────────
            # past_pred_coeffs = (centroids (B,T,K,D), durations (B,T,K))
            # past_target      = (B, T, K, D+1) — centroids + duration
            # past_mask         = (B, T, K)      — per-group validity
            pred_centroids, pred_durations = past_pred_coeffs
            D_act = pred_centroids.shape[-1]
            target_centroids = past_target[..., :D_act]     # (B, T, K, D)
            target_durations = past_target[..., D_act]      # (B, T, K)
            m_kp  = past_mask                               # (B, T, K)
            m_kp4 = m_kp.unsqueeze(-1)                      # (B, T, K, 1)

            # Duration-weighted centroid MSE (longer groups contribute more)
            dur_w = target_durations.unsqueeze(-1)          # (B, T, K, 1)
            sq_cent = (pred_centroids - target_centroids).pow(2) * m_kp4 * dur_w
            centroid_loss = sq_cent.sum() / (m_kp4.sum() * D_act + 1e-8)

            # Duration MSE
            sq_dur = (pred_durations - target_durations).pow(2) * m_kp
            duration_loss = sq_dur.sum() / (m_kp.sum() + 1e-8)

            past_mse = centroid_loss + duration_loss

        else:
            # Non-spatial_keypoint modes (raw / DCT / segment_pool)
            # past_mask can be (B, T, pastH) per-position or (B, T) per-timestep
            if past_mask.dim() == 3:
                m_past = past_mask.unsqueeze(-1).expand_as(past_pred)   # (B,T,pH,D)
            else:
                m_past = past_mask.unsqueeze(-1).unsqueeze(-1).expand_as(past_pred)

            if (past_pred_coeffs is not None
                    and past_dct_basis is not None
                    and past_dct_freq_decay > 0.0):
                # ── Frequency-weighted coefficient-space loss (DCT/segment_pool) ──
                basis = past_dct_basis.to(device)                        # (H_past, K)
                coeffs_target = dct_forward(past_target * m_past, basis) # (B, T, K, D)
                K = past_pred_coeffs.shape[2]
                k_idx  = torch.arange(K, device=device, dtype=past_pred.dtype)
                freq_w = torch.exp(-past_dct_freq_decay * k_idx / max(K - 1, 1))
                freq_w = freq_w * K / freq_w.sum()   # normalise so mean weight = 1
                if past_mask.dim() == 3:
                    m_coeff_t = past_mask.any(dim=2).float().unsqueeze(-1).unsqueeze(-1)
                else:
                    m_coeff_t = past_mask.unsqueeze(-1).unsqueeze(-1).float()
                sq_coeff = (past_pred_coeffs - coeffs_target).pow(2) * freq_w.view(1, 1, K, 1)
                past_mse = (sq_coeff * m_coeff_t).sum() / (m_coeff_t.sum() * K * past_pred.shape[-1] + 1e-8)
            else:
                # ── Time-domain masked loss ──
                sq_past = (past_pred - past_target).pow(2) * m_past
                if past_temporal_weight != 'uniform':
                    pastH = past_pred.shape[2]
                    k = torch.arange(pastH, device=device, dtype=past_pred.dtype)
                    if past_temporal_weight == 'linear':
                        w = 1.0 + past_temporal_alpha * k / max(pastH - 1, 1)
                    elif past_temporal_weight == 'exponential':
                        w = torch.exp(past_temporal_alpha * k / max(pastH - 1, 1))
                    else:
                        raise ValueError(f"Unknown past_temporal_weight: {past_temporal_weight!r}")
                    sq_past = sq_past * w.view(1, 1, pastH, 1)
                past_mse = sq_past.sum() / (m_past.sum() + 1e-8)

    # ---- L1 on hidden states --------------------------------------------
    l1_loss = torch.tensor(0.0, device=device)
    if l1_on_hidden_state and lstm_output is not None:
        m_h     = obs_mask.unsqueeze(-1)
        l1_loss = (lstm_output.abs() * m_h).sum() / (
            m_h.sum() * lstm_output.shape[-1] + 1e-8)

    # ---- Multi-step temporal consistency loss ---------------------------
    # past_pred[b, t, k, :] ≈ normalized action a_{t-1-k}
    # past_pred[b, t+N, k+N, :] ≈ normalized action a_{(t+N)-1-(k+N)} = a_{t-1-k}
    # → predictions N steps apart that refer to the same underlying action must agree:
    #       past_pred[:, :-N, :H-N, :] ≈ past_pred[:, N:, N:H, :]
    # (The slice formula is independent of whether slot 0 represents a_{t-1}
    # or a_{t-2}; only the index semantics in the comments differ.)
    # Checking multiple N values gives a stronger gradient signal than N=1 alone.
    # NOTE: This only applies to raw / DCT / segment_pool / dft / dwt modes where
    #       past_pred is a full time-domain reconstruction. For spatial_keypoint the
    #       overlap semantics don't hold, so we skip consistency loss there.
    consistency_loss = torch.tensor(0.0, device=device)
    if (past_consistency_weight > 0.0
            and past_pred is not None
            and past_pred.shape[1] > 1
            and not isinstance(past_pred_coeffs, tuple)):  # skip for spatial_keypoint
        pastH   = past_pred.shape[2]
        D_a     = past_pred.shape[3]
        T_seq   = past_pred.shape[1]
        # Default: check N=1 only (backward-compatible behaviour)
        n_steps_list = past_consistency_n_steps if past_consistency_n_steps else [1]
        n_terms = 0
        for N in n_steps_list:
            if N <= 0 or N >= pastH or N >= T_seq:
                continue
            pred_t  = past_pred[:, :-N, :pastH - N, :]   # (B, T-N, H-N, D)
            pred_tN = past_pred[:, N:,  N:pastH,    :]   # (B, T-N, H-N, D)
            # Validity mask for overlap region
            if past_mask is not None and past_mask.dim() == 3:
                m_c = (past_mask[:, :-N, :pastH - N]
                       * past_mask[:, N:, N:pastH]).unsqueeze(-1)   # (B, T-N, H-N, 1)
            elif past_mask is not None:
                m_c = (past_mask[:, :-N]
                       * past_mask[:, N:]).unsqueeze(-1).unsqueeze(-1)  # (B, T-N, 1, 1)
            else:
                m_c = torch.ones(pred_t.shape[:3] + (1,), device=device)
            sq_cons = (pred_t - pred_tN).pow(2) * m_c
            consistency_loss = consistency_loss + sq_cons.sum() / (m_c.sum() * D_a + 1e-8)
            n_terms += 1
        if n_terms > 1:
            consistency_loss = consistency_loss / n_terms

    # ---- VQ loss (already weighted by commitment_weight in model.forward) --
    vq_loss_val = torch.tensor(0.0, device=device)
    if vq_loss is not None and torch.is_tensor(vq_loss) and vq_loss.item() != 0.0:
        vq_loss_val = vq_loss

    total = (obs_loss_weight         * obs_mse
             + future_loss_weight    * future_mse
             + past_loss_weight      * past_mse
             + l1_weight             * l1_loss
             + past_consistency_weight * consistency_loss
             + future_mode_cls_weight * future_mode_cls
             + future_entropy_weight  * future_entropy
             + vq_loss_val)

    # When all heads are disabled, total is a plain constant with no grad_fn.
    # Ensure it is differentiable so .backward() does not error.
    if not total.requires_grad:
        total = total.requires_grad_()

    return {
        "loss":               total,
        "obs_mse_loss":       obs_mse,
        "future_mse_loss":    future_mse,
        "past_mse_loss":      past_mse,
        "l1_loss":            l1_loss,
        "consistency_loss":   consistency_loss,
        "vq_loss":            vq_loss_val,
        "future_mode_cls_loss": future_mode_cls,
        "future_entropy_loss":  future_entropy,
        "obs_mse_per_dim":    obs_per_dim,
        "future_mse_per_step": future_per_step,
    }
