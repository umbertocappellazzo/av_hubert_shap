#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 17:29:58 2026

@author: umbertocappellazzo
"""

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SHAP analysis utilities for AV-HuBERT model.

This module implements SHAP (SHapley Additive exPlanations) analysis to measure
audio vs video modality contributions in AV-HuBERT for audio-visual speech recognition.

Key design decisions:
- Masking point: After feature extraction, BEFORE fusion (concat/add)
- Both modalities have equal temporal resolution (4:1 audio stacking → 25Hz)
- No grouped masking needed (unlike Whisper-Flamingo with 2:1 ratio)
- Teacher forcing with greedy-decoded baseline for coalition evaluation
"""

import torch
import torch.nn.functional as F
import numpy as np
import shap
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def extract_features_separate(
    model,
    src_audio: torch.Tensor,
    src_video: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract audio and video features separately BEFORE fusion.
    
    This is the key function for SHAP analysis - we extract features
    for each modality independently so we can apply masks before fusion.
    
    Args:
        model: The AVHubertModel (w2v_model inside HubertEncoderWrapper)
        src_audio: Audio input tensor [B, C, T_audio]
        src_video: Video input tensor [B, C, T, H, W]
        
    Returns:
        features_audio: [B, F, T] where F = encoder_embed_dim
        features_video: [B, F, T] where F = encoder_embed_dim
    """
    with torch.no_grad():
        # Extract features using the model's feature extractors
        features_audio = model.forward_features(src_audio, modality='audio')  # [B, F, T]
        features_video = model.forward_features(src_video, modality='video')  # [B, F, T]
    
    return features_audio, features_video


def forward_with_masked_features(
    model,
    features_audio: torch.Tensor,
    features_video: torch.Tensor,
    audio_mask: torch.Tensor,
    video_mask: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Forward pass through encoder with masked features.
    
    This function:
    1. Applies masks to audio/video features (zero out masked time steps)
    2. Fuses features (concat or add based on model config)
    3. Passes through layer norm, projection, and transformer encoder
    
    Args:
        model: The AVHubertModel (w2v_model)
        features_audio: [B, F, T] audio features
        features_video: [B, F, T] video features
        audio_mask: [T] binary mask, 1=keep, 0=zero out
        video_mask: [T] binary mask, 1=keep, 0=zero out
        padding_mask: Optional [B, T_original] padding mask
        
    Returns:
        encoder_out: dict with "encoder_out" [T, B, C] and "padding_mask" [B, T]
    """
    # Apply masks by zeroing out features at masked time steps
    # audio_mask and video_mask are [T], need to broadcast to [B, F, T]
    audio_mask_expanded = audio_mask.view(1, 1, -1).expand_as(features_audio)
    video_mask_expanded = video_mask.view(1, 1, -1).expand_as(features_video)
    
    masked_audio = features_audio * audio_mask_expanded.float()
    masked_video = features_video * video_mask_expanded.float()
    
    # Fuse features based on model configuration
    if model.modality_fuse == 'concat':
        features = torch.cat([masked_audio, masked_video], dim=1)  # [B, 2F, T]
    elif model.modality_fuse == 'add':
        features = masked_audio + masked_video  # [B, F, T]
    else:
        raise ValueError(f"Unknown modality_fuse: {model.modality_fuse}")
    
    # Continue through the rest of the encoder pipeline
    # (Replicate the logic from extract_finetune after fusion)
    
    # Transpose: [B, F, T] -> [B, T, F]
    features = features.transpose(1, 2)
    
    # Layer norm
    features = model.layer_norm(features)
    
    # Handle padding mask if provided
    if padding_mask is not None:
        padding_mask = model.forward_padding_mask(features, padding_mask)
    
    # Post-extract projection (if embed != encoder_embed_dim)
    if model.post_extract_proj is not None:
        features = model.post_extract_proj(features)
    
    # Dropout (use eval mode behavior - no dropout during inference)
    features = model.dropout_input(features)
    
    x = features
    
    # Pass through transformer encoder
    x, _ = model.encoder(
        x,
        padding_mask=padding_mask,
        layer=None
    )
    
    # Transpose for decoder: B x T x C -> T x B x C
    x = x.transpose(0, 1)
    
    return {
        "encoder_out": x,  # [T, B, C]
        "encoder_padding_mask": padding_mask,  # [B, T]
        "padding_mask": padding_mask,
    }


def generate_baseline_greedy(
    model,
    source: Dict[str, torch.Tensor],
    padding_mask: Optional[torch.Tensor],
    decoder,
    bos_idx: int,
    eos_idx: int,
    max_len: int = 256,
) -> torch.Tensor:
    """
    Generate baseline transcription using greedy decoding.
    
    This generates the "reference" output that we use for SHAP analysis.
    We use greedy decoding (not beam search) for consistency.
    
    Args:
        model: The full AVHubertSeq2Seq model
        source: Dict with 'audio' and 'video' tensors
        padding_mask: Optional padding mask
        decoder: The TransformerDecoder
        bos_idx: Beginning-of-sequence token index
        eos_idx: End-of-sequence token index
        max_len: Maximum sequence length
        
    Returns:
        tokens: [seq_len] tensor of token indices (WITHOUT bos, WITH eos)
    """
    device = source['audio'].device
    
    with torch.no_grad():
        # Get encoder output using standard forward pass
        encoder_out = model.encoder(source=source, padding_mask=padding_mask)
        
        # Start with BOS token
        generated = [bos_idx]
        
        for _ in range(max_len):
            # Prepare decoder input
            prev_tokens = torch.tensor([generated], dtype=torch.long, device=device)  # [1, seq_len]
            
            # Get decoder output
            logits, _ = decoder(
                prev_output_tokens=prev_tokens,
                encoder_out=encoder_out,
                incremental_state=None,  # Don't use incremental decoding for simplicity
            )
            
            # Get the last token prediction
            next_token_logits = logits[0, -1, :]  # [vocab_size]
            next_token = next_token_logits.argmax().item()
            
            generated.append(next_token)
            
            if next_token == eos_idx:
                break
    
    # Return tokens WITHOUT bos (for consistency with other implementations)
    # The returned sequence includes eos if generated
    return torch.tensor(generated[1:], dtype=torch.long, device=device)


def compute_log_probs_teacher_forcing(
    w2v_model,
    decoder,
    features_audio: torch.Tensor,
    features_video: torch.Tensor,
    audio_mask: torch.Tensor,
    video_mask: torch.Tensor,
    prev_output_tokens: torch.Tensor,
    baseline_tokens: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute log probabilities for baseline tokens using teacher forcing.
    
    This is the core function for SHAP coalition evaluation. Given masked
    features and the full decoder input (with teacher forcing), we compute
    the log probability of each baseline token.
    
    Args:
        w2v_model: The AVHubertModel (encoder)
        decoder: The TransformerDecoder
        features_audio: [B, F, T] pre-extracted audio features
        features_video: [B, F, T] pre-extracted video features
        audio_mask: [T] binary mask for audio
        video_mask: [T] binary mask for video
        prev_output_tokens: [B, seq_len+1] decoder input WITH bos
        baseline_tokens: [seq_len] baseline tokens WITHOUT bos (what we predict)
        padding_mask: Optional [B, T_original] padding mask
        
    Returns:
        log_probs: [seq_len] log probabilities for each baseline token
    """
    with torch.no_grad():
        # Get encoder output with masked features
        encoder_out = forward_with_masked_features(
            w2v_model,
            features_audio,
            features_video,
            audio_mask,
            video_mask,
            padding_mask,
        )
        
        # Run decoder with teacher forcing
        logits, _ = decoder(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=None,
        )
        # logits: [B, seq_len+1, vocab_size] where seq_len+1 includes bos position
        
        # Extract log probabilities for baseline tokens
        # Off-by-one handling: logits[t] predicts token at position t+1
        # prev_output_tokens = [bos, tok0, tok1, ..., tok_{n-1}]
        # logits positions:     [0,   1,    2,   ...,  n]
        # logits[0] predicts tok0, logits[1] predicts tok1, etc.
        # baseline_tokens = [tok0, tok1, ..., tok_{n-1}] (no bos)
        # So: log_prob[t] = log_softmax(logits[0, t, :])[baseline_tokens[t]]
        
        log_probs_all = F.log_softmax(logits, dim=-1)  # [B, seq_len+1, vocab_size]
        
        seq_len = baseline_tokens.size(0)
        log_probs = torch.zeros(seq_len, device=logits.device)
        
        for t in range(seq_len):
            # logits at position t predicts baseline_tokens[t]
            log_probs[t] = log_probs_all[0, t, baseline_tokens[t]]
        
        return log_probs


def evaluate_coalitions_avhubert(
    w2v_model,
    decoder,
    features_audio: torch.Tensor,
    features_video: torch.Tensor,
    baseline_tokens: torch.Tensor,
    bos_idx: int,
    padding_mask: Optional[torch.Tensor] = None,
):
    """
    Create a coalition evaluation function for SHAP.
    
    This returns a function that takes a mask array and returns the sum of
    log probabilities for the baseline tokens. This is what SHAP uses to
    compute Shapley values.
    
    Args:
        w2v_model: The AVHubertModel
        decoder: The TransformerDecoder
        features_audio: [1, F, T] audio features (batch size 1)
        features_video: [1, F, T] video features (batch size 1)
        baseline_tokens: [seq_len] baseline tokens (no bos)
        bos_idx: BOS token index
        padding_mask: Optional padding mask
        
    Returns:
        eval_fn: Function that takes mask array [n_coalitions, p] and returns [n_coalitions]
    """
    device = features_audio.device
    T = features_audio.size(2)  # Number of time steps
    seq_len = baseline_tokens.size(0)
    
    # Prepare prev_output_tokens for teacher forcing: [bos, tok0, ..., tok_{n-1}]
    prev_output_tokens = torch.cat([
        torch.tensor([bos_idx], dtype=torch.long, device=device),
        baseline_tokens
    ]).unsqueeze(0)  # [1, seq_len+1]
    
    def eval_fn(mask_array: np.ndarray) -> np.ndarray:
        """
        Evaluate coalitions.
        
        Args:
            mask_array: [n_coalitions, p] where p = 2*T
                        First T elements: audio mask
                        Last T elements: video mask
                        
        Returns:
            scores: [n_coalitions] sum of log probs for each coalition
        """
        n_coalitions = mask_array.shape[0]
        scores = np.zeros(n_coalitions)
        
        for i in range(n_coalitions):
            mask = mask_array[i]  # [p] = [2*T]
            
            # Split into audio and video masks
            audio_mask = torch.tensor(mask[:T], dtype=torch.float32, device=device)
            video_mask = torch.tensor(mask[T:], dtype=torch.float32, device=device)
            
            # Compute log probs with this coalition
            log_probs = compute_log_probs_teacher_forcing(
                w2v_model,
                decoder,
                features_audio,
                features_video,
                audio_mask,
                video_mask,
                prev_output_tokens,
                baseline_tokens,
                padding_mask,
            )
            
            # Sum log probs as the coalition score
            scores[i] = log_probs.sum().item()
        
        return scores
    
    return eval_fn


def forward_shap_avhubert(
    model,
    source: Dict[str, torch.Tensor],
    padding_mask: Optional[torch.Tensor],
    bos_idx: int,
    eos_idx: int,
    n_shap_samples: int = 2000,
) -> Dict[str, float]:
    """
    Main SHAP analysis function for AV-HuBERT.
    
    This function:
    1. Extracts audio/video features separately
    2. Generates baseline transcription with greedy decoding
    3. Runs SHAP analysis to compute Shapley values for each time step
    4. Aggregates values into audio/video contribution percentages
    
    Args:
        model: AVHubertSeq2Seq model
        source: Dict with 'audio' and 'video' tensors (batch size 1)
        padding_mask: Optional padding mask
        bos_idx: BOS token index
        eos_idx: EOS token index
        n_shap_samples: Number of SHAP samples (recommend ≥2000)
        
    Returns:
        Dict with:
            - audio_pct_abs: Audio contribution (% of total absolute SHAP)
            - video_pct_abs: Video contribution (% of total absolute SHAP)
            - audio_pct_pos: Audio contribution (% of total positive SHAP)
            - video_pct_pos: Video contribution (% of total positive SHAP)
            - audio_pct_neg: Audio contribution (% of total negative SHAP)
            - video_pct_neg: Video contribution (% of total negative SHAP)
            - shap_values_audio: Raw SHAP values for audio [T]
            - shap_values_video: Raw SHAP values for video [T]
            - baseline_text: Decoded baseline transcription
            - num_features: Total number of features (2*T)
    """
    device = source['audio'].device
    
    # Get the underlying w2v_model (AVHubertModel)
    w2v_model = model.encoder.w2v_model
    decoder = model.decoder
    
    # Step 1: Extract features separately
    features_audio, features_video = extract_features_separate(
        w2v_model,
        source['audio'],
        source['video'],
    )
    # Both are [1, F, T]
    
    T = features_audio.size(2)
    p = 2 * T  # Total number of features (audio + video time steps)
    
    logger.info(f"Feature dimensions: T={T}, p={p}")
    
    # Step 2: Generate baseline transcription
    baseline_tokens = generate_baseline_greedy(
        model,
        source,
        padding_mask,
        decoder,
        bos_idx,
        eos_idx,
    )
    
    
    logger.info(f"Baseline tokens: {baseline_tokens.shape[0]} tokens")
    
    # Step 3: Create coalition evaluation function
    eval_fn = evaluate_coalitions_avhubert(
        w2v_model,
        decoder,
        features_audio,
        features_video,
        baseline_tokens,
        bos_idx,
        padding_mask,
    )
    
    # Step 4: Run SHAP analysis
    # Background: all features ABSENT (all zeros)
    # When SHAP creates coalitions:
    #   - Features IN coalition use values from test_sample (1 = present)
    #   - Features NOT in coalition use values from background (0 = absent/zeroed)
    background = np.zeros((1, p))
    
    explainer = shap.KernelExplainer(eval_fn, background)
    
    # Explain the "all features present" case
    test_sample = np.ones((1, p))
    
    shap_values = explainer.shap_values(test_sample, nsamples=n_shap_samples)
    
    # Handle different SHAP versions
    if isinstance(shap_values, list):
        # SHAP 0.44.1 returns list of arrays
        shap_values = np.stack(shap_values, axis=0)
    
    # shap_values should be [1, p] or [n_outputs, 1, p]
    if shap_values.ndim == 3:
        shap_values = shap_values.squeeze(1)  # [n_outputs, p]
        shap_values = shap_values[0]  # [p] - take first output
    elif shap_values.ndim == 2:
        shap_values = shap_values[0]  # [p]
    
    
    vals = shap_values
    
    # 7. Compute metrics (IDENTICAL to Llama-AVSR)
    # Absolute SHAP - sum over tokens (axis=1)
    mm_raw_abs = np.sum(np.abs(vals), axis=1)  # (p,)
    mm_audio_abs = mm_raw_abs[:T].sum()
    mm_video_abs = mm_raw_abs[T:].sum()
    total_abs = mm_audio_abs + mm_video_abs
    
    audio_pct_abs = mm_audio_abs / total_abs
    video_pct_abs = mm_video_abs / total_abs
    
    # Positive SHAP
    mm_raw_pos = np.sum(np.maximum(vals, 0), axis=1)
    mm_audio_pos = mm_raw_pos[:T].sum()
    mm_video_pos = mm_raw_pos[T:].sum()
    total_pos = mm_audio_pos + mm_video_pos
    
    audio_pct_pos = mm_audio_pos / total_pos
    video_pct_pos = mm_video_pos / total_pos
    
    # Negative SHAP
    mm_raw_neg = np.sum(np.abs(np.minimum(vals, 0)), axis=1)
    mm_audio_neg = mm_raw_neg[:T].sum()
    mm_video_neg = mm_raw_neg[T:].sum()
    total_neg = mm_audio_neg + mm_video_neg
    
    audio_pct_neg = mm_audio_neg / total_neg
    video_pct_neg = mm_video_neg / total_neg
    
    return (
        audio_pct_abs, video_pct_abs,
        audio_pct_pos, video_pct_pos,
        audio_pct_neg, video_pct_neg,
        vals, baseline_tokens
    )


def run_sanity_checks(
    model,
    source: Dict[str, torch.Tensor],
    padding_mask: Optional[torch.Tensor],
    bos_idx: int,
    eos_idx: int,
) -> Dict[str, float]:
    """
    Run sanity checks to verify SHAP implementation.
    
    Checks:
    1. Zero audio → audio contribution should be ~0%
    2. Zero video → video contribution should be ~0%
    
    Args:
        model: AVHubertSeq2Seq model
        source: Dict with 'audio' and 'video' tensors
        padding_mask: Optional padding mask
        bos_idx: BOS token index
        eos_idx: EOS token index
        
    Returns:
        Dict with sanity check results
    """
    results = {}
    
    w2v_model = model.encoder.w2v_model
    decoder = model.decoder
    device = source['audio'].device
    
    # Extract features
    features_audio, features_video = extract_features_separate(
        w2v_model,
        source['audio'],
        source['video'],
    )
    
    T = features_audio.size(2)
    p = 2 * T
    
    # Generate baseline
    baseline_tokens = generate_baseline_greedy(
        model, source, padding_mask, decoder, bos_idx, eos_idx
    )
    
    if baseline_tokens.numel() == 0:
        return {"error": "Empty baseline"}
    
    # Prepare prev_output_tokens
    prev_output_tokens = torch.cat([
        torch.tensor([bos_idx], dtype=torch.long, device=device),
        baseline_tokens
    ]).unsqueeze(0)
    
    # Check 1: All features present (baseline)
    audio_mask_full = torch.ones(T, device=device)
    video_mask_full = torch.ones(T, device=device)
    
    log_probs_full = compute_log_probs_teacher_forcing(
        w2v_model, decoder, features_audio, features_video,
        audio_mask_full, video_mask_full, prev_output_tokens, baseline_tokens, padding_mask
    )
    results['log_prob_full'] = log_probs_full.sum().item()
    
    # Check 2: Zero audio
    audio_mask_zero = torch.zeros(T, device=device)
    
    log_probs_no_audio = compute_log_probs_teacher_forcing(
        w2v_model, decoder, features_audio, features_video,
        audio_mask_zero, video_mask_full, prev_output_tokens, baseline_tokens, padding_mask
    )
    results['log_prob_no_audio'] = log_probs_no_audio.sum().item()
    
    # Check 3: Zero video
    video_mask_zero = torch.zeros(T, device=device)
    
    log_probs_no_video = compute_log_probs_teacher_forcing(
        w2v_model, decoder, features_audio, features_video,
        audio_mask_full, video_mask_zero, prev_output_tokens, baseline_tokens, padding_mask
    )
    results['log_prob_no_video'] = log_probs_no_video.sum().item()
    
    # Check 4: Zero both
    log_probs_no_both = compute_log_probs_teacher_forcing(
        w2v_model, decoder, features_audio, features_video,
        audio_mask_zero, video_mask_zero, prev_output_tokens, baseline_tokens, padding_mask
    )
    results['log_prob_no_both'] = log_probs_no_both.sum().item()
    
    # Compute contribution estimates
    # Audio contribution: (full - no_audio) / (full - no_both)
    # Video contribution: (full - no_video) / (full - no_both)
    full = results['log_prob_full']
    no_audio = results['log_prob_no_audio']
    no_video = results['log_prob_no_video']
    no_both = results['log_prob_no_both']
    
    total_range = full - no_both
    if abs(total_range) > 1e-6:
        results['audio_contribution_est'] = (full - no_audio) / total_range
        results['video_contribution_est'] = (full - no_video) / total_range
    else:
        results['audio_contribution_est'] = 0.5
        results['video_contribution_est'] = 0.5
    
    return results