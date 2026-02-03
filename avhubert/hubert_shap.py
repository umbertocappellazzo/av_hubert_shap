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
- Uses SamplingExplainer (matching Llama-AVSR and Whisper-Flamingo)
"""

import torch
import torch.nn.functional as F
import numpy as np
import shap
from typing import Tuple, Dict, List, Optional
import logging
import warnings
logger = logging.getLogger(__name__)


def extract_features_separate(
    model,
    src_audio: torch.Tensor,
    src_video: torch.Tensor,
    debug: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract audio and video features separately BEFORE fusion.
    
    This is the key function for SHAP analysis - we extract features
    for each modality independently so we can apply masks before fusion.
    
    Args:
        model: The AVHubertModel (w2v_model inside HubertEncoderWrapper)
        src_audio: Audio input tensor [B, C, T_audio]
        src_video: Video input tensor [B, C, T, H, W]
        debug: Enable debug output

        
    Returns:
        features_audio: [B, F, T] where F = encoder_embed_dim
        features_video: [B, F, T] where F = encoder_embed_dim
    """
    
    if debug:
        print(f"\n[DEBUG extract_features_separate]")
        print(f"  Input src_audio shape: {src_audio.shape}")
        print(f"  Input src_video shape: {src_video.shape}")


    with torch.no_grad():
        # Extract features using the model's feature extractors
        features_audio = model.forward_features(src_audio, modality='audio')  # [B, F, T]
        features_video = model.forward_features(src_video, modality='video')  # [B, F, T]
    
    # Validate outputs
    if torch.isnan(features_audio).any():
        warnings.warn("NaN detected in audio features!")
    if torch.isnan(features_video).any():
        warnings.warn("NaN detected in video features!")
    
    if debug:
        print(f"  Output features_audio: {features_audio.shape}, range: [{features_audio.min():.4f}, {features_audio.max():.4f}]")
        print(f"  Output features_video: {features_video.shape}, range: [{features_video.min():.4f}, {features_video.max():.4f}]")
    
    return features_audio, features_video


def forward_with_masked_features(
    model,
    features_audio: torch.Tensor,
    features_video: torch.Tensor,
    audio_mask: np.ndarray,
    video_mask: np.ndarray,
    padding_mask: Optional[torch.Tensor] = None,
    debug: bool = False,
) -> Dict[str, torch.Tensor]:
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
        audio_mask: [T] binary mask, 1=keep, 0=zero out (numpy array)
        video_mask: [T] binary mask, 1=keep, 0=zero out (numpy array)
        padding_mask: Optional [B, T_original] padding mask
        debug: Enable debug output
        
    Returns:
        encoder_out: dict with "encoder_out" [T, B, C] and "padding_mask" [B, T]
    """
    device = features_audio.device
    T = features_audio.shape[2]
    
    # Clone features to avoid modifying originals
    audio_masked = features_audio.clone()
    video_masked = features_video.clone()
    
    # Apply masks by zeroing out features at masked time steps
    for t in range(T):
        if audio_mask[t] == 0:
            audio_masked[:, :, t] = 0
        if video_mask[t] == 0:
            video_masked[:, :, t] = 0
    
    if debug:
        audio_zero_ratio = (audio_masked == 0).float().mean().item()
        video_zero_ratio = (video_masked == 0).float().mean().item()
        print(f"    Audio masked zero ratio: {audio_zero_ratio:.4f}")
        print(f"    Video masked zero ratio: {video_zero_ratio:.4f}")
    
    # Fuse features based on model configuration
    if model.modality_fuse == 'concat':
        features = torch.cat([audio_masked, video_masked], dim=1)  # [B, 2F, T]
    elif model.modality_fuse == 'add':
        features = audio_masked + video_masked  # [B, F, T]
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
    debug: bool = False,
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
        debug: Enable debug output
        
    Returns:
        tokens: [seq_len] tensor of token indices (WITHOUT bos, WITH eos if generated)
    """
    device = source['audio'].device
    
    if debug:
        print(f"\n[DEBUG generate_baseline_greedy]")
        print(f"  Max length: {max_len}")
        print(f"  BOS idx: {bos_idx}, EOS idx: {eos_idx}")
    
    with torch.no_grad():
        # Get encoder output using standard forward pass
        encoder_out = model.encoder(source=source, padding_mask=padding_mask)
        
        # Start with BOS token
        generated = [bos_idx]
        
        for step in range(max_len):
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
            
            if debug and step < 5:
                print(f"  Step {step}: predicted token {next_token}")
            
            if next_token == eos_idx:
                if debug:
                    print(f"  EOS reached at step {step}")
                break
    
    # Return tokens WITHOUT bos (for consistency with other implementations)
    # The returned sequence includes eos if generated
    result = torch.tensor(generated[1:], dtype=torch.long, device=device)
    
    if debug:
        print(f"  Total tokens generated: {len(result)}")
        print(f"  First 10 tokens: {result[:10].tolist()}")
    
    return result





def evaluate_coalitions_avhubert(
    model,
    masks: np.ndarray,
    features_audio: torch.Tensor,
    features_video: torch.Tensor,
    baseline_tokens_generated: torch.Tensor,
    baseline_tokens_full: torch.Tensor,
    bos_idx: int,
    decoder,
    padding_mask: Optional[torch.Tensor] = None,
    debug: bool = False,
    coalition_idx: int = 0,
) -> np.ndarray:
    """
    SHAP wrapper: evaluate coalitions via teacher forcing.
    
    This function is called by SHAP to evaluate different feature coalitions.
    It returns the raw logits for each baseline token, matching Whisper-Flamingo.
    
    Args:
        model: The AVHubertModel (w2v_model)
        masks: [n_coalitions, p] binary masks where p = 2*T
        features_audio: [1, F, T] pre-extracted audio features
        features_video: [1, F, T] pre-extracted video features
        baseline_tokens_generated: [seq_len] baseline tokens WITHOUT bos
        baseline_tokens_full: [seq_len+1] baseline tokens WITH bos
        bos_idx: BOS token index
        decoder: The TransformerDecoder
        padding_mask: Optional padding mask
        debug: Enable debug output
        coalition_idx: Running index for debug output
        
    Returns:
        results: [n_coalitions, seq_len] raw logits for each token
    """
    if masks.ndim == 1:
        masks = masks.reshape(1, -1)
    
    n_coalitions = masks.shape[0]
    
    # Handle empty coalition case
    if n_coalitions == 0:
        T_out = len(baseline_tokens_generated)
        return np.empty((0, T_out), dtype=np.float32)
    
    device = features_audio.device
    T = features_audio.shape[2]  # Number of time steps
    
    # Mask dimensions
    N_a = T  # Audio mask elements
    N_v = T  # Video mask elements
    
    if debug and coalition_idx == 0:
        print(f"\n[DEBUG evaluate_coalitions_avhubert]")
        print(f"  Number of coalitions: {n_coalitions}")
        print(f"  Audio timesteps: {T}, Video timesteps: {T}")
        print(f"  Baseline tokens (generated): {len(baseline_tokens_generated)}")
        print(f"  Baseline tokens (full with BOS): {len(baseline_tokens_full)}")
    
    results = []
    
    for i in range(n_coalitions):
        mask = masks[i]
        
        if debug and (coalition_idx + i) == 0:
            print(f"\n  Coalition 0 analysis:")
            print(f"    Mask shape: {mask.shape}")
            print(f"    Audio mask elements kept: {mask[:N_a].sum()}/{N_a} ({mask[:N_a].mean()*100:.1f}%)")
            print(f"    Video mask elements kept: {mask[N_a:].sum()}/{N_v} ({mask[N_a:].mean()*100:.1f}%)")
        
        # Split mask into audio and video
        mask_audio = mask[:N_a]
        mask_video = mask[N_a:]
        
        # Get encoder output with masked features
        with torch.no_grad():
            encoder_out = forward_with_masked_features(
                model,
                features_audio,
                features_video,
                mask_audio,
                mask_video,
                padding_mask,
                debug=(debug and (coalition_idx + i) == 0),
            )
            
            # Teacher forcing: run decoder with full baseline sequence
            logits, _ = decoder(
                prev_output_tokens=baseline_tokens_full.unsqueeze(0),
                encoder_out=encoder_out,
                incremental_state=None,
            )
            # logits: [1, seq_len+1, vocab_size]
            
            if debug and (coalition_idx + i) == 0:
                print(f"    Decoder output logits shape: {logits.shape}")
                print(f"    Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
            
            # Extract logits for baseline tokens
            # Off-by-one: logits[t] predicts token at position t+1
            # baseline_tokens_full = [bos, tok0, tok1, ..., tok_{n-1}]
            # logits positions:       [0,   1,    2,   ..., n]
            # logits[0] predicts tok0, logits[1] predicts tok1, etc.
            # baseline_tokens_generated = [tok0, tok1, ..., tok_{n-1}]
            
            T_out = len(baseline_tokens_generated)
            positions = torch.arange(T_out, device=device)  # [0, 1, 2, ..., T_out-1]
            
            if debug and (coalition_idx + i) == 0:
                print(f"    Extracting logits at positions: {positions[:5].tolist()} ... {positions[-3:].tolist()}")
                print(f"    For tokens: {baseline_tokens_generated[:5].tolist()} ... {baseline_tokens_generated[-3:].tolist()}")
            
            # Get logits for the predicted tokens
            # logits[0, t, baseline_tokens_generated[t]] gives the logit for predicting token t
            logit_vec = logits[0, positions, baseline_tokens_generated]
            
            if debug and (coalition_idx + i) == 0:
                print(f"    Extracted logit_vec shape: {logit_vec.shape}")
                print(f"    Logit values range: [{logit_vec.min():.4f}, {logit_vec.max():.4f}]")
                print(f"    First 5 logits: {logit_vec[:5].tolist()}")
            
            logit_vec_np = logit_vec.detach().cpu().numpy()
            
            if np.isnan(logit_vec_np).any():
                warnings.warn(f"NaN detected in coalition {coalition_idx + i} logits!")
            if np.isinf(logit_vec_np).any():
                warnings.warn(f"Inf detected in coalition {coalition_idx + i} logits!")
            
            results.append(logit_vec_np)
    
    result_array = np.array(results, dtype=np.float32)
    
    if debug and coalition_idx == 0:
        print(f"\n  Final results array shape: {result_array.shape}")
        print(f"  Results range: [{result_array.min():.4f}, {result_array.max():.4f}]")
    
    return result_array


def forward_shap_avhubert(
    model,
    source: Dict[str, torch.Tensor],
    padding_mask: Optional[torch.Tensor],
    bos_idx: int,
    eos_idx: int,
    nsamples: int = 2000,
    shap_alg: str = "kernel",
    device: str = 'cuda',
    verbose: bool = False,
    debug: bool = False,
) -> Tuple[float, float, float, float, float, float, int, np.ndarray]:
    """
    Main SHAP analysis function for AV-HuBERT.
    
    This function:
    1. Extracts audio/video features separately
    2. Generates baseline transcription with greedy decoding
    3. Runs SHAP analysis to compute Shapley values for each time step
    4. Aggregates values into audio/video contribution percentages
    
    Matches the methodology from Llama-AVSR and Whisper-Flamingo exactly.
    
    Args:
        model: AVHubertSeq2Seq model
        source: Dict with 'audio' and 'video' tensors (batch size 1)
        padding_mask: Optional padding mask
        bos_idx: BOS token index
        eos_idx: EOS token index
        nsamples: Number of SHAP samples (recommend ≥2000)
        shap_alg: SHAP algorithm - "kernel" (SamplingExplainer) or "permutation" (PermutationExplainer)
        device: Device string
        verbose: Enable verbose output
        debug: Enable detailed debug output
        
    Returns:
        Tuple of:
            - audio_pct_abs: Audio contribution (% of total absolute SHAP)
            - video_pct_abs: Video contribution (% of total absolute SHAP)
            - audio_pct_pos: Audio contribution (% of total positive SHAP)
            - video_pct_pos: Video contribution (% of total positive SHAP)
            - audio_pct_neg: Audio contribution (% of total negative SHAP)
            - video_pct_neg: Video contribution (% of total negative SHAP)
            - T: Number of time steps
            - vals: Raw SHAP values array (p, T_tokens)
    """
    model.eval()
    
    # Verify single sample
    assert source['audio'].shape[0] == 1, f"Expected batch size 1, got {source['audio'].shape[0]}"
    assert source['video'].shape[0] == 1, f"Expected batch size 1, got {source['video'].shape[0]}"
    
    if debug:
        print("\n" + "="*80)
        print("SHAP COMPUTATION DEBUG MODE - AV-HuBERT")
        print("="*80)
    
    # Get the underlying w2v_model (AVHubertModel) and decoder
    w2v_model = model.encoder.w2v_model
    decoder = model.decoder
    
    # 1. Extract features
    if verbose or debug:
        print("\n[1] Extracting features...")
    
    features_audio, features_video = extract_features_separate(
        w2v_model,
        source['audio'],
        source['video'],
        debug=debug,
    )
    # Both are [1, F, T]
    
    T = features_audio.shape[2]  # Number of time steps
    N_a = T  # Audio mask elements (1:1 mapping, no grouping needed)
    N_v = T  # Video mask elements
    p = N_a + N_v  # Total mask features
    
    if verbose or debug:
        print(f"  Audio features: {features_audio.shape}")
        print(f"  Video features: {features_video.shape}")
        print(f"  Time steps T: {T}")
        print(f"  Total mask features p: {p} (audio: {N_a}, video: {N_v})")
    
    # 2. Generate baseline tokens
    if verbose or debug:
        print("\n[2] Generating baseline tokens...")
    
    baseline_tokens_generated = generate_baseline_greedy(
        model,
        source,
        padding_mask,
        decoder,
        bos_idx,
        eos_idx,
        debug=debug,
    )
    
    if len(baseline_tokens_generated) == 0:
        raise ValueError("Baseline generation failed: no tokens generated")
    
    if verbose or debug:
        print(f"  Baseline tokens: {len(baseline_tokens_generated)}")
    
    # Create full sequence WITH BOS for teacher forcing
    bos_tensor = torch.tensor([bos_idx], dtype=torch.long, device=source['audio'].device)
    baseline_tokens_full = torch.cat([bos_tensor, baseline_tokens_generated])
    
    if debug:
        print(f"\n  Baseline construction:")
        print(f"    BOS token: {bos_idx}")
        print(f"    Generated tokens length: {len(baseline_tokens_generated)}")
        print(f"    Full baseline length: {len(baseline_tokens_full)}")
    
    # 3. SHAP setup
    if verbose or debug:
        print(f"\n[3] SHAP setup:")
        print(f"  Total mask features: {p} (audio: {N_a}, video: {N_v})")
    
    background = np.zeros((1, p), dtype=np.float32)
    x_explain = np.ones((1, p), dtype=np.float32)
    
    if debug:
        print(f"  Background (all removed): {background.shape}")
        print(f"  Explain (all present): {x_explain.shape}")
    
    # 4. SHAP wrapper function
    coalition_counter = [0]  # Mutable counter for debugging
    
    def shap_model(masks):
        result = evaluate_coalitions_avhubert(
            w2v_model,
            masks,
            features_audio,
            features_video,
            baseline_tokens_generated,
            baseline_tokens_full,
            bos_idx,
            decoder,
            padding_mask,
            debug=debug,
            coalition_idx=coalition_counter[0],
        )
        coalition_counter[0] += masks.shape[0] if masks.ndim > 1 else 1
        return result
    
    # 5. Compute SHAP (matching Whisper-Flamingo exactly)
    if verbose or debug:
        print(f"\n[4] Computing SHAP with {nsamples} samples using {shap_alg}...")
    
    if shap_alg == "kernel":
        explainer = shap.SamplingExplainer(
            model=shap_model,
            data=background,
        )
        shap_values_raw = explainer.shap_values(x_explain, nsamples=nsamples)
        
        # DEBUG: See what SHAP returns
        if debug or verbose:
            print(f"\n[SHAP OUTPUT DEBUG]")
            print(f"  Type: {type(shap_values_raw)}")
            if isinstance(shap_values_raw, list):
                print(f"  List length: {len(shap_values_raw)}")
                print(f"  First element shape: {np.array(shap_values_raw[0]).shape}")
                if len(shap_values_raw) > 1:
                    print(f"  Second element shape: {np.array(shap_values_raw[1]).shape}")
            else:
                print(f"  Array shape: {np.array(shap_values_raw).shape}")
        
        # SHAP 0.44.1 returns list of (1, features) arrays, one per token
        # Stack them to get (features, tokens)
        if isinstance(shap_values_raw, list):
            if debug or verbose:
                print(f"  Stacking {len(shap_values_raw)} token outputs...")
            shap_values = np.stack([np.array(sv).squeeze() for sv in shap_values_raw], axis=1)
        else:
            # Unexpected format - try to handle it
            shap_values = np.array(shap_values_raw)
            if shap_values.ndim == 3:
                shap_values = shap_values[0]
    
    elif shap_alg == "permutation":
        from shap.maskers import Independent
        masker = Independent(background, max_samples=100)
        
        explainer = shap.PermutationExplainer(
            model=shap_model,
            masker=masker,
            algorithm='auto'
        )
        shap_obj = explainer(x_explain, max_evals=nsamples, silent=True)
        shap_values = shap_obj.values
    
    else:
        raise ValueError(f"Unknown SHAP algorithm: {shap_alg}")
    
    print(f"\n  Total coalitions evaluated: {coalition_counter[0]}")
    
    # 6. Process SHAP output (MATCHING Whisper-Flamingo EXACTLY)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[0]
    
    # CRITICAL: SHAP returns (p, T_tokens) - one row per feature, one col per token
    vals = shap_values
    
    if verbose or debug:
        print(f"\n[5] SHAP values processing:")
        print(f"  SHAP values shape: {vals.shape}")
        print(f"  Expected shape: ({p}, {len(baseline_tokens_generated)})")
    
    # Validate shape
    expected_shape = (p, len(baseline_tokens_generated))
    if vals.shape != expected_shape:
        warnings.warn(
            f"SHAP values shape {vals.shape} doesn't match expected {expected_shape}!"
        )
    
    if debug:
        print(f"  SHAP values range: [{vals.min():.4f}, {vals.max():.4f}]")
        print(f"  SHAP values mean: {vals.mean():.4f}, std: {vals.std():.4f}")
    
    # 7. Compute metrics (IDENTICAL to Whisper-Flamingo)
    # Absolute SHAP - sum over tokens (axis=1)
    mm_raw_abs = np.sum(np.abs(vals), axis=1)  # (p,)
    mm_audio_abs = mm_raw_abs[:N_a].sum()
    mm_video_abs = mm_raw_abs[N_a:].sum()
    total_abs = mm_audio_abs + mm_video_abs
    
    if debug:
        print(f"\n[6] Computing metrics:")
        print(f"  Audio absolute contribution: {mm_audio_abs:.4f}")
        print(f"  Video absolute contribution: {mm_video_abs:.4f}")
        print(f"  Total absolute: {total_abs:.4f}")
    
    audio_pct_abs = mm_audio_abs / total_abs
    video_pct_abs = mm_video_abs / total_abs
    
    # Positive SHAP
    mm_raw_pos = np.sum(np.maximum(vals, 0), axis=1)
    mm_audio_pos = mm_raw_pos[:N_a].sum()
    mm_video_pos = mm_raw_pos[N_a:].sum()
    total_pos = mm_audio_pos + mm_video_pos
    
    audio_pct_pos = mm_audio_pos / total_pos
    video_pct_pos = mm_video_pos / total_pos
    
    # Negative SHAP
    mm_raw_neg = np.sum(np.abs(np.minimum(vals, 0)), axis=1)
    mm_audio_neg = mm_raw_neg[:N_a].sum()
    mm_video_neg = mm_raw_neg[N_a:].sum()
    total_neg = mm_audio_neg + mm_video_neg
    
    audio_pct_neg = mm_audio_neg / total_neg
    video_pct_neg = mm_video_neg / total_neg
    
    if verbose or debug:
        print(f"\n[7] Final Results:")
        print(f"  Absolute - Audio: {audio_pct_abs*100:.2f}%, Video: {video_pct_abs*100:.2f}%")
        print(f"  Positive - Audio: {audio_pct_pos*100:.2f}%, Video: {video_pct_pos*100:.2f}%")
        print(f"  Negative - Audio: {audio_pct_neg*100:.2f}%, Video: {video_pct_neg*100:.2f}%")
    
    if debug:
        print("="*80)
        print("SHAP COMPUTATION COMPLETE")
        print("="*80 + "\n")
    
    return (
        audio_pct_abs, video_pct_abs,
        audio_pct_pos, video_pct_pos,
        audio_pct_neg, video_pct_neg,
        T, vals
    )


def run_sanity_checks(
    model,
    source: Dict[str, torch.Tensor],
    padding_mask: Optional[torch.Tensor],
    bos_idx: int,
    eos_idx: int,
    debug: bool = False,
) -> Dict[str, float]:
    """
    Run sanity checks to verify SHAP implementation.
    
    Checks:
    1. All features present → baseline log-probs
    2. Zero audio → should decrease log-probs
    3. Zero video → should decrease log-probs
    4. Zero both → should have lowest log-probs
    
    Args:
        model: AVHubertSeq2Seq model
        source: Dict with 'audio' and 'video' tensors
        padding_mask: Optional padding mask
        bos_idx: BOS token index
        eos_idx: EOS token index
        debug: Enable debug output
        
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
        debug=debug,
    )
    
    T = features_audio.shape[2]
    
    # Generate baseline
    baseline_tokens_generated = generate_baseline_greedy(
        model, source, padding_mask, decoder, bos_idx, eos_idx, debug=debug
    )
    
    if baseline_tokens_generated.numel() == 0:
        return {"error": "Empty baseline"}
    
    # Create full sequence with BOS
    bos_tensor = torch.tensor([bos_idx], dtype=torch.long, device=device)
    baseline_tokens_full = torch.cat([bos_tensor, baseline_tokens_generated])
    
    # Helper function to compute sum of logits
    def compute_logit_sum(audio_mask, video_mask):
        masks = np.concatenate([audio_mask, video_mask]).reshape(1, -1)
        result = evaluate_coalitions_avhubert(
            w2v_model, masks, features_audio, features_video,
            baseline_tokens_generated, baseline_tokens_full, bos_idx, decoder,
            padding_mask, debug=False, coalition_idx=0
        )
        return result[0].sum()
    
    # Check 1: All features present (baseline)
    audio_mask_full = np.ones(T, dtype=np.float32)
    video_mask_full = np.ones(T, dtype=np.float32)
    results['logit_sum_full'] = compute_logit_sum(audio_mask_full, video_mask_full)
    
    # Check 2: Zero audio
    audio_mask_zero = np.zeros(T, dtype=np.float32)
    results['logit_sum_no_audio'] = compute_logit_sum(audio_mask_zero, video_mask_full)
    
    # Check 3: Zero video
    video_mask_zero = np.zeros(T, dtype=np.float32)
    results['logit_sum_no_video'] = compute_logit_sum(audio_mask_full, video_mask_zero)
    
    # Check 4: Zero both
    results['logit_sum_no_both'] = compute_logit_sum(audio_mask_zero, video_mask_zero)
    
    # Compute contribution estimates
    full = results['logit_sum_full']
    no_audio = results['logit_sum_no_audio']
    no_video = results['logit_sum_no_video']
    no_both = results['logit_sum_no_both']
    
    total_range = full - no_both
    if abs(total_range) > 1e-6:
        results['audio_contribution_est'] = (full - no_audio) / total_range
        results['video_contribution_est'] = (full - no_video) / total_range
    else:
        results['audio_contribution_est'] = 0.5
        results['video_contribution_est'] = 0.5
    
    if debug:
        print("\n[SANITY CHECK RESULTS]")
        print(f"  Logit sum (all present): {full:.4f}")
        print(f"  Logit sum (no audio): {no_audio:.4f}")
        print(f"  Logit sum (no video): {no_video:.4f}")
        print(f"  Logit sum (no both): {no_both:.4f}")
        print(f"  Audio contribution estimate: {results['audio_contribution_est']:.4f}")
        print(f"  Video contribution estimate: {results['video_contribution_est']:.4f}")
    
    return results