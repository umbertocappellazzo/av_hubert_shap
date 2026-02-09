#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 16:53:55 2026

@author: umbertocappellazzo
"""

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SHAP analysis inference script for AV-HuBERT.

This script runs SHAP analysis to measure audio vs video modality contributions
in the AV-HuBERT model for audio-visual speech recognition.

Usage:
    python infer_s2s_shap.py --config-dir /path/to/conf --config-name s2s_decode \
        common.user_dir=/path/to/avhubert \
        override.data=/path/to/data \
        override.modalities=['audio','video'] \
        --num-samples-shap 2000 \
        --wandb-project avhubert-shap \
"""

import ast
from itertools import chain
import logging
import math
import os
import sys
import json
import hashlib
import editdistance
import argparse
from argparse import Namespace

import numpy as np
import torch
from fairseq import checkpoint_utils, options, tasks, utils, distributed_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.models import FairseqLanguageModel
from omegaconf import DictConfig

from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
from fairseq.dataclass.configs import (
    CheckpointConfig,
    CommonConfig,
    CommonEvalConfig,
    DatasetConfig,
    DistributedTrainingConfig,
    GenerationConfig,
    FairseqDataclass,
)
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from omegaconf import OmegaConf


def parse_shap_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--shap-alg', default='kernel', choices=['kernel', 'permutation'],
                        help='SHAP algorithm')
    parser.add_argument('--num-samples-shap', type=int, default=2000,
                        help='Number of SHAP samples per evaluation')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--wandb-project', default=None, type=str, 
                        help='WandB project name (if None, WandB disabled)')
    parser.add_argument('--exp-name', default=None, type=str,
                        help='WandB run name (auto-generated if None)')
    parser.add_argument('--run-sanity-check', action='store_true',
                        help='Run sanity checks on first sample')
    parser.add_argument('--output-path', default=None, type=str)
    
    args, remaining = parser.parse_known_args()
    
    # Update sys.argv to only contain arguments Hydra should see
    sys.argv = [sys.argv[0]] + remaining
    
    return args

# Parse before importing Hydra
SHAP_ARGS = parse_shap_args()
    
import wandb

# Import SHAP utilities
from hubert_shap import (
    forward_shap_avhubert,
    run_sanity_checks,
)

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_path = Path(__file__).resolve().parent / "conf"    


@dataclass
class OverrideConfig(FairseqDataclass):
    noise_wav: Optional[str] = field(default=None, metadata={'help': 'noise wav file'})
    noise_prob: float = field(default=0, metadata={'help': 'noise probability'})
    noise_snr: float = field(default=0, metadata={'help': 'noise SNR in audio'})
    modalities: List[str] = field(default_factory=lambda: [""], metadata={'help': 'which modality to use'})
    data: Optional[str] = field(default=None, metadata={'help': 'path to test data directory'})
    label_dir: Optional[str] = field(default=None, metadata={'help': 'path to test label directory'})

@dataclass
class InferConfig(FairseqDataclass):
    task: Any = None
    generation: GenerationConfig = GenerationConfig()
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    override: OverrideConfig = OverrideConfig()
    is_ax: bool = field(
        default=False,
        metadata={
            "help": "if true, assumes we are using ax for tuning and returns a tuple for ax to consume"
        },
    )


def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for recognition!"

    return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos, generator.pad}


def compute_wer(hyp: str, ref: str) -> float:
    """Compute Word Error Rate."""
    hyp_words = hyp.split()
    ref_words = ref.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 100.0
    return 100 * editdistance.eval(hyp_words, ref_words) / len(ref_words)


def _main(cfg, output_file):
    """Main SHAP analysis function."""
    
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("avhubert.shap_recognize")
    
    # Use the pre-parsed custom arguments
    args = SHAP_ARGS
    
    # Parse additional arguments from command line
    
    
    wandb.init(
        project=args.wandb_project,
        name=args.exp_name,
    )
    logger.info(f"WandB initialized: {args.wandb_project}")
    
    
    utils.import_user_module(cfg.common)
    
    logger.info(f"Loading model from: {cfg.common_eval.path}")
    
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([cfg.common_eval.path])
    
    models = [model.eval().cuda() for model in models]
    saved_cfg.task.modalities = cfg.override.modalities
    task = tasks.setup_task(saved_cfg.task)

    task.build_tokenizer(saved_cfg.tokenizer)
    task.build_bpe(saved_cfg.bpe)

    logger.info(cfg)
    
    # Move model to GPU and set to eval mode
    # Fix seed
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    # Get dictionary
    dictionary = task.target_dictionary
    bos_idx = dictionary.bos()
    eos_idx = dictionary.eos()
    pad_idx = dictionary.pad()
    
    logger.info(f"Vocabulary size: {len(dictionary)}, BOS: {bos_idx}, EOS: {eos_idx}, PAD: {pad_idx}")
    
    
    # =========================================================================
    # Dataset loading - MATCHING infer_s2s.py EXACTLY
    # Apply overrides BEFORE loading dataset
    # =========================================================================
    
    # Set noise parameters
    task.cfg.noise_prob = cfg.override.noise_prob
    task.cfg.noise_snr = cfg.override.noise_snr
    task.cfg.noise_wav = cfg.override.noise_wav
    
    # Set data and label paths from overrides
    if cfg.override.data is not None:
        task.cfg.data = cfg.override.data
    if cfg.override.label_dir is not None:
        task.cfg.label_dir = cfg.override.label_dir
    
    # Load dataset with saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)
    
    # Get model
    model = models[0]
    dataset = task.dataset(cfg.dataset.gen_subset)
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Limit samples if specified
    num_samples = len(dataset)
    if args.max_samples is not None:
        num_samples = min(num_samples, args.max_samples)
    
    logger.info(f"Processing {num_samples} samples with {args.num_samples_shap} SHAP samples each")
    logger.info(f"SHAP algorithm: {args.shap_alg}")
    
    
    # =========================================================================
    # Decode function - from infer_s2s.py
    # =========================================================================
    def decode_fn(x):
        if hasattr(task.datasets[cfg.dataset.gen_subset].label_processors[0], 'decode'):
            symbols_ignore = {eos_idx, pad_idx}
            return task.datasets[cfg.dataset.gen_subset].label_processors[0].decode(x, symbols_ignore)
        chars = dictionary.string(x, extra_symbols_to_ignore={eos_idx, pad_idx})
        words = " ".join("".join(chars.split()).replace('|', ' ').split())
        return words
    
    results = {
        'audio_abs': [],
        'video_abs': [],
        'audio_pos': [],
        'video_pos': [],
        'audio_neg': [],
        'video_neg': [],
        'num_audio_tokens': [],
        'shapley_values': [],
    }
    
    # Process samples one by one (SHAP requires batch_size=1)
    for sample_idx in range(num_samples):
        try:
            # Get sample
            sample = dataset[sample_idx]
            
            # Collate single sample
            batch = dataset.collater([sample])
            
            # Move to device
            batch = utils.move_to_cuda(batch) if device.type == 'cuda' else batch
            
            # Extract source and target
            source = batch['net_input']['source']
            padding_mask = batch['net_input'].get('padding_mask', None)
            
            if args.run_sanity_check and sample_idx == 0:
                logger.info("Running sanity checks...")
                sanity_results = run_sanity_checks(
                    model, source, padding_mask, bos_idx, eos_idx, debug=True
                )
                logger.info(f"Sanity check results: {sanity_results}")
            
            # Run SHAP analysis
            
            (audio_abs, video_abs,
             audio_pos, video_pos,
             audio_neg, video_neg,
             num_audio_tokens, shapley_values) = forward_shap_avhubert(
                model,
                source,
                padding_mask,
                bos_idx,
                eos_idx,
                nsamples=args.num_samples_shap,
                shap_alg=args.shap_alg,
                device=str(device),
                verbose=(sample_idx == 0),
                debug=(sample_idx == 0)
            )
            
            
            # Compute WER
            
            # Store results
            results['audio_abs'].append(audio_abs)
            results['video_abs'].append(video_abs)
            results['audio_pos'].append(audio_pos)
            results['video_pos'].append(video_pos)
            results['audio_neg'].append(audio_neg)
            results['video_neg'].append(video_neg)
            results['shapley_values'].append(shapley_values)
            results['num_audio_tokens'].append(num_audio_tokens)
            
            wandb.log({
                'sample_idx': sample_idx,
                'sample_audio_abs': audio_abs,
                'sample_video_abs': video_abs,
                'sample_audio_pos': audio_pos,
                'sample_video_pos': video_pos,
                'sample_audio_neg': audio_neg,
                'sample_video_neg': video_neg,
                'sample_num_audio_tokens': num_audio_tokens
            })
            
        
        except Exception as e:
            logger.error(f"Error processing sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compute aggregate statistics
    print("\n" + "="*80)
    print("SHAP Analysis Complete")
    print("="*80)
 
    mean_audio_abs = np.mean(results['audio_abs'])
    mean_video_abs = np.mean(results['video_abs'])
    mean_audio_pos = np.mean(results['audio_pos'])
    mean_video_pos = np.mean(results['video_pos'])
    mean_audio_neg = np.mean(results['audio_neg'])
    mean_video_neg = np.mean(results['video_neg'])
    mean_num_audio_tokens = np.mean(results['num_audio_tokens'])
    
 
    std_audio_abs = np.std(results['audio_abs'])
    std_video_abs = np.std(results['video_abs'])
 
    print(f"\nAggregate Results (n={len(results['audio_abs'])}):")
    print(f"\nAbsolute SHAP:")
    print(f"  Audio: {mean_audio_abs*100:.2f}% ± {std_audio_abs*100:.2f}%")
    print(f"  Video: {mean_video_abs*100:.2f}% ± {std_video_abs*100:.2f}%")
    print(f"\nPositive SHAP:")
    print(f"  Audio: {mean_audio_pos*100:.2f}%")
    print(f"  Video: {mean_video_pos*100:.2f}%")
    print(f"\nNegative SHAP:")
    print(f"  Audio: {mean_audio_neg*100:.2f}%")
    print(f"  Video: {mean_video_neg*100:.2f}%")
 
    wandb.log({
        'audio-ABS-SHAP': mean_audio_abs,
        'video-ABS-SHAP': mean_video_abs,
        'audio-POS-SHAP': mean_audio_pos,
        'video-POS-SHAP': mean_video_pos,
        'audio-NEG-SHAP': mean_audio_neg,
        'video-NEG-SHAP': mean_video_neg,
        'audio-ABS-STD': std_audio_abs,
        'video-ABS-STD': std_video_abs,
        'num-audio-tokens': mean_num_audio_tokens
    })
    
    
    output_file = os.path.join(
        args.output_path,
        args.exp_name
        
    )
    
    print("Output dir: ", output_file)

    np.savez_compressed(
            output_file,
            # Aggregated metrics
            audio_abs=np.array(results['audio_abs']),
            video_abs=np.array(results['video_abs']),
            audio_pos=np.array(results['audio_pos']),
            video_pos=np.array(results['video_pos']),
            audio_neg=np.array(results['audio_neg']),
            video_neg=np.array(results['video_neg']),
            num_audio_tokens=np.array(results['num_audio_tokens']),
            
            # Raw SHAP values (ragged array - stored as object array)
            shap_values=np.array(results['shapley_values'], dtype=object),
        )

@hydra.main(config_path=config_path, config_name="s2s_decode")
def hydra_main(cfg: InferConfig) -> Union[float, Tuple[float, Optional[float]]]:
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg = OmegaConf.create(container)
    OmegaConf.set_struct(cfg, True)

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, main)
        else:
            distributed_utils.call_main(cfg, main)
    except BaseException as e:
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! %s", str(e))
    return


def cli_main() -> None:
    try:
        from hydra._internal.utils import get_args
        cfg_name = get_args().config_name or "s2s_decode"
    except ImportError:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "s2s_decode"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=InferConfig)

    for k in InferConfig.__dataclass_fields__:
        if is_dataclass(InferConfig.__dataclass_fields__[k].type):
            v = InferConfig.__dataclass_fields__[k].default
            cs.store(name=k, node=v)

    hydra_main()


if __name__ == "__main__":
    cli_main()