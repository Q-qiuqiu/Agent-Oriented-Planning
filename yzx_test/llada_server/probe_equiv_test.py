#!/usr/bin/env python
"""In-process equivalence and probe-behaviour check.

Runs one MMLU planner prompt through generate_with_dual_cache four times on
the same loaded model:

1. vanilla dual cache, no Agent controller
2. prediction-only controller, probe_period=0 (P0)
3. prediction-only controller, probe_period=8 (P8)
4. prediction-only controller, probe_period=4 (P4)

Verifies that all four produce byte-identical token canvases and identical
generation NFE (probes excluded from NFE by design), and prints the
prediction/materialization metrics each probe configuration observed.

Usage:
    CUDA_VISIBLE_DEVICES=1 python probe_equiv_test.py
"""

import json
import sys
from pathlib import Path

import torch

from generate import generate_with_dual_cache
from json_agent_priority import (
    JsonAgentFieldController,
    JsonAgentPriorityConfig,
    extract_agent_registry,
)

ROOT = Path(__file__).resolve().parent
YZX_TEST = ROOT.parent


def load_planner_prompt():
    sys.path.insert(0, str(YZX_TEST / "mmlu_test"))
    module = __import__("build_subtask_full_benchmark_v2")
    queries = module.load_queries(YZX_TEST / "benchmarks/mmlu/mmlu_pro_sampled.json")
    return module.FULL_PLANNER_PROMPT, queries[0]["query"]


def main():
    from transformers import AutoTokenizer

    from model.modeling_llada import LLaDAModelLM

    device = "cuda"
    model = LLaDAModelLM.from_pretrained(
        "/data/labshare/Param/llada",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/labshare/Param/llada", trust_remote_code=True
    )

    system_prompt, query = load_planner_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    registry = extract_agent_registry(messages, [])
    print(f"registry: {registry}")

    rendered = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    input_ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
    mask_id = tokenizer.mask_token_id or 126336

    def run(probe_period):
        controller = None
        if probe_period is not None:
            controller = JsonAgentFieldController(
                tokenizer=tokenizer,
                config=JsonAgentPriorityConfig(
                    catalog=registry,
                    priority_slots=3,
                    tracking_slots=3,
                    anchor_min_logit_margin=-6.0,
                    tentative_probability=0.45,
                    tentative_margin=0.20,
                    probe_period=probe_period,
                ),
                prompt_length=input_ids.shape[1],
                gen_length=1024,
                mask_id=mask_id,
            )
        with torch.inference_mode():
            x, nfe = generate_with_dual_cache(
                model,
                input_ids,
                steps=1024,
                gen_length=1024,
                block_length=32,
                temperature=0.0,
                remasking="low_confidence",
                threshold=0.8,
                mask_id=mask_id,
                agent_controller=controller,
                probe_period=probe_period,
            )
        return x, nfe, controller

    results = {}
    for label, period in [("vanilla", None), ("P0", 0), ("P8", 8), ("P4", 4)]:
        x, nfe, controller = run(period)
        results[label] = (x, nfe, controller)
        print(f"{label}: nfe={nfe}")
        if controller is not None:
            metrics = controller.metrics()
            print(
                f"  probe_forwards={metrics['probe_forwards']} "
                f"plan_complete={metrics['plan_complete_seconds']}s@{metrics['plan_complete_step']}"
            )
            for slot in metrics["agent_slots"]:
                print(
                    f"  slot{slot['slot']} {slot['agent']}: "
                    f"predicted={slot['predicted_seconds']}s@{slot['predicted_step']} "
                    f"materialized={slot['materialized_seconds']}s@{slot['materialized_step']} "
                    f"recognized={slot['recognized_seconds']}s@{slot['recognized_step']}"
                )

    vanilla_x, vanilla_nfe, _ = results["vanilla"]
    for label in ("P0", "P8", "P4"):
        x, nfe, _ = results[label]
        tokens_equal = bool(torch.equal(x, vanilla_x))
        print(
            f"equivalence {label} vs vanilla: tokens_equal={tokens_equal} "
            f"nfe_equal={nfe == vanilla_nfe}"
        )
        if not tokens_equal:
            diff = (x != vanilla_x).sum().item()
            print(f"  MISMATCH positions={diff}")
            raise SystemExit(1)
    print("ALL_EQUIVALENT")


if __name__ == "__main__":
    main()
