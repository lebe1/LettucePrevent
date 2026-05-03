"""
Main entry point experiments.

Runs a single (generator, detector, skip threshold) cell of the W&B sweep, or — when invoked
without W&B — runs once with the constants below.

Sweep dimensions:
  - GENERATOR_MODELS       (3 values)
  - DETECTOR_TYPES_SWEEPED (2 values: lettuceprevent, baseline-run-facts)
  - SKIP_THRESHOLDS        (optional: only confidence level of the generating model lower than skip threshold will be evaluated for hallucinations)

Outside the sweep:
  - 'number' and 'baseline-run-numbers' run on local summary-only data (existing
    experiment is finished, kept available for manual reruns).
"""

import os


import argparse
import json
import random
import time
import warnings
from datetime import datetime

import numpy as np
import torch
import wandb
from datasets import disable_caching
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
    set_seed as hf_set_seed,
)

# IMPORTANT: set the debug-print env var BEFORE importing detector modules,
# since they read it at import time.
DEBUG_PRINT_TO_CONSOLE = False
os.environ["DEBUG_PRINT_TO_CONSOLE"] = "1" if DEBUG_PRINT_TO_CONSOLE else "0"

from detectors.dataset_loader import load_prompts_for_detector
from detectors.factory import DetectorFactory, VALID_DETECTOR_TYPES
from logits_processors.hallucination_logits_processor import HallucinationLogitsProcessor
from analysis.post_eval import (
    evaluate_generated_answers,
    write_hallucinations_json,
    write_stats_txt,
)


warnings.filterwarnings("ignore", message=r".*Pydantic serializer warnings.*")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ===========================================================================
# Top-of-file configuration
# ===========================================================================

GENERATOR_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Llama-2-7b-chat-hf",
    "Qwen/Qwen2.5-14B-Instruct",
]

DETECTOR_TYPES_SWEEPED = [
    "lettuceprevent",
    "baseline-run-facts",
]

SKIP_THRESHOLDS = [0.8,0.9,0.99,1.0]
N_PER_TASK = 20

DETECTOR_BEST_THRESHOLDS = {
    "lettucedetect":  0.6,
    "lettuceprevent": 0.8,
}

LETTUCEDETECT_MODEL_PATH = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
LETTUCEPREVENT_MODEL_PATH = "lebe1/lettuceprevent-ettin-decoder-68m-en"

# Beam search settings. NUM_BEAMS = 1 = greedy (3-4x faster than 4-beam).
NUM_BEAMS = 1
GENERATION_CONFIG_KWARGS = {
    "max_new_tokens":       300,
    "min_length":           150,
    "do_sample":            False,
    "num_beams":            NUM_BEAMS,
    "num_return_sequences": 1,
}

LAST_K_TOKENS_TO_CONSIDER = 10
TOP_K_LOGITS              = 10
PENALTY_VALUE             = 0
USE_ALL_TOKENS            = True

SEED                      = 14
DATA_DIR                  = "./data"
LOCAL_SUMMARY_FILE        = f"{DATA_DIR}/ragtruth_unique_summary_prompts.json"
WANDB_ENTITY              = "lebeccard-technical-university-wien"
WANDB_PROJECT             = "hdm-rq1"


# ===========================================================================
# Determinism
# ===========================================================================

def make_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# ===========================================================================
# Stopping criteria
# ===========================================================================

class TokenPrintStoppingCriteria(StoppingCriteria):
    """Per-token printing for debugging. Only used when DEBUG_PRINT_TO_CONSOLE=True."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, _scores, **_kwargs):
        last_token_str = self.tokenizer.decode([input_ids[0, -1].item()])
        print(f">>> Actually generated token: {repr(last_token_str)}")
        return False


# ===========================================================================
# Helpers
# ===========================================================================

def build_messages_for_generator(generator_model_name: str, prompt: str):
    return [
        {"role": "user",   "content": prompt},
    ]


def resolve_detector_kwargs(detector_type: str) -> dict:
    det = detector_type.lower()
    kwargs: dict = {}
    if det == "lettucedetect":
        kwargs["confidence_threshold"] = DETECTOR_BEST_THRESHOLDS["lettucedetect"]
        kwargs["model_path"] = LETTUCEDETECT_MODEL_PATH
    elif det == "lettuceprevent":
        kwargs["confidence_threshold"] = DETECTOR_BEST_THRESHOLDS["lettuceprevent"]
        kwargs["model_path"] = LETTUCEPREVENT_MODEL_PATH
    return kwargs


# ===========================================================================
# Single-cell execution
# ===========================================================================

def run_one_cell(
    generator_model: str,
    detector_type: str,
    output_prefix: str,
    n_per_task: int,
    confidence_floor_post_eval: float,
    skip_threshold: float,
    use_wandb: bool = True,
):
    run = None
    if use_wandb:
        run_name = (
            f"{generator_model.split('/')[-1]}__"
            f"{detector_type}__skip_{skip_threshold:.2f}"
        )
        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=run_name,
            config={
                "generator_model":            generator_model,
                "detector_type":              detector_type,
                "seed":                       SEED,
                "generation_kwargs":          GENERATION_CONFIG_KWARGS,
                "num_beams":                  NUM_BEAMS,
                "last_k_tokens_to_consider":  LAST_K_TOKENS_TO_CONSIDER,
                "top_k_logits":               TOP_K_LOGITS,
                "penalty_value":              str(PENALTY_VALUE),
                "use_all_tokens":             USE_ALL_TOKENS,
                "logits_skip_threshold":      skip_threshold,
                "lettucedetect_model_path":   LETTUCEDETECT_MODEL_PATH,
                "lettuceprevent_model_path":  LETTUCEPREVENT_MODEL_PATH,
                "detector_thresholds":        DETECTOR_BEST_THRESHOLDS,
                "post_eval_confidence_floor": confidence_floor_post_eval,
                "n_per_task":                 n_per_task,
                "debug_print_to_console":     DEBUG_PRINT_TO_CONSOLE,
            },
            reinit=True,
        )

    make_deterministic(SEED)

    prompts, source_label = load_prompts_for_detector(
        detector_type=detector_type,
        local_summary_filepath=LOCAL_SUMMARY_FILE,
        n_per_task=n_per_task,
    )
    print(f"\n>>> Generator:      {generator_model}")
    print(f">>> Detector:       {detector_type}")
    print(f">>> Skip threshold: {skip_threshold}")
    print(f">>> Source:         {source_label}")
    print(f">>> Prompts:        {len(prompts)}")
    print(f">>> Num beams:      {NUM_BEAMS}")
    print(f">>> Debug print:    {DEBUG_PRINT_TO_CONSOLE}\n")

    print(f"Loading generator tokenizer + model: {generator_model}")
    tokenizer = AutoTokenizer.from_pretrained(generator_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        generator_model,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    gen_config = GenerationConfig(
        **GENERATION_CONFIG_KWARGS,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    detector_kwargs = resolve_detector_kwargs(detector_type)
    results = []
    total_modifications = 0
    total_skips = 0
    total_checks = 0
    overall_start = time.time()
    overall_start_dt = datetime.now()

    for prompt_idx, item in enumerate(tqdm(prompts)):
        prompt_start = time.time()
        prompt_start_dt = datetime.now()
        raw_prompt = item["prompt"]
        task_type = item.get("task_type", "Summary")

        logits_processor = None
        detector = DetectorFactory.create_detector(
            detector_type=detector_type,
            tokenizer=tokenizer,
            input_text=raw_prompt,
            **detector_kwargs,
        )
        if detector is not None:
            logits_processor = HallucinationLogitsProcessor(
                hallucination_detector=detector,
                last_k_tokens_to_consider=LAST_K_TOKENS_TO_CONSIDER,
                top_k_logits=TOP_K_LOGITS,
                penalty_value=PENALTY_VALUE,
                use_all_tokens=USE_ALL_TOKENS,
                skip_threshold=skip_threshold,
                debug_print=DEBUG_PRINT_TO_CONSOLE,
            )

        messages = build_messages_for_generator(generator_model, raw_prompt)
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        input_data = tokenizer(
            formatted_prompt, return_tensors="pt",
            padding=True, truncation=False,
        )
        input_data = {k: v.to(model.device) for k, v in input_data.items()}

        torch.manual_seed(SEED + prompt_idx)

        gen_kwargs = dict(
            **input_data,
            generation_config=gen_config,
        )
        if DEBUG_PRINT_TO_CONSOLE:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [TokenPrintStoppingCriteria(tokenizer)]
            )
        if logits_processor is not None:
            gen_kwargs["logits_processor"] = LogitsProcessorList([logits_processor])

        output = model.generate(**gen_kwargs)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        if "[/INST]" in decoded:
            answer_only = decoded.split("[/INST]", 1)[-1].strip()
        elif "<|im_start|>assistant" in decoded:
            answer_only = decoded.split("<|im_start|>assistant", 1)[-1].strip()
        else:
            answer_only = decoded.strip()

        prompt_end = time.time()
        prompt_dur = round(prompt_end - prompt_start, 2)

        result_data = {
            "prompt":            raw_prompt,
            "query":             item.get("query"),
            "answer":            answer_only,
            "task_type":         task_type,
            "dataset":           source_label,
            "language":          "en",
            "start_time":        prompt_start_dt.isoformat(),
            "end_time":          datetime.now().isoformat(),
            "duration_seconds":  prompt_dur,
            "detector_type":     detector_type,
            "generator_model":   generator_model,
            "skip_threshold":    skip_threshold,
        }

        if detector_type == "lettucedetect":
            result_data["logits_modifications"] = logits_processor.modifications_count
            result_data["confidence_threshold"] = detector_kwargs.get("confidence_threshold")
            result_data["lettucedetect_model_path"] = LETTUCEDETECT_MODEL_PATH
        elif detector_type == "lettuceprevent":
            result_data["logits_modifications"] = logits_processor.modifications_count
            result_data["confidence_threshold"] = detector_kwargs.get("confidence_threshold")
            result_data["model_path"] = LETTUCEPREVENT_MODEL_PATH
        elif detector_type == "number":
            result_data["logits_modifications"] = logits_processor.modifications_count
            result_data["allowed_numbers"] = list(detector.allowed_numbers)
        elif detector_type in ("baseline-run-numbers", "baseline-run-facts"):
            result_data["logits_modifications"] = 0
            result_data["comparison_experiment"] = True

        if logits_processor is not None:
            result_data["hdm_skip_count"]  = logits_processor.skip_count
            result_data["hdm_check_count"] = logits_processor.check_count
            total_modifications += logits_processor.modifications_count
            total_skips          += logits_processor.skip_count
            total_checks         += logits_processor.check_count

        results.append(result_data)

        if run is not None:
            run.log({
                "prompt_index":         prompt_idx,
                "task_type":            task_type,
                "duration_seconds":     prompt_dur,
                "logits_modifications": result_data.get("logits_modifications", 0),
                "hdm_skip_count":       result_data.get("hdm_skip_count", 0),
                "hdm_check_count":      result_data.get("hdm_check_count", 0),
            })

    overall_end = time.time()
    total_dur = round(overall_end - overall_start, 2)

    results.append({
        "_meta": {
            "generator_model":   generator_model,
            "detector_type":     detector_type,
            "skip_threshold":    skip_threshold,
            "num_prompts":       len(prompts),
            "total_generations": len(prompts),
            "seed":              SEED,
            "start_time":        overall_start_dt.isoformat(),
            "end_time":          datetime.now().isoformat(),
            "duration_seconds":  total_dur,
            "generation_config": gen_config.to_dict(),
            "detector_config": {
                "detector_type":              detector_type,
                "last_k_tokens_to_consider":  LAST_K_TOKENS_TO_CONSIDER,
                "top_k_logits":               TOP_K_LOGITS,
                "penalty_value":              str(PENALTY_VALUE),
                "use_all_tokens":             USE_ALL_TOKENS,
                "logits_skip_threshold":      skip_threshold,
                "confidence_threshold":       detector_kwargs.get("confidence_threshold"),
                "lettucedetect_model_path":   LETTUCEDETECT_MODEL_PATH if detector_type == "lettucedetect" else None,
                "lettuceprevent_model_path":  LETTUCEPREVENT_MODEL_PATH if detector_type == "lettuceprevent" else None,
            },
            "total_logits_modifications": total_modifications,
            "total_hdm_skips":            total_skips,
            "total_hdm_checks":           total_checks,
        }
    })

    timestamp = overall_start_dt.strftime("%Y%m%d_%H%M%S")
    safe_gen   = generator_model.replace("/", "_")
    skip_label = f"skip_{skip_threshold:.2f}".replace(".", "_")
    base_name  = f"{output_prefix}_{safe_gen}_{detector_type}_{skip_label}_{timestamp}"
    gen_path   = f"{DATA_DIR}/{base_name}_generations.json"
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(gen_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved generations: {gen_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Post-eval with LettuceDetect-base")
    print("=" * 60)
    items_for_eval = [r for r in results if "_meta" not in r]
    stats, hallucinated = evaluate_generated_answers(
        items=items_for_eval,
        confidence_floor=confidence_floor_post_eval,
    )

    halluc_json_path = f"{DATA_DIR}/{base_name}_hallucinations.json"
    stats_txt_path   = f"{DATA_DIR}/{base_name}_stats.txt"
    write_hallucinations_json(hallucinated, halluc_json_path, confidence_floor_post_eval)
    write_stats_txt(stats, stats_txt_path)
    print(f"Saved post-eval JSON: {halluc_json_path}")
    print(f"Saved post-eval TXT:  {stats_txt_path}")

    if run is not None:
        run.summary["total_generations"]          = len(items_for_eval)
        run.summary["total_logits_modifications"] = total_modifications
        run.summary["total_hdm_skips"]            = total_skips
        run.summary["total_hdm_checks"]           = total_checks
        run.summary["skip_rate"] = (
            total_skips / total_checks if total_checks else 0.0
        )
        run.summary["total_runtime_seconds"]      = total_dur
        run.summary["post_eval_total_hallucinations"]      = stats["total"]["hallucinations"]
        run.summary["post_eval_items_with_hallucinations"] = stats["total"]["items_with_hallucinations"]
        for label, n in stats["total"]["buckets"].items():
            run.summary[f"post_eval_total_bucket_{label}"] = n
        for tt, info in stats["per_task"].items():
            run.summary[f"post_eval_{tt}_items"]                 = info["items"]
            run.summary[f"post_eval_{tt}_items_with_halluc"]     = info["items_with_hallucinations"]
            run.summary[f"post_eval_{tt}_hallucinations"]        = info["hallucinations"]
            for label, n in info["buckets"].items():
                run.summary[f"post_eval_{tt}_bucket_{label}"] = n

        artifact = wandb.Artifact(
            f"{safe_gen}__{detector_type}__{skip_label}__results",
            type="rq_results",
        )
        artifact.add_file(gen_path)
        artifact.add_file(halluc_json_path)
        artifact.add_file(stats_txt_path)
        run.log_artifact(artifact)
        run.finish()

    print("\nFinal summary:")
    print(f"  Generations:          {len(items_for_eval)}")
    print(f"  Total runtime:        {total_dur}s")
    print(f"  Logits modifications: {total_modifications}")
    print(f"  HDM skips:            {total_skips}")
    print(f"  HDM checks:           {total_checks}")
    if total_checks:
        print(f"  Skip rate:            {total_skips / total_checks:.4f}")
    print(f"  Post-eval hallucs:    {stats['total']['hallucinations']} "
          f"in {stats['total']['items_with_hallucinations']} items")


# ===========================================================================
# Sweep entry
# ===========================================================================

def sweep_train_fn():
    run = wandb.init()
    cfg = wandb.config
    run_one_cell(
        generator_model            = cfg.generator_model,
        detector_type              = cfg.detector_type,
        skip_threshold             = float(cfg.skip_threshold),
        output_prefix              = cfg.get("output_prefix", "rq1"),
        n_per_task                 = int(cfg.get("n_per_task", N_PER_TASK)),
        confidence_floor_post_eval = float(cfg.get("post_eval_floor", 0.70)),
        use_wandb                  = False,
    )
    if run is not None and not run._is_finished:
        run.finish()


def build_sweep_config(
    output_prefix: str, n_per_task: int, post_eval_floor: float,
) -> dict:
    return {
        "name":   "rq-generator-detector-skip-sweep",
        "method": "grid",
        "metric": {"name": "post_eval_total_hallucinations", "goal": "minimize"},
        "parameters": {
            "generator_model":  {"values": GENERATOR_MODELS},
            "detector_type":    {"values": DETECTOR_TYPES_SWEEPED},
            "skip_threshold":   {"values": SKIP_THRESHOLDS},
            "output_prefix":    {"value":  output_prefix},
            "n_per_task":       {"value":  n_per_task},
            "post_eval_floor":  {"value":  post_eval_floor},
        },
    }


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="RQ1 / RQ2 main entry point.")
    parser.add_argument(
        "--mode",
        choices=["sweep-create", "sweep-agent", "single"],
        default="single",
    )
    parser.add_argument("--sweep-id", type=str, default=None)
    parser.add_argument(
        "--count", type=int,
        default=(
            len(GENERATOR_MODELS)
            * len(DETECTOR_TYPES_SWEEPED)
            * len(SKIP_THRESHOLDS)
        ),
    )
    parser.add_argument("--entity",  type=str, default=WANDB_ENTITY)
    parser.add_argument("--project", type=str, default=WANDB_PROJECT)

    parser.add_argument("--generator-model", type=str,
                        default=GENERATOR_MODELS[0])
    parser.add_argument("--detector-type",   type=str,
                        default="lettucedetect",
                        choices=sorted(VALID_DETECTOR_TYPES))
    parser.add_argument("--skip-threshold", type=float, default=1.0)
    parser.add_argument("--n-per-task", type=int, default=N_PER_TASK)
    parser.add_argument("--post-eval-floor", type=float, default=0.70)
    parser.add_argument("--output-prefix", type=str, default="rq1")
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args()

    if args.mode == "sweep-create":
        cfg = build_sweep_config(
            output_prefix=args.output_prefix,
            n_per_task=args.n_per_task,
            post_eval_floor=args.post_eval_floor,
        )
        sweep_id = wandb.sweep(sweep=cfg, entity=args.entity, project=args.project)
        print(f"Created sweep_id: {sweep_id}")
        return

    if args.mode == "sweep-agent":
        if not args.sweep_id:
            cfg = build_sweep_config(
                output_prefix=args.output_prefix,
                n_per_task=args.n_per_task,
                post_eval_floor=args.post_eval_floor,
            )
            sweep_id = wandb.sweep(sweep=cfg, entity=args.entity, project=args.project)
            print(f"Created sweep_id (agent will run it): {sweep_id}")
        else:
            sweep_id = args.sweep_id

        wandb.agent(
            sweep_id=sweep_id, function=sweep_train_fn,
            entity=args.entity, project=args.project, count=args.count,
        )
        return

    run_one_cell(
        generator_model            = args.generator_model,
        detector_type              = args.detector_type,
        skip_threshold             = args.skip_threshold,
        output_prefix              = args.output_prefix,
        n_per_task                 = args.n_per_task,
        confidence_floor_post_eval = args.post_eval_floor,
        use_wandb                  = not args.no_wandb,
    )


if __name__ == "__main__":
    main()