"""GEPA optimization on a synthetic pair-sum-product math task.

Given a list of integers, add adjacent pairs and multiply the resulting sums.
Ported from the `verifiers` RLVR library to GEPA + LangChain v1.

Run:
    uv run python examples/pair_sum_product.py
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import re

from dotenv import load_dotenv
from gepa import optimize
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_gepa_adapter import (
    LangChainGEPAAdapter,
    enable_verbose_logging,
    last_message_text,
    load_chat_model,
    make_default_proposer,
    make_reflection_lm,
)

load_dotenv()


SEED_SYSTEM_PROMPT = """Add adjacent pairs of numbers, then multiply the results.
If only one pair numbers, just add the numbers

Pairs Example: 1,2
- Add pairs: 1+2=3
- multiply: Assume 1, 3*1=3
- Answer: 3

Example: 3, 5, 2, 4
- Add pairs: 3+5=8, 2+4=6
- Multiply: 8*6=48
- Answer: 48

Now solve the problem below. Put your final answer in <answer> tags. Be very concise"""


def generate_problem(rng: random.Random, difficulty: str) -> dict:
    if difficulty == "easy":
        n = 4
    elif difficulty == "medium":
        n = 6
    else:
        n = 12

    nums = [
        rng.randint(1, 9) if rng.random() < 0.7 else rng.randint(10, 19)
        for _ in range(n)
    ]
    sums = [nums[i] + nums[i + 1] for i in range(0, len(nums), 2)]
    answer = math.prod(sums)

    return {
        "input": f"Numbers: {', '.join(map(str, nums))}",
        "answer": str(answer),
        "additional_context": {"nums": nums, "sums": sums},
    }


def generate_dataset(num_examples: int, difficulty: str, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [generate_problem(rng, difficulty) for _ in range(num_examples)]


def evaluate_response(data: dict, state: dict) -> tuple[float, str]:
    correct_answer = data["answer"]
    nums = data["additional_context"]["nums"]
    sums = data["additional_context"]["sums"]

    response = last_message_text(state)
    match = re.search(r"<answer>\s*(\S+)\s*</answer>", response)
    if not match:
        return 0.0, (
            f"Missing <answer> tags. Could not parse answer. "
            f"The correct answer is {correct_answer}."
        )

    parsed = match.group(1).strip()
    if parsed == correct_answer.strip():
        return 1.0, "Correct."

    pairs_str = ", ".join(
        f"{nums[i]}+{nums[i+1]}={sums[i // 2]}" for i in range(0, len(nums), 2)
    )
    return 0.0, (
        f"Wrong answer: you said {parsed}, correct is {correct_answer}. "
        f"Steps: add pairs [{pairs_str}], then multiply sums to get {correct_answer}."
    )


def rollout(candidate: dict[str, str], example: dict, llm: BaseChatModel) -> dict:
    messages = [
        SystemMessage(content=candidate["system_prompt"]),
        HumanMessage(content=example["input"]),
    ]
    result = llm.invoke(messages)
    if not isinstance(result, AIMessage):
        result = AIMessage(content=getattr(result, "content", str(result)))
    return {"messages": messages + [result]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # dataset
    p.add_argument("--train-size", type=int, default=100)
    p.add_argument("--val-size", type=int, default=50)
    p.add_argument("--test-size", type=int, default=50)
    p.add_argument("--difficulty", default="hard", choices=["easy", "medium", "hard"])
    p.add_argument("--data-seed", type=int, default=42)

    # models — defaults use standard LangChain `init_chat_model` strings.
    # Pass --*-config for a JSON file of init_chat_model kwargs (gateway URL, headers, etc.).
    p.add_argument("--task-model", default="openai:gpt-4o-mini")
    p.add_argument("--task-effort", default=None, choices=["minimal", "low", "medium", "high"], help="Reasoning effort (ignored for non-reasoning models)")
    p.add_argument("--task-model-config", default=None, help="Optional JSON path overriding --task-model")
    p.add_argument("--reflection-model", default="openai:gpt-5-mini")
    p.add_argument("--reflection-effort", default="medium", choices=["minimal", "low", "medium", "high"], help="Reasoning effort (ignored for non-reasoning models)")
    p.add_argument("--reflection-model-config", default=None, help="Optional JSON path overriding --reflection-model")

    # optimizer
    p.add_argument("--max-metric-calls", type=int, default=500)
    p.add_argument("--reflection-minibatch-size", type=int, default=3)
    p.add_argument("--num-threads", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-baseline", action="store_true", help="Skip baseline test-set eval")
    p.add_argument("--skip-optimize", action="store_true", help="Run baseline only; skip optimization")

    # logging
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main():
    args = parse_args()
    enable_verbose_logging(getattr(logging, args.log_level))

    total = args.train_size + args.val_size + args.test_size
    all_data = generate_dataset(num_examples=total, difficulty=args.difficulty, seed=args.data_seed)
    train_set = all_data[: args.train_size]
    val_set = all_data[args.train_size : args.train_size + args.val_size]
    test_set = all_data[args.train_size + args.val_size :]
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    task_llm = load_chat_model(
        args.task_model,
        config_path=args.task_model_config,
        reasoning_effort=args.task_effort,
    )
    reflection_llm = load_chat_model(
        args.reflection_model,
        config_path=args.reflection_model_config,
        reasoning_effort=args.reflection_effort,
    )

    adapter = LangChainGEPAAdapter(
        rollout_fn=lambda candidate, example: rollout(candidate, example, task_llm),
        eval_fn=evaluate_response,
        num_threads=args.num_threads,
        custom_proposer=make_default_proposer(make_reflection_lm(reflection_llm)),
    )

    n = len(test_set)
    baseline_correct = 0
    baseline_acc = 0.0
    if not args.skip_baseline:
        print("\nBaseline evaluation on test set...")
        baseline_batch = adapter.evaluate(
            batch=test_set,
            candidate={"system_prompt": SEED_SYSTEM_PROMPT},
            capture_traces=False,
        )
        baseline_correct = sum(1 for s in baseline_batch.scores if s == 1.0)
        baseline_acc = baseline_correct / n * 100
        print(f"\nBaseline:  {baseline_correct}/{n} ({baseline_acc:.1f}%)")

    if args.skip_optimize:
        return

    result = optimize(
        seed_candidate={"system_prompt": SEED_SYSTEM_PROMPT},
        trainset=train_set,
        valset=val_set,
        adapter=adapter,
        reflection_lm=make_reflection_lm(reflection_llm),
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        candidate_selection_strategy="pareto",
        use_merge=True,
        display_progress_bar=True,
        seed=args.seed,
    )

    print(f"\nBest val score: {result.val_aggregate_scores[result.best_idx]}")
    print("\nOptimized system prompt:")
    print("=" * 80)
    print(result.best_candidate["system_prompt"])
    print("=" * 80)

    print("\nOptimized evaluation on test set...")
    optimized_batch = adapter.evaluate(
        batch=test_set,
        candidate=result.best_candidate,
        capture_traces=False,
    )
    optimized_correct = sum(1 for s in optimized_batch.scores if s == 1.0)
    optimized_acc = optimized_correct / n * 100

    print(f"\nBaseline:  {baseline_correct}/{n} ({baseline_acc:.1f}%)")
    print(f"Optimized: {optimized_correct}/{n} ({optimized_acc:.1f}%)")
    print(f"Delta:     {optimized_acc - baseline_acc:+.1f}%")


if __name__ == "__main__":
    main()
