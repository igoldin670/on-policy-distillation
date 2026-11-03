"""Frozen, disjoint data selections for the Notebook 04 controlled experiment."""

from __future__ import annotations

import random

from src.evaluation import extract_strict_final_answer
from src.sft import encode_example, normalize_question


def prepare_opd_splits(
    source_train,
    source_test,
    tokenizer,
    protocol,
    cold_splits,
    *,
    train_examples=1024,
    test_examples=256,
    seed=42,
    max_prompt_length=512,
    max_sequence_length=1024,
):
    """Choose fresh train prompts and a previously unused test subset.

    All official test question text is blocked from practice. The old 03 train
    and validation questions are also blocked from new training. Filtering is
    deterministic and records every exclusion; no response is silently truncated.
    """
    cold_train = cold_splits["train"]
    validation = cold_splits["validation"]
    historical_test = protocol["examples"]
    old_questions = {
        normalize_question(row["question"]) for row in cold_train + validation + historical_test
    }
    official_test_questions = {normalize_question(row["question"]) for row in source_test}
    blocked = old_questions | official_test_questions
    seen, practice, exclusions = set(blocked), [], []
    order = list(range(len(source_train)))
    random.Random(seed).shuffle(order)
    for index in order:
        row = source_train[index]
        question_key = normalize_question(row["question"])
        reason = (
            "prior_or_test_overlap"
            if question_key in blocked
            else "duplicate_question"
            if question_key in seen
            else None
        )
        if reason is None:
            try:
                encoded = encode_example(
                    tokenizer, row["question"], row["answer"], protocol, max_sequence_length
                )
                if encoded["prompt_length"] > max_prompt_length:
                    raise ValueError("overlong_prompt")
            except ValueError as error:
                reason = str(error)
        if reason:
            exclusions.append({"split": "train", "source_index": index, "reason": reason})
            continue
        practice.append({"example_id": f"gsm8k/train/{index}", "source_index": index, **encoded})
        seen.add(question_key)
        if len(practice) == train_examples:
            break
    if len(practice) != train_examples:
        raise ValueError("Not enough eligible new training examples.")

    # The primary exam excludes every test example examined in 02/03. Never use
    # its answers for checkpoint selection, loss, or an optimizer update.
    test_order = list(range(len(source_test)))
    random.Random(seed + 1000).shuffle(test_order)
    test_seen = {
        normalize_question(row["question"])
        for row in historical_test + cold_train + validation + practice
    }
    primary_test = []
    for index in test_order:
        row = source_test[index]
        key = normalize_question(row["question"])
        prompt = protocol["prompt_template"].format(question=row["question"])
        reason = "prior_or_duplicate_overlap" if key in test_seen else None
        if (
            reason is None
            and len(tokenizer.encode(prompt, add_special_tokens=False)) > max_prompt_length
        ):
            reason = "overlong_prompt"
        if reason:
            exclusions.append({"split": "test", "source_index": index, "reason": reason})
            continue
        number = extract_strict_final_answer(row["answer"])
        if number is None:
            raise ValueError("Invalid reference in the pinned test source.")
        primary_test.append(
            {
                "example_id": f"gsm8k/test/{index}",
                "source_index": index,
                "question": row["question"],
                "prompt": prompt,
                "expected_answer": str(number),
            }
        )
        test_seen.add(key)
        if len(primary_test) == test_examples:
            break
    if len(primary_test) != test_examples:
        raise ValueError("Not enough eligible unseen test questions.")
    groups = {
        "train": practice,
        "validation": validation,
        "primary_test": primary_test,
        "historical_test": historical_test,
    }
    normalized = {
        name: {normalize_question(row["question"]) for row in rows} for name, rows in groups.items()
    }
    names = list(groups)
    for index, left in enumerate(names):
        if len(normalized[left]) != len(groups[left]):
            raise ValueError(f"Duplicate questions in {left}.")
        for right in names[index + 1 :]:
            if normalized[left] & normalized[right]:
                raise ValueError(f"Question-text leakage between {left} and {right}.")
    return {
        **groups,
        "exclusions": exclusions,
        "seed": seed,
        "train_source_split": "train",
        "primary_test_source_split": "test",
    }
