"""CPU-only, reference-blind answer extraction and separate baseline rescoring.

The original strict metric is preserved. The prose metric is exploratory because
it was introduced after inspecting the first baseline run, not before that run.
No model, tokenizer, or GPU is needed to apply either checker to saved responses.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
from collections.abc import Sequence
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

PROSE_CHECKER_VERSION = "clear_final_numeric_answer_v2"
NUMBER_PATTERN = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
ANSWER_CUE = re.compile(r"\b(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*:?|[:=])\s*", re.I)
ANSWER_HEADER = re.compile(r"(?:the\s+)?(?:final\s+)?answer\s*(?::|is\s*:?)", re.I)
CONCLUSION_CUE = re.compile(
    r"^(?:(?:so|therefore|thus|hence|in conclusion)\b[, :]*|"
    r"the\s+(?:total|number|amount|result|cost|difference)\b)",
    re.I,
)
UNCERTAIN = re.compile(
    r"\b(?:maybe|perhaps|might|could|probably|possibly|approximately|about|"
    r"guess|guessing|unsure|uncertain|wrong|incorrect|between)\b|"
    r"\b(?:at least|at most|not sure)\b",
    re.I,
)


def extract_strict_final_answer(response_text: str) -> Decimal | None:
    """Apply the original notebook's single-####-final-line rule unchanged."""
    lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    marked_lines = [line for line in lines if line.startswith("####")]
    if not lines or len(marked_lines) != 1:
        return None
    match = re.fullmatch(r"####\s*\$?\s*(" + NUMBER_PATTERN + r")", lines[-1])
    return Decimal(match.group(1).replace(",", "")) if match else None


def _clean_final_line(line: str) -> str:
    """Remove display markup, preserving decimal points and arithmetic operators."""
    line = line.strip().replace(r"\$", "$")
    line = re.sub(r"(?<!\w)\*\*(\S(?:.*?\S)?)\*\*(?!\w)", r"\1", line)
    line = re.sub(r"`([^`\n]+)`", r"\1", line)
    line = re.sub(r"^(?:\d+[.)]\s+|[-*]\s+)", "", line)
    return line.strip()


def _read_number_and_units(text: str) -> Decimal | None:
    """Read a number followed only by short unit text, never a second calculation."""
    text = text.strip().rstrip(".!").strip()
    match = re.fullmatch(
        r"\$?\s*("
        + NUMBER_PATTERN
        + r")(?P<units>(?:\s+[A-Za-z]+(?:[-/][A-Za-z]+)?){0,5}|\s*%)",
        text,
    )
    if match is None:
        return None
    # A suffix such as 'or 7', 'is wrong', or 'but maybe' is not a unit.
    units = match.group("units")
    if re.search(
        r"\b(?:or|and|but|if|unless|because|is|are|was|were|not)\b", units, re.I
    ):
        return None
    if UNCERTAIN.search(text):
        return None
    return Decimal(match.group(1).replace(",", ""))


def _conclusion_candidate(line: str) -> tuple[Decimal | None, str | None]:
    """Recognize a declared conclusion; never search for an arbitrary last number."""
    if (
        UNCERTAIN.search(line)
        or "?" in line
        or re.search(
            r"\b(?:not|neither|nor|isn't|aren't|wasn't|weren't|cannot|can't|won't|doesn't|don't|didn't)\b",
            line,
            re.I,
        )
    ):
        return None, None
    cues = list(ANSWER_CUE.finditer(line))
    if len(cues) == 1:
        value = _read_number_and_units(line[cues[0].end() :])
        return value, "explicit_answer" if value is not None else None
    if not CONCLUSION_CUE.search(line):
        return None, None

    # A concluding sentence with exactly one number can state an answer in prose.
    # Consume a whole comma/decimal token so malformed '1,23' cannot become 23.
    numbers = list(re.finditer(r"\$?[+-]?\d[\d,]*(?:\.\d+)?", line))
    if len(numbers) == 1:
        number = numbers[0]
        value = _read_number_and_units(number.group())
        suffix = line[number.end() :]
        # Prose after the one number may describe its units or purpose. It must
        # contain no new calculations, alternatives, or conditional qualifications.
        if not re.fullmatch(
            r"(?:\s+[A-Za-z]+(?:[-/][A-Za-z]+)?)*(?:\s*%)?[.!]?", suffix
        ) or re.search(
            r"\b(?:or|but|if|unless|because)\b",
            suffix,
            re.I,
        ):
            return None, None
        return value, "concluding_sentence" if value is not None else None

    # Also accept a declared conclusion ending in a simple arithmetic equality,
    # e.g. 'The total is 3 + 6 + 20 = 29 liters'. We read the claimed result;
    # the reference comparison below, not expression execution, decides correctness.
    equation = re.search(r"\b(?:is|equals)\s+([\d,.+*/()\s-]+)\s*=\s*(.+)$", line, re.I)
    if equation and re.search(r"\d", equation.group(1)):
        value = _read_number_and_units(equation.group(2))
        return value, "concluding_equation" if value is not None else None
    return None, None


def extract_prose_final_answer(
    response_text: str, *, hit_token_limit: bool = False
) -> dict:
    """Return an auditable extraction using only response text and stopping status.

    This function deliberately has no reference-answer argument. Its result must
    not depend on which candidate happens to match the correct answer.
    """

    def result(
        answer=None, method="unparsed", reason="no_clear_final_answer", evidence=""
    ):
        return {
            "answer": str(answer) if answer is not None else None,
            "method": method,
            "reason": reason,
            "evidence": evidence,
        }

    # Conservative policy: an unfinished token-budget stop is not a final answer,
    # even when a correct number occurred somewhere in the reasoning.
    if hit_token_limit:
        return result(reason="token_limit_stop")
    lines = [
        _clean_final_line(line) for line in response_text.splitlines() if line.strip()
    ]
    if not lines:
        return result(reason="empty_response")
    strict_answer = extract_strict_final_answer(response_text)
    if strict_answer is None and any(line.startswith("####") for line in lines):
        return result(reason="invalid_or_ambiguous_markers", evidence=lines[-1])

    # Conflicting explicitly declared conclusions require human review. Do not
    # choose whichever answer matches the reference or simply take the last one.
    declarations = [strict_answer] if strict_answer is not None else []
    for index, line in enumerate(lines):
        value, method = _conclusion_candidate(line)
        if value is None and ANSWER_HEADER.fullmatch(line) and index + 1 < len(lines):
            value = _read_number_and_units(lines[index + 1])
            if value is None:
                value, _ = _conclusion_candidate("Therefore, " + lines[index + 1])
        if value is not None:
            declarations.append(value)
    if len(set(declarations)) > 1:
        return result(reason="conflicting_conclusions", evidence=lines[-1])
    if strict_answer is not None:
        return result(strict_answer, "strict_marker", "accepted", lines[-1])

    final_line = lines[-1]
    # A standalone 'Final answer:' header followed by a number is explicit too.
    if len(lines) >= 2 and ANSWER_HEADER.fullmatch(lines[-2]):
        answer = _read_number_and_units(final_line)
        if answer is None:
            # The header can introduce prose too, e.g. 'The shop has 8 books.'
            # Use the same conservative one-number rules as a concluding sentence.
            answer, _ = _conclusion_candidate("Therefore, " + final_line)
        if answer is not None:
            return result(
                answer, "answer_header", "accepted", lines[-2] + "\n" + final_line
            )
    answer, method = _conclusion_candidate(final_line)
    if answer is not None:
        return result(answer, method, "accepted", final_line)
    return result(evidence=final_line)


def rescore_response_rows(response_rows: list[dict]) -> tuple[list[dict], dict]:
    """Create separate annotations and metrics without mutating original rows."""
    if not response_rows or len({row["example_id"] for row in response_rows}) != len(
        response_rows
    ):
        raise ValueError("Expected nonempty responses with unique example IDs.")
    scored_rows = []
    for row in response_rows:
        strict_answer = extract_strict_final_answer(row["response_text"])
        if type(row.get("hit_token_limit", False)) is not bool:
            raise ValueError("Token-limit status must be a Boolean.")
        extraction = extract_prose_final_answer(
            row["response_text"],
            hit_token_limit=row.get("hit_token_limit", False),
        )
        # Extract first, then consult the reference. Never use it to select a number.
        expected = Decimal(row["expected_answer"])
        if not expected.is_finite():
            raise ValueError("Reference answers must be finite numbers.")
        strict_correct = strict_answer is not None and strict_answer == expected
        if "is_correct" in row and row["is_correct"] != strict_correct:
            raise ValueError(
                "Saved strict score disagrees with its original rule: "
                + row["example_id"]
            )
        if "predicted_answer" in row:
            saved_answer = (
                Decimal(row["predicted_answer"])
                if row["predicted_answer"] is not None
                else None
            )
            if saved_answer != strict_answer:
                raise ValueError(
                    "Saved strict extraction disagrees with its original rule: "
                    + row["example_id"]
                )
        if "answer_parsed" in row and row["answer_parsed"] != (
            strict_answer is not None
        ):
            raise ValueError(
                "Saved strict parse status disagrees with its original rule: "
                + row["example_id"]
            )
        scored_rows.append(
            {
                "example_id": row["example_id"],
                "expected_answer": str(expected),
                "strict_answer": str(strict_answer)
                if strict_answer is not None
                else None,
                "strict_correct": strict_correct,
                "prose_answer": extraction["answer"],
                "prose_correct": extraction["answer"] is not None
                and Decimal(extraction["answer"]) == expected,
                "extraction_method": extraction["method"],
                "extraction_reason": extraction["reason"],
                "extraction_evidence": extraction["evidence"],
                "hit_token_limit": row.get("hit_token_limit", False),
                "question": row.get("question", ""),
                "response_text": row["response_text"],
            }
        )
    total = len(scored_rows)
    summaries = {}
    for label in ["strict", "prose"]:
        correct = sum(row[label + "_correct"] for row in scored_rows)
        parsed = sum(row[label + "_answer"] is not None for row in scored_rows)
        summaries[label] = {
            "number_of_examples": total,
            "number_correct": correct,
            "number_parsed": parsed,
            "accuracy": correct / total,
            "parse_rate": parsed / total,
        }
    summaries["prose"]["checker_version"] = PROSE_CHECKER_VERSION
    summaries["prose"]["status"] = "exploratory_post_hoc"
    summaries["prose"]["number_token_limit_stops"] = sum(
        row["hit_token_limit"] for row in scored_rows
    )
    return scored_rows, summaries


def categorical_kl(
    probabilities: Sequence[float], reference_probabilities: Sequence[float]
) -> float:
    """Compute KL(p || q) in nats for a small, explicitly normalized distribution.

    This transparent CPU reference is for the notebook's mathematical sanity
    check. It is not the large-vocabulary, differentiable training implementation.
    Zero p contributes zero; positive p with zero q gives positive infinity.
    """
    p, q = tuple(probabilities), tuple(reference_probabilities)
    if not p or len(p) != len(q):
        raise ValueError("Distributions must have equal, nonzero lengths.")
    for distribution in (p, q):
        if any(not math.isfinite(value) or value < 0 for value in distribution):
            raise ValueError("Probabilities must be finite and nonnegative.")
        if not math.isclose(math.fsum(distribution), 1.0, abs_tol=1e-12, rel_tol=0):
            raise ValueError("Each probability distribution must sum to one.")
    if any(left > 0 and right == 0 for left, right in zip(p, q, strict=True)):
        return math.inf
    return math.fsum(
        left * (math.log(left) - math.log(right))
        for left, right in zip(p, q, strict=True)
        if left > 0
    )


def load_saved_baseline(input_directory: str | Path) -> tuple[list[dict], dict]:
    """Read the immutable source run and verify its manifest and record alignment."""
    directory = Path(input_directory)
    manifest_bytes = (directory / "manifest.json").read_bytes()
    response_bytes = (directory / "responses.jsonl").read_bytes()
    manifest = json.loads(manifest_bytes)
    protocol = manifest["protocol"]
    digest = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    if digest != manifest["protocol_sha256"]:
        raise ValueError("Source protocol fingerprint does not match its manifest.")
    rows = [
        json.loads(line)
        for line in response_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != protocol["number_of_examples"]:
        raise ValueError("Source response count does not match its manifest.")
    if [row["example_id"] for row in rows] != [
        example["example_id"] for example in protocol["examples"]
    ]:
        raise ValueError("Source response IDs do not match the selected examples.")
    for row, example in zip(rows, protocol["examples"]):
        if any(row.get(key) != value for key, value in example.items()):
            raise ValueError("Source example changed: " + row["example_id"])
    source = {
        "original_run_id": manifest["run_id"],
        "original_protocol_sha256": digest,
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_responses_sha256": hashlib.sha256(response_bytes).hexdigest(),
    }
    return rows, source


def save_rescore_report(
    output_directory: str | Path, scored_rows: list[dict], summaries: dict, source: dict
) -> Path:
    """Write a new, explicitly exploratory report; never overwrite a source run."""
    output = Path(output_directory)
    # Existing directories, including the original downloaded run, are rejected.
    output.mkdir(parents=True, exist_ok=False)
    report_manifest = {
        "report_type": "exploratory_rescoring",
        "checker_version": PROSE_CHECKER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "note": "Checker designed after inspecting baseline responses. This is not a preregistered test score.",
        "truncation_policy": "Exclude token-limit stops from prose extraction; retain all rows in the denominator.",
        "checker_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    (output / "rescore_manifest.json").write_text(
        json.dumps(report_manifest, indent=2), encoding="utf-8"
    )
    # Archive the exact reference-blind checker, not just a mutable version label.
    (output / "checker_source.py").write_bytes(Path(__file__).read_bytes())
    (output / "rescore_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    with (output / "rescore_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in scored_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "rescore_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored_rows[0]))
        writer.writeheader()
        writer.writerows(scored_rows)
    restored = [
        json.loads(line)
        for line in (output / "rescore_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert restored == scored_rows
    archive = Path(shutil.make_archive(str(output), "zip", root_dir=output))
    return archive


def main() -> None:
    """Rescore a downloaded baseline on CPU: python -m src.evaluation RUN_FOLDER."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    rows, source = load_saved_baseline(args.input_directory)
    scored_rows, summaries = rescore_response_rows(rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_directory or Path("results") / (
        source["original_run_id"] + "_prose_" + stamp
    )
    archive = save_rescore_report(output, scored_rows, summaries, source)
    print(json.dumps(summaries, indent=2))
    print("Exploratory report:", output.resolve())
    print("Report ZIP:", archive.resolve())
    print("Original downloaded files were not modified. No model or GPU was used.")


if __name__ == "__main__":
    main()
