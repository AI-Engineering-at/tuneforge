"""Structured Text syntax validation for training data quality.

Basic IEC 61131-3 syntax checker — validates block pairing, VAR declarations,
and common mistakes. Used to filter synthetic training data.
"""

import re
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    valid: bool
    error: str = ""
    warnings: list[str] = field(default_factory=list)


class STValidator:
    """Basic IEC 61131-3 Structured Text syntax validator."""

    BLOCK_PAIRS = [
        ("FUNCTION_BLOCK", "END_FUNCTION_BLOCK"),
        ("FUNCTION", "END_FUNCTION"),
        ("PROGRAM", "END_PROGRAM"),
        ("IF", "END_IF"),
        ("FOR", "END_FOR"),
        ("WHILE", "END_WHILE"),
        ("CASE", "END_CASE"),
    ]

    VAR_KEYWORDS = ["VAR_INPUT", "VAR_OUTPUT", "VAR_IN_OUT", "VAR"]

    VALID_TYPES = [
        "BOOL",
        "INT",
        "DINT",
        "REAL",
        "LREAL",
        "STRING",
        "TIME",
        "BYTE",
        "WORD",
        "DWORD",
        "UINT",
        "UDINT",
        "SINT",
        "USINT",
        "DATE",
        "TOD",
        "DT",
        "ARRAY",
    ]

    @staticmethod
    def validate_syntax(code: str) -> ValidationResult:
        warnings = []
        lines = code.strip().split("\n")

        # Check block pairing (non-VAR blocks)
        for open_kw, close_kw in STValidator.BLOCK_PAIRS:
            opens = sum(1 for line in lines if re.search(rf"\b{open_kw}\b", line.strip()))
            closes = sum(1 for line in lines if re.search(rf"\b{close_kw}\b", line.strip()))
            if opens > closes:
                return ValidationResult(
                    valid=False,
                    error=(f"Missing {close_kw} (found {opens} {open_kw} but {closes} {close_kw})"),
                )

        # Check VAR blocks (all share END_VAR)
        var_opens = sum(
            1
            for line in lines
            if re.search(
                r"\bVAR\b|\bVAR_INPUT\b|\bVAR_OUTPUT\b|\bVAR_IN_OUT\b",
                line.strip(),
            )
        )
        var_closes = sum(1 for line in lines if re.search(r"\bEND_VAR\b", line.strip()))
        if var_opens > var_closes:
            return ValidationResult(
                valid=False,
                error=(f"Missing END_VAR (found {var_opens} VAR blocks but {var_closes} END_VAR)"),
            )

        # Check semicolons in assignments
        for i, line in enumerate(lines):
            stripped = line.strip()
            if ":=" in stripped and not stripped.endswith(";"):
                if not stripped.endswith(","):
                    warnings.append(f"Line {i + 1}: Assignment may be missing semicolon")

        return ValidationResult(valid=True, warnings=warnings)
