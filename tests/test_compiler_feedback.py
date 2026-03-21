"""Tests for IEC 61131-3 Structured Text validator."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.compiler_feedback import STValidator, ValidationResult


def test_basic_st_validation():
    code = """FUNCTION_BLOCK FB_PID
VAR_INPUT
    rSetpoint : REAL;
    rActual : REAL;
END_VAR
VAR_OUTPUT
    rOutput : REAL;
END_VAR
END_FUNCTION_BLOCK"""
    result = STValidator.validate_syntax(code)
    assert result.valid is True


def test_invalid_st_missing_end():
    code = """FUNCTION_BLOCK FB_Test
VAR_INPUT
    x : INT;
END_VAR"""
    result = STValidator.validate_syntax(code)
    assert result.valid is False
    assert "END_FUNCTION_BLOCK" in result.error


def test_missing_end_var():
    code = """FUNCTION_BLOCK FB_Test
VAR_INPUT
    x : INT;
END_FUNCTION_BLOCK"""
    result = STValidator.validate_syntax(code)
    assert result.valid is False
    assert "END_VAR" in result.error


def test_valid_program():
    code = """PROGRAM Main
VAR
    counter : INT := 0;
END_VAR
    counter := counter + 1;
END_PROGRAM"""
    result = STValidator.validate_syntax(code)
    assert result.valid is True


def test_semicolon_warning():
    code = """FUNCTION_BLOCK FB_Test
VAR
    x : INT;
END_VAR
    x := 42
END_FUNCTION_BLOCK"""
    result = STValidator.validate_syntax(code)
    assert result.valid is True
    assert len(result.warnings) >= 1
    assert "semicolon" in result.warnings[0].lower()


def test_nested_if():
    code = """FUNCTION_BLOCK FB_Test
VAR
    x : INT;
END_VAR
    IF x > 0 THEN
        IF x > 10 THEN
            x := 10;
        END_IF
    END_IF
END_FUNCTION_BLOCK"""
    result = STValidator.validate_syntax(code)
    assert result.valid is True


def test_validation_result_defaults():
    r = ValidationResult(valid=True)
    assert r.warnings == []
    assert r.error == ""
