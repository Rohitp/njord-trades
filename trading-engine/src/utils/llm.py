"""
LLM utilities for parsing and validating responses.

Provides shared functionality for:
- Extracting JSON from LLM responses (handles markdown code blocks)
- Validating responses against Pydantic models
- Logging malformed responses for debugging
"""

import json
from typing import TypeVar, Type

from pydantic import BaseModel, ValidationError

from src.utils.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


def extract_json(response: str) -> str:
    """
    Extract JSON from an LLM response.

    Handles common formats:
    - Plain JSON
    - JSON wrapped in ```json ... ``` code blocks
    - JSON wrapped in ``` ... ``` code blocks

    Args:
        response: Raw LLM response text

    Returns:
        Extracted JSON string (not parsed)

    Raises:
        ValueError: If no JSON-like content found
    """
    if not response or not response.strip():
        raise ValueError("Empty response from LLM")

    text = response.strip()

    # Try to extract from markdown code blocks
    if "```json" in text:
        try:
            json_str = text.split("```json")[1].split("```")[0]
            return json_str.strip()
        except IndexError:
            pass  # Fall through to other methods

    if "```" in text:
        try:
            json_str = text.split("```")[1].split("```")[0]
            return json_str.strip()
        except IndexError:
            pass  # Fall through

    # Assume the whole response is JSON
    return text


def parse_json(response: str, context: str = "LLM response") -> list | dict:
    """
    Parse JSON from an LLM response.

    Args:
        response: Raw LLM response text
        context: Description for error messages (e.g., "DataAgent response")

    Returns:
        Parsed JSON as dict or list

    Raises:
        ValueError: If JSON extraction or parsing fails
    """
    try:
        json_str = extract_json(response)
        parsed = json.loads(json_str)
        return parsed
    except json.JSONDecodeError as e:
        log.warning(
            "json_parse_failed",
            context=context,
            error=str(e),
            response_preview=response[:500] if response else None,
        )
        raise ValueError(f"Failed to parse JSON from {context}: {e}") from e
    except ValueError as e:
        log.warning(
            "json_extraction_failed",
            context=context,
            error=str(e),
        )
        raise


def parse_json_list(response: str, context: str = "LLM response") -> list:
    """
    Parse JSON array from an LLM response.

    If the response is a single object, wraps it in a list.

    Args:
        response: Raw LLM response text
        context: Description for error messages

    Returns:
        List of parsed JSON objects

    Raises:
        ValueError: If parsing fails
    """
    parsed = parse_json(response, context)

    if isinstance(parsed, dict):
        return [parsed]
    elif isinstance(parsed, list):
        return parsed
    else:
        raise ValueError(f"Expected JSON array or object from {context}, got {type(parsed).__name__}")


def validate_and_parse(
    response: str,
    model: Type[T],
    context: str = "LLM response",
) -> T:
    """
    Parse JSON and validate against a Pydantic model.

    Args:
        response: Raw LLM response text
        model: Pydantic model class to validate against
        context: Description for error messages

    Returns:
        Validated Pydantic model instance

    Raises:
        ValueError: If parsing or validation fails
    """
    parsed = parse_json(response, context)

    try:
        return model.model_validate(parsed)
    except ValidationError as e:
        log.warning(
            "validation_failed",
            context=context,
            model=model.__name__,
            errors=e.errors(),
        )
        raise ValueError(f"Validation failed for {context}: {e}") from e


def validate_and_parse_list(
    response: str,
    model: Type[T],
    context: str = "LLM response",
) -> list[T]:
    """
    Parse JSON array and validate each item against a Pydantic model.

    Args:
        response: Raw LLM response text
        model: Pydantic model class to validate each item against
        context: Description for error messages

    Returns:
        List of validated Pydantic model instances

    Raises:
        ValueError: If parsing or validation fails
    """
    parsed_list = parse_json_list(response, context)
    results = []

    for i, item in enumerate(parsed_list):
        try:
            validated = model.model_validate(item)
            results.append(validated)
        except ValidationError as e:
            log.warning(
                "item_validation_failed",
                context=context,
                model=model.__name__,
                index=i,
                errors=e.errors(),
            )
            # Continue processing other items, log the failure
            continue

    return results
