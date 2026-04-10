"""Generate synthetic IEC 61131-3 training data via LLM.

Uses the multi-provider abstraction — works with Claude, OpenAI, Ollama, etc.

Usage:
    from providers import create_provider
    from data_utils.synthetic_generator import SyntheticGenerator

    provider = create_provider("ollama", model="qwen2.5-coder:7b")
    gen = SyntheticGenerator(provider=provider)
    examples = gen.generate_batch("pid_controller", "basic", count=5)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from data_utils.data_formats import AlpacaExample, to_alpaca_format

logger = logging.getLogger(__name__)

SPS_CATEGORIES = [
    # Basic control
    "pid_controller",
    "on_off_controller",
    "cascade_control",
    "motor_start_stop",
    "star_delta_starter",
    "vfd_control",
    # Process control
    "batch_sequence",
    "tank_level_control",
    "conveyor_belt",
    "sorting_station",
    "pick_and_place",
    "palletizer",
    # Safety
    "emergency_stop",
    "light_curtain",
    "safety_gate",
    "two_hand_control",
    "safe_speed_monitor",
    # Communication
    "modbus_rtu_read",
    "profinet_io",
    "opc_ua_client",
    # Data handling
    "recipe_manager",
    "alarm_handler",
    "data_logger",
    "fifo_buffer",
    "moving_average",
    "timer_manager",
]

COMPLEXITY_LEVELS = ["basic", "intermediate", "advanced"]

IEC_LANGUAGES = [
    "structured_text",
    "ladder_logic_description",
    "function_block_diagram_description",
]


@dataclass
class GenerationTask:
    category: str
    complexity: str
    language: str
    prompt_template: str = (
        "Write a {complexity} PLC program for {category} in {language}. "
        "Follow IEC 61131-3 standard. Include variable declarations, "
        "the implementation, and inline comments explaining the logic."
    )

    def build_prompt(self) -> str:
        return self.prompt_template.format(
            category=self.category.replace("_", " "),
            complexity=self.complexity,
            language=self.language.replace("_", " "),
        )


@dataclass
class SyntheticGenerator:
    """Generate synthetic SPS/PLC training data using any LLM provider."""

    provider: object  # LLMProvider instance
    output_dir: Path = field(default_factory=lambda: Path("datasets/generated/sps"))
    max_tokens: int = 4096

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_batch(
        self, category: str, complexity: str, count: int = 5, language: str = "structured_text"
    ) -> list[dict]:
        """Generate a batch of examples for one category/complexity."""
        examples = []
        for i in range(count):
            task = GenerationTask(
                category=category,
                complexity=complexity,
                language=language,
            )
            prompt = task.build_prompt()

            try:
                response = self.provider.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert PLC/SPS programmer. "
                                "Generate IEC 61131-3 compliant code. "
                                "Always use proper variable declarations."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.8,
                )

                example = AlpacaExample(
                    instruction=prompt,
                    output=response,
                )
                examples.append(to_alpaca_format(example))
                logger.info(f"Generated {category}/{complexity} [{i + 1}/{count}]")
            except Exception as e:
                logger.warning(f"Failed {category}/{complexity} [{i + 1}]: {e}")

        return examples

    def generate_all(self, examples_per_combo: int = 3) -> list[dict]:
        """Generate examples across all categories and complexity levels."""
        all_examples = []
        total = len(SPS_CATEGORIES) * len(COMPLEXITY_LEVELS)
        done = 0

        for category in SPS_CATEGORIES:
            for complexity in COMPLEXITY_LEVELS:
                done += 1
                logger.info(f"[{done}/{total}] Generating {category}/{complexity}")
                batch = self.generate_batch(
                    category,
                    complexity,
                    count=examples_per_combo,
                )
                all_examples.extend(batch)

        self._save(all_examples)
        logger.info(f"Generated {len(all_examples)} total examples")
        return all_examples

    def _save(self, examples: list[dict]):
        """Save generated examples to JSONL."""
        output_file = self.output_dir / "sps_synthetic.jsonl"
        with open(output_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.info(f"Saved to {output_file}")
