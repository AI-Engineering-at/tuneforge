"""Legal text dataset loader for DSGVO/EU AI Act fine-tuning.

Data sources:
- Open Legal Data (openlegaldata.io): 251K German court decisions
- DSGVO full text: via EUR-Lex
- EU AI Act: via EUR-Lex

Usage:
    from datasets.legal_data import LegalDataLoader, LegalDataConfig

    loader = LegalDataLoader(LegalDataConfig())
    cases = loader.download_openlegaldata()
    loader.save(cases, "openlegaldata")
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LegalDataConfig:
    sources: list[str] = field(default_factory=lambda: [
        "openlegaldata",
        "eurlex_dsgvo",
        "eurlex_aiact",
    ])
    language: str = "de"
    output_dir: Path = field(default_factory=lambda: Path("datasets/generated/legal"))
    max_examples: int = 10000
    train_split: float = 0.9


class LegalDataLoader:
    def __init__(self, config: LegalDataConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def download_openlegaldata(self) -> list[dict]:
        """Download from openlegaldata.io API."""
        import requests

        examples = []
        page = 1
        while len(examples) < self.config.max_examples:
            resp = requests.get(
                "https://de.openlegaldata.io/api/cases/",
                params={"page": page, "page_size": 100},
                timeout=30,
            )
            if resp.status_code != 200:
                break
            data = resp.json()
            for case in data.get("results", []):
                if case.get("content"):
                    examples.append(self._format_case(case))
            if not data.get("next"):
                break
            page += 1
            logger.info(f"Downloaded {len(examples)} cases...")
        return examples

    def _format_case(self, case: dict) -> dict:
        return {
            "instruction": (
                "Analysiere folgendes Gerichtsurteil und "
                "fasse die Kernaussage zusammen."
            ),
            "input": case.get("content", "")[:4000],
            "output": case.get("abstract", case.get("content", "")[:500]),
        }

    def _format_qa_pair(self, context: str, question: str,
                        answer: str) -> dict:
        return {
            "instruction": question,
            "input": context,
            "output": answer,
        }

    def generate_dsgvo_qa(self, provider) -> list[dict]:
        """Generate DSGVO Q&A pairs via any LLM provider."""
        examples = []
        for art in range(1, 100):
            try:
                response = provider.chat(
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Generate 5 German Q&A training pairs about "
                            f"DSGVO Article {art}. Each pair: a practical "
                            f"question a company might ask, and a detailed "
                            f"answer citing the article. Return JSON array "
                            f'with "instruction", "input" (article text), '
                            f'"output" (answer) fields.'
                        ),
                    }],
                    max_tokens=4096,
                )
                import re
                json_match = re.search(r"\[.*\]", response, re.DOTALL)
                if json_match:
                    examples.extend(json.loads(json_match.group()))
                logger.info(f"DSGVO Art. {art}: {len(examples)} examples")
            except Exception as e:
                logger.warning(f"Failed DSGVO Art. {art}: {e}")
        return examples

    def save(self, examples: list[dict], name: str):
        output = self.config.output_dir / f"{name}.jsonl"
        with open(output, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(examples)} examples to {output}")
