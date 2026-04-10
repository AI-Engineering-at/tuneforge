"""Legal text dataset loader for DSGVO/EU AI Act fine-tuning."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


@dataclass
class LegalDataConfig:
    sources: list[str] = field(
        default_factory=lambda: [
            "openlegaldata",
            "eurlex_dsgvo",
            "eurlex_aiact",
        ]
    )
    language: str = "de"
    output_dir: Path = field(default_factory=lambda: Path("datasets/generated/legal"))
    max_examples: int = 10000
    train_split: float = 0.9


class LegalDataLoader:
    def __init__(self, config: LegalDataConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_json(self, url: str, params: dict[str, object] | None = None) -> dict:
        if params:
            url = f"{url}?{urlencode(params)}"
        request = Request(
            url,
            headers={"User-Agent": "TuneForge/0.2.0 (+https://github.com/AI-Engineerings-at/tuneforge)"},
        )
        with urlopen(request, timeout=30) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return json.loads(response.read().decode(charset))

    def download_openlegaldata(self) -> list[dict]:
        """Download from openlegaldata.io API."""
        examples = []
        page = 1
        while len(examples) < self.config.max_examples:
            data = self._fetch_json(
                "https://de.openlegaldata.io/api/cases/",
                params={"page": page, "page_size": 100},
            )
            for case in data.get("results", []):
                if case.get("content"):
                    examples.append(self._format_case(case))
                    if len(examples) >= self.config.max_examples:
                        break
            if not data.get("next"):
                break
            page += 1
            logger.info(f"Downloaded {len(examples)} cases...")
        return examples

    def _format_case(self, case: dict) -> dict:
        return {
            "instruction": ("Analysiere folgendes Gerichtsurteil und fasse die Kernaussage zusammen."),
            "input": case.get("content", "")[:4000],
            "output": case.get("abstract", case.get("content", "")[:500]),
        }

    def _format_qa_pair(self, context: str, question: str, answer: str) -> dict:
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
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"Generate 5 German Q&A training pairs about "
                                f"DSGVO Article {art}. Each pair: a practical "
                                f"question a company might ask, and a detailed "
                                f"answer citing the article. Return JSON array "
                                f'with "instruction", "input" (article text), '
                                f'"output" (answer) fields.'
                            ),
                        }
                    ],
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

    def save(self, examples: list[dict], name: str) -> Path:
        output = self.config.output_dir / f"{name}.jsonl"
        with open(output, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(examples)} examples to {output}")
        return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TuneForge legal dataset helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser(
        "download-openlegaldata",
        help="Download German court decisions from Open Legal Data",
    )
    download.add_argument("--output-dir", default="datasets/generated/legal")
    download.add_argument("--max-examples", type=int, default=10000)
    download.add_argument("--output-name", default="openlegaldata")
    download.add_argument("--language", default="de")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    config = LegalDataConfig(
        output_dir=Path(args.output_dir),
        max_examples=args.max_examples,
        language=args.language,
    )
    loader = LegalDataLoader(config)

    if args.command == "download-openlegaldata":
        examples = loader.download_openlegaldata()
        output_path = loader.save(examples, args.output_name)
        print(
            json.dumps(
                {
                    "source": "openlegaldata",
                    "example_count": len(examples),
                    "output_path": str(output_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
