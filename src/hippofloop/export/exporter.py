"""GGUF export — merge LoRA adapter and quantize."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SUPPORTED_QUANTIZATIONS = frozenset({"Q4_K_M", "Q5_K_M", "Q8_0"})


class GgufExporter:
    """Merges a LoRA adapter into the base model and exports to GGUF.

    Uses Unsloth's built-in GGUF export. GPU-dependent methods
    import lazily.
    """

    def __init__(self, quantization: str = "Q4_K_M") -> None:
        if quantization not in _SUPPORTED_QUANTIZATIONS:
            raise ValueError(
                f"Unsupported quantization: {quantization}. "
                f"Supported: {sorted(_SUPPORTED_QUANTIZATIONS)}"
            )
        self.quantization = quantization

    def export(self, model_path: str, output_path: str) -> str:
        """Merge LoRA adapter and export to GGUF.

        Requires GPU. Lazy-imports unsloth.
        Returns the path to the exported GGUF file.
        """
        from unsloth import FastLanguageModel

        logger.info("Loading model from %s", model_path)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=4096,
            load_in_4bit=True,
        )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting GGUF (%s) to %s", self.quantization, output_path)
        model.save_pretrained_gguf(
            str(output.parent),
            tokenizer,
            quantization_method=self.quantization,
        )

        # Unsloth writes the file with a model-specific name, rename to target
        gguf_files = list(output.parent.glob("*.gguf"))
        if gguf_files:
            actual = gguf_files[0]
            if actual != output:
                actual.rename(output)

        logger.info("GGUF export complete: %s", output_path)
        return str(output)
