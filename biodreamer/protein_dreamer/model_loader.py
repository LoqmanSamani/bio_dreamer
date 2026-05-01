from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    EsmForMaskedLM,
    EsmForProteinFolding,
    EsmModel,
    EsmTokenizer,
    T5EncoderModel,
    T5Tokenizer,
)

logger = logging.getLogger(__name__)


class ModelTask(Enum):
    SEQUENCE_EMBEDDING = "sequence_embedding"
    STRUCTURE_EMBEDDING = "structure_embedding"
    STRUCTURE_PREDICTION = "structure_prediction"
    INVERSE_FOLDING = "inverse_folding"
    FUNCTION_PREDICTION = "function_prediction"
    OTHER = "other"


class ModelBackend(Enum):
    AUTO = "auto"
    ESM = "esm"
    ESM_IF = "esm_if"
    ESM_LM = "esm_lm"
    ESMFOLD = "esmfold"
    SAPROT = "saprot"
    PROTTRANS = "prottrans"
    CUSTOM = "custom"
    OTHER = "other"


@dataclass
class ModelConfig:
    hf_id: str
    task: ModelTask
    backend: ModelBackend = ModelBackend.AUTO
    layer_index: int = -1
    half_precision: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


class HFModelLoader:
    """Registry and loader for pretrained protein backbone models."""

    REGISTRY: Dict[str, ModelConfig] = {
        "esm2-8m": ModelConfig(
            hf_id="facebook/esm2_t6_8M_UR50D",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.ESM,
        ),
        "esm2-35m": ModelConfig(
            hf_id="facebook/esm2_t12_35M_UR50D",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.ESM,
        ),
        "esm2-150m": ModelConfig(
            hf_id="facebook/esm2_t30_150M_UR50D",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.ESM,
        ),
        "esm2-650m": ModelConfig(
            hf_id="facebook/esm2_t33_650M_UR50D",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.ESM,
        ),
        "esm2-3b": ModelConfig(
            hf_id="facebook/esm2_t36_3B_UR50D",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.ESM,
            half_precision=True,
        ),
        "esm2-15b": ModelConfig(
            hf_id="facebook/esm2_t48_15B_UR50D",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.ESM,
            half_precision=True,
        ),
        "esm2-650m-lm": ModelConfig(
            hf_id="facebook/esm2_t33_650M_UR50D",
            task=ModelTask.FUNCTION_PREDICTION,
            backend=ModelBackend.ESM_LM,
        ),
        "prottrans-t5-xl": ModelConfig(
            hf_id="Rostlab/prot_t5_xl_uniref50",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.PROTTRANS,
            half_precision=True,
        ),
        "prottrans-t5-bfd": ModelConfig(
            hf_id="Rostlab/prot_t5_xl_bfd",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.PROTTRANS,
            half_precision=True,
        ),
        "ankh-base": ModelConfig(
            hf_id="ElnaggarLab/ankh-base",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.AUTO,
        ),
        "ankh-large": ModelConfig(
            hf_id="ElnaggarLab/ankh-large",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.AUTO,
            half_precision=True,
        ),
        "esmfold": ModelConfig(
            hf_id="facebook/esmfold_v1",
            task=ModelTask.STRUCTURE_PREDICTION,
            backend=ModelBackend.ESMFOLD,
            half_precision=True,
        ),
        "saprot-35m": ModelConfig(
            hf_id="westlake-repl/SaProt_35M_AF2",
            task=ModelTask.STRUCTURE_EMBEDDING,
            backend=ModelBackend.SAPROT,
        ),
        "saprot-650m": ModelConfig(
            hf_id="westlake-repl/SaProt_650M_AF2",
            task=ModelTask.STRUCTURE_EMBEDDING,
            backend=ModelBackend.SAPROT,
        ),
        "esm-if1": ModelConfig(
            hf_id="facebook/esm-if1-gvp",
            task=ModelTask.INVERSE_FOLDING,
            backend=ModelBackend.ESM_IF,
            half_precision=True,
        ),
    }

    def __init__(
        self,
        device: Optional[torch.device] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.cache_dir = cache_dir
        self._loaded: Dict[str, tuple] = {}

    def load(self, name_or_id: str, config: Optional[ModelConfig] = None) -> tuple:
        """Load by registry name or raw HF id. Returns (model, tokenizer)."""
        if name_or_id in self._loaded:
            return self._loaded[name_or_id]

        cfg = config or self.REGISTRY.get(name_or_id)
        if cfg is None:
            cfg = ModelConfig(
                hf_id=name_or_id,
                task=ModelTask.SEQUENCE_EMBEDDING,
                backend=ModelBackend.AUTO,
            )

        model, tokenizer = self._dispatch(cfg)
        self._loaded[name_or_id] = (model, tokenizer)
        return model, tokenizer

    def get(self, name: str) -> tuple:
        if name not in self._loaded:
            raise KeyError(f"Model '{name}' not loaded. Call .load('{name}') first.")
        return self._loaded[name]

    def register(self, name: str, config: ModelConfig) -> None:
        self.REGISTRY[name] = config

    def unload(self, name: str) -> None:
        if name in self._loaded:
            model, _ = self._loaded.pop(name)
            del model
            torch.cuda.empty_cache()

    def _dispatch(self, cfg: ModelConfig) -> tuple:
        loaders = {
            ModelBackend.AUTO:      self._load_auto,
            ModelBackend.ESM:       self._load_esm,
            ModelBackend.ESM_IF:    self._load_esm_if,
            ModelBackend.ESM_LM:    self._load_esm_lm,
            ModelBackend.ESMFOLD:   self._load_esmfold,
            ModelBackend.SAPROT:    self._load_saprot,
            ModelBackend.PROTTRANS: self._load_prottrans,
            ModelBackend.CUSTOM:    self._load_custom,
        }
        loader = loaders.get(cfg.backend)
        if loader is None:
            raise ValueError(f"Unknown backend: {cfg.backend}")
        return loader(cfg)

    def _finalize(self, model: Any, cfg: ModelConfig) -> Any:
        if cfg.half_precision:
            model = model.half()
        model = model.to(self.device)
        model.eval()
        return model

    def _load_auto(self, cfg: ModelConfig) -> tuple:
        tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        model = AutoModel.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        return self._finalize(model, cfg), tokenizer

    def _load_esm(self, cfg: ModelConfig) -> tuple:
        tokenizer = EsmTokenizer.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        model = EsmModel.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        return self._finalize(model, cfg), tokenizer

    def _load_esm_lm(self, cfg: ModelConfig) -> tuple:
        tokenizer = EsmTokenizer.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        model = EsmForMaskedLM.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        return self._finalize(model, cfg), tokenizer

    def _load_esm_if(self, cfg: ModelConfig) -> tuple:
        tokenizer = EsmTokenizer.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        model = EsmForProteinFolding.from_pretrained(
            cfg.hf_id,
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir,
        )
        return self._finalize(model, cfg), tokenizer

    def _load_saprot(self, cfg: ModelConfig) -> tuple:
        tokenizer = EsmTokenizer.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        model = EsmModel.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        return self._finalize(model, cfg), tokenizer

    def _load_prottrans(self, cfg: ModelConfig) -> tuple:
        tokenizer = T5Tokenizer.from_pretrained(
            cfg.hf_id, cache_dir=self.cache_dir, do_lower_case=False
        )
        model = T5EncoderModel.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        return self._finalize(model, cfg), tokenizer

    def _load_esmfold(self, cfg: ModelConfig) -> tuple:
        model = EsmForProteinFolding.from_pretrained(
            cfg.hf_id,
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir,
        )
        if cfg.extra.get("chunk_size"):
            model.trunk.set_chunk_size(cfg.extra["chunk_size"])
        return self._finalize(model, cfg), None

    def _load_custom(self, cfg: ModelConfig) -> tuple:
        loader_fn = cfg.extra.get("loader_fn")
        if loader_fn is None:
            raise ValueError("CUSTOM backend requires extra['loader_fn']")
        model, tokenizer = loader_fn(cfg)
        return self._finalize(model, cfg), tokenizer
