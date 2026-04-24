from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import torch
from transformers import (AutoModel, AutoTokenizer, EsmModel, EsmTokenizer, EsmForProteinFolding)




class ModelTask(Enum):
    """enumeration of possible tasks for loaded models"""
    SEQUENCE_EMBEDDING = 'sequence_embedding'   # e.g., ESM-2, ProtTrans
    STRUCTURE_EMBEDDING = 'structure_embedding' # e.g., GVP, SaPort, GVP-GNN
    STRUCTURE_PREDICTION = 'structure_prediction' # e.g., ESMFold
    INVERSE_FOLDING = 'inverse_folding' # e.g., ESM-IF1, ESM-IF2
    FUNCTION_PREDICTION = 'function_prediction' # e.g., DeepFRI, ESM-2 function head
    OTHER = 'other' # for any other tasks not covered above
    
    
    
class ModelBackend(Enum):
    """enumeration of possible backends for loading models"""
    AUTO = 'auto' # AutoModel, AutoTokenizer
    ESM = 'esm' # Fair/Meta esm family of models (ESM-1b, ESM-2, ESM-IF1/2, ESMFold, etc.)
    ESMFOLD = 'esmfold' # ESMFoldingTrunk 
    CUSTOM = 'custom' # user-supplied loader fn
    OTHER = 'other' # for any other backends not covered
    
    
    
@dataclass
class ModelConfig:
    """configuration for loading a HuggingFace model"""
    hf_id: str                          # e.g. "facebook/esm2_t33_650M_UR50D"
    task: ModelTask
    backend: ModelBackend = ModelBackend.AUTO
    layer_index: int = -1               # which layer to extract embeddings from
    half_precision: bool = False        # fp16 for large models
    extra: Dict[str, Any] = field(default_factory=dict)
    
    


class HFModelLoader:
    """
    HuggingFace model loader
    handles sequence embedding, structure prediction, and structure embedding
    """
    # registry: shorthand name -> ModelConfig
    REGISTRY: Dict[str, ModelConfig] = {
        "esm2-650m": ModelConfig(
            hf_id="facebook/esm2_t33_650M_UR50D",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.ESM,
            layer_index=-1,
            half_precision=False,
        ),
        "esm2-3b": ModelConfig(
            hf_id="facebook/esm2_t36_3B_UR50D",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.ESM,
            layer_index=-1,
            half_precision=True,
        ),
        "esmfold": ModelConfig(
            hf_id="facebook/esmfold_v1",
            task=ModelTask.STRUCTURE_PREDICTION,
            backend=ModelBackend.ESMFOLD,
            half_precision=True,
        ),
        "prottrans-t5": ModelConfig(
            hf_id="Rostlab/prot_t5_xl_uniref50",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.AUTO,
            half_precision=False,
        ),
    }
    def __init__(
        self, 
        device: Optional[torch.device] = None, 
        cache_dir: Optional[str] = None
        ) -> None:
        self.device = device if device is not None and isinstance(device, torch.device) else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.cache_dir = cache_dir
        self._loaded: Dict[str, tuple] = {} # model_name -> (model, tokenizer)
        
    # public API
    def load(self, name_or_id: str, config: Optional[ModelConfig] = None) -> tuple:
        """
        load by registry name ('esm2-650m') or raw hugging-face id with a config.
        returns (model, tokenizer). tokenizer is none for structure models.
        """
        if name_or_id in self._loaded:
            return self._loaded[name_or_id]
        
        cfg = config or self.REGISTRY.get(name_or_id)
        if cfg is None:
            # fallback: try treating it as a raw hf id with AUTO backend
            cfg = ModelConfig(
                hf_id=name_or_id,
                task=ModelTask.SEQUENCE_EMBEDDING,
                backend=ModelBackend.AUTO,
            )
        model, tokenizer = self._dispatch(cfg) # load the model based on its backend
        self._loaded[name_or_id] = (model, tokenizer)
        return model, tokenizer
    
    def get(self, name: str) -> Optional[tuple]:
        """return already-loaded (model, tokenizer) or raise"""
        if name not in self._loaded:
            raise KeyError(f"Model '{name}' not loaded. Call .load('{name}') first.")
        return self._loaded[name]
    
    def register(self, name: str, config: ModelConfig):
        """add a custom model to the registry"""
        self.REGISTRY[name] = config
    
    def unload(self, name: str):
        """free gpu memory"""
        if name in self._loaded:
            model, _ = self._loaded.pop(name)
            del model
            torch.cuda.empty_cache()
    
    def _dispatch(self, cfg: ModelConfig):
        """dispatch loading based on backend type"""
        loaders = {
            ModelBackend.AUTO:    self._load_auto,
            ModelBackend.ESM:     self._load_esm,
            ModelBackend.ESMFOLD: self._load_esmfold,
            ModelBackend.CUSTOM:  self._load_custom,
        }
        loader = loaders.get(cfg.backend)
        if loader is None:
            raise ValueError(f"Unknown backend: {cfg.backend}")
        return loader(cfg)
    
    def _load_auto(self, cfg: ModelConfig):
        """load using HuggingFace AutoModel and AutoTokenizer (default)"""
        tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        model = AutoModel.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        return self._finalize(model, cfg), tokenizer

    def _load_esm(self, cfg: ModelConfig):
        """load esm family models using fairseq's transformers (esm-1b, esm-2, esm-If1/2, etc.)"""
        tokenizer = EsmTokenizer.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        model = EsmModel.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        return self._finalize(model, cfg), tokenizer

    def _load_esmfold(self, cfg: ModelConfig):
        """load ESMFold models"""
        model = EsmForProteinFolding.from_pretrained(
            cfg.hf_id,
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir,
        )
        # ESMFold has a built-in trunk; optionally offload the LM part
        if cfg.extra.get("chunk_size"):
            model.trunk.set_chunk_size(cfg.extra["chunk_size"])
        return self._finalize(model, cfg), None   # no external tokenizer

    def _load_custom(self, cfg: ModelConfig):
        """load using a user-supplied loader function that returns (model, tokenizer)"""
        loader_fn = cfg.extra.get("loader_fn")
        if loader_fn is None:
            raise ValueError("CUSTOM backend requires extra['loader_fn']")
        model, tokenizer = loader_fn(cfg)
        return self._finalize(model, cfg), tokenizer
    
    def _finalize(self, model, cfg: ModelConfig):
        """set model to eval mode, move to device, and convert to half precision if specified"""
        if cfg.half_precision:
            model = model.half()
        model = model.to(self.device)
        model.eval()
        return model 