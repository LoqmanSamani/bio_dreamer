from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import torch
from transformers import (
    AutoModel, AutoTokenizer,
    EsmModel, EsmTokenizer, EsmForProteinFolding,
    EsmForMaskedLM,               
    T5EncoderModel, T5Tokenizer,    
)




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
    AUTO = 'auto'           # AutoModel + AutoTokenizer (generic fallback)
    ESM = 'esm'             # EsmModel + EsmTokenizer (ESM-2, ESM-1b)
    ESM_IF = 'esm_if'       # EsmForProteinFolding used as IF encoder (ESM-IF1)
    ESM_LM = 'esm_lm'       # EsmForMaskedLM (ESM-2 with LM head, function prediction)
    ESMFOLD = 'esmfold'     # EsmForProteinFolding (structure prediction)
    SAPROT = 'saprot'       # SaProt: ESM-2 backbone, needs EsmTokenizer not AutoTokenizer
    PROTTRANS = 'prottrans' # T5EncoderModel + T5Tokenizer (ProtTrans-T5 family)
    CUSTOM = 'custom'       # user-supplied loader_fn in extra dict
    OTHER = 'other'         # for any other backends not covered
    
    
    
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

        
        # sequence embedding — esm-2 family (meta)                           
        # hf docs: https://huggingface.co/facebook/esm2_t6_8M_UR50D          
        "esm2-8m": ModelConfig(
            hf_id="facebook/esm2_t6_8M_UR50D",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.ESM,
            layer_index=-1,
            half_precision=False,
        ),
        "esm2-35m": ModelConfig(
            hf_id="facebook/esm2_t12_35M_UR50D",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.ESM,
            layer_index=-1,
            half_precision=False,
        ),
        "esm2-150m": ModelConfig(
            hf_id="facebook/esm2_t30_150M_UR50D",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.ESM,
            layer_index=-1,
            half_precision=False,
        ),
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
        "esm2-15b": ModelConfig(
            hf_id="facebook/esm2_t48_15B_UR50D",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.ESM,
            layer_index=-1,
            half_precision=True,
        ),
        # sequence embedding,esm-2 with lm head (for function prediction)  
        # use esm_lm backend to keep the masked-lm head intact               
        "esm2-650m-lm": ModelConfig(
            hf_id="facebook/esm2_t33_650M_UR50D",
            task=ModelTask.FUNCTION_PREDICTION,
            backend=ModelBackend.ESM_LM,
            layer_index=-1,
            half_precision=False,
        ),
        # sequence embedding, prot-trans family                     
        # T5EncoderModel: encoder-only, no decoder needed for embeddings      
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
        # sequence embedding, Ankh                            
        # large protein language model trained on UniRef + BFD                       
        "ankh-base": ModelConfig(
            hf_id="ElnaggarLab/ankh-base",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.AUTO,
            half_precision=False,
        ),
        "ankh-large": ModelConfig(
            hf_id="ElnaggarLab/ankh-large",
            task=ModelTask.SEQUENCE_EMBEDDING,
            backend=ModelBackend.AUTO,
            half_precision=True,
        ),
        # structure prediction — eam-fold                        
        # end-to-end folding, tokenizer is internal, returns PDB/coordinates 
        "esmfold": ModelConfig(
            hf_id="facebook/esmfold_v1",
            task=ModelTask.STRUCTURE_PREDICTION,
            backend=ModelBackend.ESMFOLD,
            half_precision=True,
        ),
        # structure-aware sequence embedding, sa-prot            #
        "saprot-35m": ModelConfig(
            hf_id="westlake-repl/SaProt_35M_AF2",
            task=ModelTask.STRUCTURE_EMBEDDING,
            backend=ModelBackend.SAPROT,
            half_precision=False,
        ),
        "saprot-650m": ModelConfig(
            hf_id="westlake-repl/SaProt_650M_AF2",
            task=ModelTask.STRUCTURE_EMBEDDING,
            backend=ModelBackend.SAPROT,
            half_precision=False,
        ),
        # inverse folding, esm-if1                                  
        # gvp-transformer; takes 3d coordinates -> sequence logits       
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
        cache_dir: Optional[str] = None
        ) -> None:
        self.device = device if device is not None and isinstance(device, torch.device) else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.cache_dir = cache_dir
        self._loaded: Dict[str, tuple] = {} # model_name -> (model, tokenizer)
        
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
    
    def _load_auto(self, cfg: ModelConfig):
        """load using HuggingFace AutoModel and AutoTokenizer (default)"""
        tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        model = AutoModel.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        return self._finalize(model, cfg), tokenizer

    def _load_esm(self, cfg: ModelConfig):
        """load esm family models using fairseq's transformers (esm-1b, esm-2, etc.)"""
        tokenizer = EsmTokenizer.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        model = EsmModel.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        return self._finalize(model, cfg), tokenizer

    def _load_esm_lm(self, cfg: ModelConfig):
        """load esm-2 with the masked-lm head (for function prediction / logit extraction)"""
        tokenizer = EsmTokenizer.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        model = EsmForMaskedLM.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        return self._finalize(model, cfg), tokenizer

    def _load_esm_if(self, cfg: ModelConfig):
        """load esm-if1 (inverse folding)"""
        from transformers import EsmTokenizer as EsmTok
        tokenizer = EsmTok.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        model = EsmForProteinFolding.from_pretrained(
            cfg.hf_id,
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir,
        )
        return self._finalize(model, cfg), tokenizer

    def _load_saprot(self, cfg: ModelConfig):
        """load sa-prot (structure-aware protein transformer)"""
        tokenizer = EsmTokenizer.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        model = EsmModel.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        return self._finalize(model, cfg), tokenizer

    def _load_prottrans(self, cfg: ModelConfig):
        """load prot-trans-t5 encoder"""
        tokenizer = T5Tokenizer.from_pretrained(
            cfg.hf_id, cache_dir=self.cache_dir, do_lower_case=False
        )
        model = T5EncoderModel.from_pretrained(cfg.hf_id, cache_dir=self.cache_dir)
        return self._finalize(model, cfg), tokenizer

    def _load_esmfold(self, cfg: ModelConfig):
        """load esm-fold models"""
        model = EsmForProteinFolding.from_pretrained(
            cfg.hf_id,
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir,
        )
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