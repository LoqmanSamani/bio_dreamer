from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None  # type: ignore
    DataLoader = object  # type: ignore





def make_collate_fn(
    dataset: Optional[Any] = None,
    pad_token_id: Optional[int] = None,
) -> Callable:
    """return a collate function that pads token sequences, stacks tensors,
    and handles nested target dicts with NaN for missing values."""
    if pad_token_id is None and dataset is not None:
        tok = getattr(dataset, "tokenizer", None)
        if tok is not None and hasattr(tok, "pad_token_id"):
            try:
                pad_token_id = int(tok.pad_token_id)
            except Exception:
                pad_token_id = 0
    pad_id = 0 if pad_token_id is None else pad_token_id

    def _pad_1d(tensors: List["torch.Tensor"], pad_value: int = 0) -> "torch.Tensor":
        max_len = max(t.size(0) for t in tensors) if tensors else 0
        out = tensors[0].new_full((len(tensors), max_len), pad_value)
        for i, t in enumerate(tensors):
            out[i, : t.size(0)] = t
        return out

    def _collate_targets(target_dicts: List[Optional[Dict[str, float]]]) -> Dict[str, "torch.Tensor"]:
        """Stack per-key target values into float tensors; NaN where missing."""
        keys = {"stability", "affinity", "activity", "fitness"}
        result: Dict[str, "torch.Tensor"] = {}
        for k in keys:
            values = []
            for d in target_dicts:
                if d is None:
                    values.append(float("nan"))
                else:
                    values.append(d.get(k, float("nan")))
            result[k] = torch.tensor(values, dtype=torch.float32)
        return result

    def _collate_field(values: List[Any]) -> Any:
        if all(v is None for v in values):
            return None

        non_none = [v for v in values if v is not None]

        # tensor or tensor-like
        if all(hasattr(v, "dim") for v in non_none):
            tensors = [
                v.squeeze(0)
                if (v.dim() == 2 and v.size(0) == 1)
                else v
                for v in non_none
            ]
            if all(t.dim() == 1 for t in tensors):
                return _pad_1d(tensors, pad_value=pad_id)
            try:
                return torch.stack(tensors)
            except Exception:
                max_last = max(t.size(-1) for t in tensors)
                out_shape = (len(tensors),) + tuple(tensors[0].size()[:-1]) + (max_last,)
                out = tensors[0].new_zeros(out_shape)
                for i, t in enumerate(tensors):
                    out[i, ..., : t.size(-1)] = t
                return out

        # lists/tuples of ints -> pad as token ids
        if all(isinstance(v, (list, tuple)) for v in non_none):
            tensors = [torch.tensor(v, dtype=torch.long) for v in non_none]
            return _pad_1d(tensors, pad_value=pad_id)

        # scalars
        if all(isinstance(v, (int, float)) for v in non_none):
            if any(isinstance(v, float) for v in non_none):
                return torch.tensor(
                    [float(v) if v is not None else float("nan") for v in values],
                    dtype=torch.float32,
                )
            return torch.tensor(
                [int(v) if v is not None else 0 for v in values],
                dtype=torch.long,
            )

        return list(values)

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            return {}
        out: Dict[str, Any] = {}
        keys = set().union(*(b.keys() for b in batch))
        for k in keys:
            vals = [b.get(k) for b in batch]
            if k == "targets" and all(isinstance(v, dict) or v is None for v in vals):
                out[k] = _collate_targets(vals)  # type: ignore[arg-type]
                continue
            if any(isinstance(v, dict) for v in vals if v is not None):
                dict_list = [v if isinstance(v, dict) else {} for v in vals]
                nested_keys = set().union(*(d.keys() for d in dict_list))
                nested_out: Dict[str, Any] = {}
                for nk in nested_keys:
                    nested_vals = [d.get(nk) for d in dict_list]
                    nested_out[nk] = _collate_field(nested_vals)
                out[k] = nested_out
                continue

            out[k] = _collate_field(vals)

        return out

    return collate_fn



def make_dataloader(
    dataset: Any,
    config: Any = None,
    shuffle: bool = True,
    collate_fn: Optional[Callable] = None,
) -> "DataLoader":
    if config is None:
        from biodreamer.protein_dreamer.config import ProteinDreamerConfig
        config = ProteinDreamerConfig().default()["dataloader"]
    _batch_size  = config.get("batch_size",  64)
    _num_workers = config.get("num_workers", 4)
    _pin_memory  = config.get("pin_memory",  True)
    if collate_fn is None:
        collate_fn = make_collate_fn(dataset)
    return DataLoader(
        dataset,
        batch_size=_batch_size,
        shuffle=shuffle,
        num_workers=_num_workers,
        pin_memory=_pin_memory,
        collate_fn=collate_fn,
    )



def make_dataloaders(
    config: Any = None,
    train_dataset: Optional[Any] = None,
    val_dataset: Optional[Any] = None,
    test_dataset: Optional[Any] = None
) -> Dict[str, Optional["DataLoader"]]:
    """build train/val/test DataLoaders"""
    if config is None:
        from biodreamer.protein_dreamer.config import ProteinDreamerConfig
        config = ProteinDreamerConfig().default()["dataloader"]
    _train_shuffle = config.get("train_shuffle", True)
    _val_shuffle   = config.get("val_shuffle",   False)
    _test_shuffle  = config.get("test_shuffle",  False)
    

    loaders: Dict[str, Optional[Any]] = {}

    if train_dataset is not None:
        loaders["train"] = make_dataloader(
            train_dataset,
            config=config,
            shuffle=_train_shuffle
        )
    else:
        loaders["train"] = None

    if val_dataset is not None:
        loaders["val"] = make_dataloader(
            val_dataset,
            config=config,
            shuffle=_val_shuffle
            )
    else:
        loaders["val"] = None

    if test_dataset is not None:
        loaders["test"] = make_dataloader(
            test_dataset,
            config=config,
            shuffle=_test_shuffle
        )
    else:
        loaders["test"] = None

    return loaders
