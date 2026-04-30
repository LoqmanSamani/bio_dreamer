from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional
import math
import logging
log = logging.getLogger(__name__)
try:
    import torch
    from torch.utils.data import DataLoader
except Exception: 
    torch = None 
    DataLoader = object




def _is_tensor_like(x: Any) -> bool:
    return hasattr(x, "numpy") or hasattr(x, "detach") or isinstance(x, (list, tuple))


def make_collate_fn(dataset: Optional[Any] = None, pad_token_id: Optional[int] = None) -> Callable:
    """create a collate function that pads token sequences and stacks tensors"""
    if pad_token_id is None and dataset is not None:
        tok = getattr(dataset, "tokenizer", None)
        if tok is not None and hasattr(tok, "pad_token_id"):
            try:
                pad_token_id = int(tok.pad_token_id)
            except Exception:
                pad_token_id = 0
    pad_token_id = 0 if pad_token_id is None else pad_token_id

    def _pad_1d(tensors: List["torch.Tensor"], pad_value: int = 0) -> "torch.Tensor":
        lengths = [t.size(0) for t in tensors]
        max_len = max(lengths) if lengths else 0
        out = tensors[0].new_full((len(tensors), max_len), pad_value)
        for i, t in enumerate(tensors):
            out[i, : t.size(0)] = t
        return out

    def _collate_field(values: List[Any]) -> Any:
        if all(v is None for v in values):
            return None

        if all(hasattr(v, "dim") for v in values if v is not None):
            tensors = [v.squeeze(0) if (v is not None and getattr(v, "dim", lambda: 1)() == 2 and v.size(0) == 1) else v for v in values if v is not None]
            if all(t.dim() == 1 for t in tensors):
                return _pad_1d(tensors, pad_value=pad_token_id)
            try:
                return torch.stack(tensors)
            except Exception:
                max_last = max(t.size(-1) for t in tensors)
                out_shape = (len(tensors),) + tuple(tensors[0].size()[:-1]) + (max_last,)
                out = tensors[0].new_zeros(out_shape)
                for i, t in enumerate(tensors):
                    sl = t.size(-1)
                    out[i, ..., :sl] = t
                return out
        if all(isinstance(v, (list, tuple)) for v in values if v is not None):
            tensors = [torch.tensor(v, dtype=torch.long) for v in values if v is not None]
            return _pad_1d(tensors, pad_value=pad_token_id)
        if all(isinstance(v, (int, float)) for v in values if v is not None):
            if any(isinstance(v, float) for v in values if v is not None):
                return torch.tensor([float(v) if v is not None else float('nan') for v in values], dtype=torch.float)
            return torch.tensor([int(v) if v is not None else 0 for v in values], dtype=torch.long)
        return [v for v in values]

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            return {}
        out: Dict[str, Any] = {}
        keys = set().union(*(b.keys() for b in batch))
        for k in keys:
            vals = [b.get(k, None) for b in batch]
            if any(isinstance(v, dict) for v in vals if v is not None):
                dict_list = [v if isinstance(v, dict) else {} for v in vals]
                nested_keys = set().union(*(d.keys() for d in dict_list))
                nested_out: Dict[str, Any] = {}
                for nk in nested_keys:
                    nested_vals = [d.get(nk, None) for d in dict_list]
                    nested_out[nk] = _collate_field(nested_vals)
                out[k] = nested_out
            else:
                out[k] = _collate_field(vals)
        return out
    return collate_fn



def make_dataloader(
    dataset: Any,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    collate_fn: Optional[Callable] = None,
) -> "DataLoader":
    """construct a PyTorch DataLoader"""
    if collate_fn is None:
        collate_fn = make_collate_fn(dataset)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)



def make_dataloaders(
    train_dataset: Optional[Any],
    val_dataset: Optional[Any] = None,
    test_dataset: Optional[Any] = None,
    batch_size: int = 32,
    val_batch_size: Optional[int] = None,
    test_batch_size: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> Dict[str, Optional["DataLoader"]]:
    """build train/val/test DataLoaders"""
    loaders = {}
    if train_dataset is not None:
        train_collate = make_collate_fn(train_dataset)
        loaders["train"] = make_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=train_collate)
    else:
        loaders["train"] = None
    if val_dataset is not None:
        val_collate = make_collate_fn(val_dataset)
        vb = val_batch_size or batch_size
        loaders["val"] = make_dataloader(val_dataset, batch_size=vb, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=val_collate)
    else:
        loaders["val"] = None

    if test_dataset is not None:
        test_collate = make_collate_fn(test_dataset)
        tb = test_batch_size or batch_size
        loaders["test"] = make_dataloader(test_dataset, batch_size=tb, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=test_collate)
    else:
        loaders["test"] = None
    return loaders


__all__ = ["make_collate_fn", "make_dataloader", "make_dataloaders"]