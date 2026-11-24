"""
Virtual-Token Embedding Export for Arbitrary-Slot CSPD & CRSP Serving (SGLang / vLLM)
=====================================================================================
Generalized virtual-token checkpoint exporter supporting arbitrary-slot CRSP insertion
(P0, P1, P2, P3, P4) as well as multi-expert concatenation in standard token-ID inference
engines (SGLang, vLLM, TensorRT-LLM).

Key Properties:
  1. Appends learned soft-prompt vectors [L_total, hidden] directly to
     ``model.embed_tokens.weight`` inside a derived safetensors shard while preserving
     every original tensor byte (verified by SHA-256) and symlinking unchanged shards.
  2. Supports inserting named expert virtual-token spans at ANY slot in token-ID space
     (e.g., P2 pre-query / P4 post-query after a shared long context, preserving 100%
     RadixAttention shared-prefix KV caching across tenants).
  3. Provides mandatory hard tied-embedding output masks (``logit_bias = -1e9`` for
     SGLang or ``allowed_token_ids = range(vocab_len)`` for vLLM), preventing high-norm
     virtual rows on tied-embedding models (such as Qwen3) from ever being sampled.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HARD_LOGIT_BIAS: float = -1e9  # -100 is insufficient for tied high-norm Qwen3 rows!


def extend_embedding_shard_bytes(
    source: Path,
    destination: Path,
    appended_bf16: bytes,
    *,
    rows: int,
    hidden: int,
    embed_key: str = "model.embed_tokens.weight",
) -> Dict[str, object]:
    """
    Pure-stdlib byte-exact safetensors shard extender: copies every original tensor byte
    unchanged, appends ``appended_bf16`` to ``embed_key``, and verifies SHA-256 integrity
    of every original tensor plus the appended bytes.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists() or source.resolve() == destination.resolve():
        raise ValueError("Fresh independent destination path required")
    if rows <= 0 or len(appended_bf16) != rows * hidden * 2:
        raise ValueError(f"Expected {rows * hidden * 2} BF16 bytes, got {len(appended_bf16)}")

    with source.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError("Missing safetensors 8-byte header size")
        header_size = struct.unpack("<Q", prefix)[0]
        header = json.loads(f.read(header_size))
        data_start = 8 + header_size
        entries = sorted((v["data_offsets"][0], k, v) for k, v in header.items() if k != "__metadata__")

        cursor = 0
        for start, _, val in entries:
            end = val["data_offsets"][1]
            if start != cursor or end < start:
                raise ValueError("Non-contiguous safetensors offsets")
            cursor = end
        if data_start + cursor != source.stat().st_size:
            raise ValueError("Source safetensors file size mismatch")
        if embed_key not in header:
            raise ValueError(f"Embedding tensor {embed_key!r} not found")

        emb = header[embed_key]
        if emb["dtype"] != "BF16" or len(emb["shape"]) != 2 or emb["shape"][1] != hidden:
            raise ValueError(f"Unexpected embedding metadata: {emb}")
        orig_vocab = int(emb["shape"][0])

        new_header = copy.deepcopy(header)
        cursor = 0
        for start, key, val in entries:
            size = val["data_offsets"][1] - start
            extra = len(appended_bf16) if key == embed_key else 0
            new_header[key]["data_offsets"] = [cursor, cursor + size + extra]
            if key == embed_key:
                new_header[key]["shape"][0] += rows
            cursor += size + extra

        encoded = json.dumps(new_header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        encoded += b" " * ((-len(encoded)) % 8)

        expected_sha: Dict[str, str] = {}
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as out:
            out.write(struct.pack("<Q", len(encoded)))
            out.write(encoded)
            for start, key, val in entries:
                rem = val["data_offsets"][1] - start
                f.seek(data_start + start)
                h = hashlib.sha256()
                while rem:
                    chunk = f.read(min(4 * 1024 * 1024, rem))
                    if not chunk:
                        raise ValueError("Unexpected EOF reading source shard")
                    out.write(chunk)
                    h.update(chunk)
                    rem -= len(chunk)
                if key == embed_key:
                    out.write(appended_bf16)
                expected_sha[key] = h.hexdigest()
            out.flush()
            os.fsync(out.fileno())

    # Independent read-back verification
    with destination.open("rb") as f:
        new_start = 8 + len(encoded)
        for start, key, val in entries:
            f.seek(new_start + new_header[key]["data_offsets"][0])
            rem = val["data_offsets"][1] - start
            h = hashlib.sha256()
            while rem:
                chunk = f.read(min(4 * 1024 * 1024, rem))
                h.update(chunk)
                rem -= len(chunk)
            if h.hexdigest() != expected_sha[key]:
                raise ValueError(f"Byte corruption detected for {key}")
            if key == embed_key and f.read(len(appended_bf16)) != appended_bf16:
                raise ValueError("Appended virtual-token bytes did not match read-back")
        if destination.stat().st_size != new_start + cursor:
            raise ValueError("Extended shard file size mismatch")

    return {
        "original_vocab_size": orig_vocab,
        "extended_vocab_size": orig_vocab + rows,
        "added_bytes": len(appended_bf16),
        "appended_sha256": hashlib.sha256(appended_bf16).hexdigest(),
    }


def build_derived_virtual_checkpoint(
    source_model_dir: Path,
    dest_dir: Path,
    named_prompts: Sequence[Tuple[str, "object"]],  # sequence of (name, torch.Tensor [L_i, hidden])
) -> Dict[str, object]:
    """
    Create a derived HuggingFace checkpoint directory containing one or more named
    soft prompts appended as contiguous virtual token IDs. Unmodified safetensors
    shards are symlinked (zero extra disk copy).
    """
    import torch

    source_model_dir, dest_dir = Path(source_model_dir), Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=False)

    config = json.loads((source_model_dir / "config.json").read_text(encoding="utf-8"))
    hidden = int(config["hidden_size"])
    index_path = source_model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_name = index["weight_map"]["model.embed_tokens.weight"]
    else:
        index = None
        shard_name = "model.safetensors"

    tensors = []
    span_lengths: List[Tuple[str, int]] = []
    for name, t in named_prompts:
        tf = t.detach().cpu().to(torch.bfloat16).contiguous()
        if tf.ndim != 2 or tf.size(1) != hidden or not torch.isfinite(tf).all():
            raise ValueError(f"Prompt {name!r} has invalid shape/values: {tuple(tf.shape)}")
        tensors.append(tf)
        span_lengths.append((name, int(tf.size(0))))

    stacked = torch.cat(tensors, dim=0).contiguous()
    raw_bf16 = stacked.view(torch.uint8).numpy().tobytes()
    proof = extend_embedding_shard_bytes(
        source_model_dir / shard_name,
        dest_dir / shard_name,
        raw_bf16,
        rows=int(stacked.size(0)),
        hidden=hidden,
    )

    config["vocab_size"] = int(proof["extended_vocab_size"])
    (dest_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if index is not None:
        index["metadata"]["total_size"] += int(proof["added_bytes"])
        (dest_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    for item in source_model_dir.iterdir():
        if item.name in {shard_name, "config.json", "model.safetensors.index.json"}:
            continue
        if item.suffix == ".safetensors":
            try:
                os.symlink(item.resolve(), dest_dir / item.name)
            except OSError:
                shutil.copyfile(item, dest_dir / item.name)
        elif item.is_file():
            shutil.copyfile(item, dest_dir / item.name)

    cursor = int(proof["original_vocab_size"])
    expert_token_ids: Dict[str, List[int]] = {}
    all_virtual_ids: List[int] = []
    for name, length in span_lengths:
        ids = list(range(cursor, cursor + length))
        expert_token_ids[name] = ids
        all_virtual_ids.extend(ids)
        cursor += length

    meta = {
        "original_vocab_size": int(proof["original_vocab_size"]),
        "extended_vocab_size": int(proof["extended_vocab_size"]),
        "expert_token_ids": expert_token_ids,
        "all_virtual_ids": all_virtual_ids,
        "appended_sha256": proof["appended_sha256"],
    }
    (dest_dir / "virtual_experts_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def build_hard_output_mask_ids(tokenizer_vocab_size: int, extended_vocab_size: int) -> List[int]:
    """
    Return all token IDs >= tokenizer_vocab_size (both unused padded embedding rows and
    appended virtual soft-prompt rows) that MUST be hard-masked (-1e9 in SGLang or
    excluded via allowed_token_ids in vLLM) on tied-embedding models such as Qwen3.
    """
    if extended_vocab_size < tokenizer_vocab_size:
        raise ValueError("extended_vocab_size must be >= tokenizer_vocab_size")
    return list(range(tokenizer_vocab_size, extended_vocab_size))


def render_multislot_virtual_ids(
    sys_open: Sequence[int],
    im_end: Sequence[int],
    user_open: Sequence[int],
    asst_open: Sequence[int],
    question_ids: Sequence[int],
    *,
    sys_text_ids: Optional[Sequence[int]] = None,
    history_turns_ids: Optional[Sequence[Tuple[Sequence[int], Sequence[int]]]] = None,
    rag_ids: Optional[Sequence[int]] = None,
    vids_p0_sys_start: Optional[Sequence[int]] = None,
    vids_p1_sys_end: Optional[Sequence[int]] = None,
    vids_p2_pre_query: Optional[Sequence[int]] = None,
    vids_p4_post_query: Optional[Sequence[int]] = None,
) -> List[int]:
    """
    Render exact token-ID sequence inserting virtual expert IDs at ANY slot(s) (P0/P1/P2/P4).
    Note: placing experts at P2/P4 leaves the entire [system + multi-turn history + RAG] prefix
    token-identical across tenants, preserving 100% RadixAttention prefix-cache sharing.
    """
    out: List[int] = []
    out.extend(sys_open)
    if vids_p0_sys_start:
        out.extend(vids_p0_sys_start)
    if sys_text_ids:
        out.extend(sys_text_ids)
    if vids_p1_sys_end:
        out.extend(vids_p1_sys_end)
    out.extend(im_end)

    if history_turns_ids:
        for u_ids, a_ids in history_turns_ids:
            out.extend(user_open)
            out.extend(u_ids)
            out.extend(im_end)
            out.extend(asst_open)
            out.extend(a_ids)
            out.extend(im_end)

    out.extend(user_open)
    if rag_ids:
        out.extend(rag_ids)
    if vids_p2_pre_query:
        out.extend(vids_p2_pre_query)
    out.extend(question_ids)
    if vids_p4_post_query:
        out.extend(vids_p4_post_query)
    out.extend(im_end)
    out.extend(asst_open)
    return [int(x) for x in out]
