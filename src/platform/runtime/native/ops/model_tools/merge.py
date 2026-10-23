from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from safetensors import safe_open

from ...lora.key_mapping import (
    LoraDelta,
    _iter_lokr_stems,
    _iter_stems,
    map_lora_keys,
)
from .errors import MergeError
from .lora_io import (
    LoraInput,
    is_text_encoder_stem,
    refusal_for_unsupported_kind,
    resolve_family,
)
from .skeleton import Skeleton, build_skeleton
from .source import read_header
from .spec import DTYPE_TO_TORCH, TensorSpec

MODES = ("concat", "reduce")
OUTPUT_DTYPES = ("F32", "BF16", "F16")
KEY_PREFIX = "diffusion_model."
STEM_PREFIXES = ("lora_unet_", "diffusion_model.", "transformer.", "base_model.model.", "lycoris_", "lora_transformer_")

_SPEC_TO_COARSE = {
    "flux": "flux", "flux3": "flux", "qwen_image": "qwen", "qwen_image21": "qwen", "wan": "wan",
    "z_image": "z_image", "krea2": "krea2", "moss_image": "moss", "ltx": "ltx",
}

_FAMILY_SIGNATURES: dict[str, tuple[re.Pattern[str], ...]] = {
    "moss": (re.compile(r"^layers_\d+_proj_(q|k|v|qkv|o|ffn_up_gate|ffn_down)(_|$)"), re.compile(r"^text_embedder_blocks_")),
    "krea2": (
        re.compile(r"^(transformer_)?blocks_\d+_attn_(wq|wk|wv|wo|gate|to_gate)(_|$)"),
        re.compile(r"^(txtfusion|text_fusion)_"),
        re.compile(r"^transformer_blocks_\d+_attn_to_gate"),
    ),
    "flux": (
        re.compile(r"^(double|single)_blocks_\d+"),
        re.compile(r"^single_transformer_blocks_\d+"),
        re.compile(r"^transformer_blocks_\d+_(ff_net|ff_context|norm1_linear|norm1_context)"),
    ),
    "qwen": (re.compile(r"^transformer_blocks_\d+_(img|txt)_(mlp|mod)"),),
    "wan": (re.compile(r"^blocks_\d+_(self_attn|cross_attn|attn1|attn2|ffn)(_|$)"),),
    "z_image": (
        re.compile(r"^(layers|noise_refiner|context_refiner)_\d+_(attention|feed_forward|adaLN)", re.IGNORECASE),
    ),
    "ltx": (re.compile(r"^transformer_blocks_\d+_(attn1|attn2|audio_attn1|audio_attn2|audio_to_video|video_to_audio)"),),
}


@dataclass
class MergeReport:
    mode: str
    family: str | None
    key_format: str
    rank: int | None
    layers: int
    output_dtype: str
    energy: dict[str, float] = field(default_factory=dict)
    lokr_energy: dict[str, float] = field(default_factory=dict)
    per_lora: dict[str, dict[str, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def min_energy_kept(self) -> float | None:
        values = list(self.energy.values()) + list(self.lokr_energy.values())
        return min(values) if values else None


@dataclass
class _Side:
    keys: tuple[str, ...]
    shape: tuple[int, int]


@dataclass
class _Entry:
    lora: int
    strength: float
    source: str
    alpha: float
    plain: bool
    up: _Side | None = None
    down: _Side | None = None
    w1: _Side | None = None
    w2: _Side | None = None
    row_slice: tuple[int, int] | None = None

    @property
    def rank(self) -> int:
        return self.down.shape[0] if self.plain else 0

    @property
    def out_dim(self) -> int:
        return self.up.shape[0] if self.plain else self.w1.shape[0] * self.w2.shape[0]

    @property
    def in_dim(self) -> int:
        return self.down.shape[1] if self.plain else self.w1.shape[1] * self.w2.shape[1]


def _flatten(stem: str) -> str:
    for prefix in STEM_PREFIXES:
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    return stem.replace(".", "_")


def detect_lora_family(sd: dict[str, torch.Tensor]) -> str | None:
    stems = [s for s, _, _ in _iter_stems(sd)] + [s for s, _ in _iter_lokr_stems(sd)]
    scores: dict[str, int] = defaultdict(int)
    for stem in stems:
        flat = _flatten(stem)
        for family, patterns in _FAMILY_SIGNATURES.items():
            if any(p.search(flat) for p in patterns):
                scores[family] += 1
    if not scores:
        return None
    return max(scores, key=lambda f: scores[f])


def kron_to_lora(
    w1: torch.Tensor, w2: torch.Tensor, scale: float, rank: int
) -> tuple[torch.Tensor, torch.Tensor, float]:
    ua, sa, vha = torch.linalg.svd(w1.to(torch.float32), full_matrices=False)
    ub, sb, vhb = torch.linalg.svd(w2.to(torch.float32), full_matrices=False)
    products = torch.outer(sa, sb).reshape(-1)
    total = float((products ** 2).sum())
    k = max(1, min(int(rank), products.numel()))
    top = torch.topk(products, k)
    ia = top.indices // sb.numel()
    ib = top.indices % sb.numel()
    up = (ua[:, ia].unsqueeze(1) * ub[:, ib].unsqueeze(0)).reshape(ua.shape[0] * ub.shape[0], k)
    down = (vha[ia].unsqueeze(2) * vhb[ib].unsqueeze(1)).reshape(k, vha.shape[1] * vhb.shape[1])
    up = up * (top.values * float(scale)).unsqueeze(0)
    kept = float((top.values ** 2).sum())
    return up, down, (kept / total if total > 0 else 1.0)


def _reduce(up: torch.Tensor, down: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    qu, ru = torch.linalg.qr(up, mode="reduced")
    qv, rv = torch.linalg.qr(down.T, mode="reduced")
    core = ru @ rv.T
    uc, s, vch = torch.linalg.svd(core, full_matrices=False)
    k = max(1, min(rank, s.numel()))
    total = float((s ** 2).sum())
    kept = float((s[:k] ** 2).sum())
    new_up = qu @ (uc[:, :k] * s[:k].unsqueeze(0))
    new_down = vch[:k] @ qv.T
    return new_up, new_down, (kept / total if total > 0 else 1.0)


def _check_dialects(per_lora_stems: dict[str, set[str]]) -> None:
    seen: dict[str, tuple[str, str]] = {}
    for name, stems in per_lora_stems.items():
        for stem in stems:
            ident = _flatten(stem)
            prior = seen.get(ident)
            if prior is not None and prior[0] != stem:
                raise MergeError(
                    f"{name} and {prior[1]} name the same layer with different key spellings "
                    f"({stem} vs {prior[0]}); pass model_path so layers can be matched by the model"
                )
            seen.setdefault(ident, (stem, name))


class _LoraFile:
    def __init__(self, path: str) -> None:
        path_obj = Path(path)
        if path_obj.suffix.lower() not in (".safetensors", ".sft"):
            raise MergeError(f"{path_obj.name}: LoRA files must be safetensors")
        if not path_obj.is_file():
            raise MergeError(f"{path_obj.name}: file not found")
        self.path = path
        self.name = path_obj.name
        with safe_open(path, framework="pt", device="cpu") as f:
            self.shapes: dict[str, tuple[int, ...]] = {}
            self.dtypes: dict[str, str] = {}
            self.scalars: dict[str, float] = {}
            for key in f.keys():
                view = f.get_slice(key)
                self.shapes[key] = tuple(view.get_shape())
                self.dtypes[key] = view.get_dtype()
                if key.endswith(".alpha"):
                    self.scalars[key] = float(f.get_tensor(key).item())

    def meta_state(self, index: int) -> dict[str, torch.Tensor]:
        sd: dict[str, torch.Tensor] = {}
        for key, shape in self.shapes.items():
            if key in self.scalars:
                sd[key] = torch.tensor(self.scalars[key])
                continue
            tensor = torch.empty(shape, dtype=DTYPE_TO_TORCH.get(self.dtypes[key], torch.float32), device="meta")
            tensor._src = (index, key)
            sd[key] = tensor
        return sd


def _src(tensor: torch.Tensor) -> str:
    return tensor._src[1]


def _plain_entry(index: int, lora: LoraInput, name: str, up: torch.Tensor, down: torch.Tensor,
                 alpha: float, row_slice) -> _Entry:
    return _Entry(
        lora=index, strength=lora.strength, source=name, alpha=alpha, plain=True,
        up=_Side((_src(up),), tuple(up.shape)), down=_Side((_src(down),), tuple(down.shape)),
        row_slice=row_slice,
    )


def _lokr_entry(index: int, lora: LoraInput, name: str, d: LoraDelta) -> _Entry:
    sides = []
    for i, direct in enumerate((d.up, d.down)):
        factors = d.kron_factors[i] if d.kron_factors is not None else None
        if factors is None:
            sides.append(_Side((_src(direct),), tuple(direct.shape)))
        else:
            a, b = factors
            sides.append(_Side((_src(a), _src(b)), (a.shape[0], b.shape[1])))
    return _Entry(
        lora=index, strength=lora.strength, source=name, alpha=d.alpha, plain=False,
        w1=sides[0], w2=sides[1],
    )


def _index_stem_mode(index: int, lora: LoraInput, file: _LoraFile, sd) -> dict[str, list[_Entry]]:
    out: dict[str, list[_Entry]] = defaultdict(list)
    consumed: set[str] = set()
    for stem, up_key, down_key in _iter_stems(sd):
        consumed.update((up_key, down_key))
        alpha_key = f"{stem}.alpha"
        consumed.add(alpha_key)
        rank = sd[down_key].shape[0]
        alpha = file.scalars.get(alpha_key, float(rank))
        out[stem].append(_plain_entry(index, lora, file.name, sd[up_key], sd[down_key], alpha, None))
    for stem, parts in _iter_lokr_stems(sd):
        consumed.update(parts.values())
        sides = []
        dims = []
        for direct, a_suf, b_suf in ((".lokr_w1", ".lokr_w1_a", ".lokr_w1_b"), (".lokr_w2", ".lokr_w2_a", ".lokr_w2_b")):
            if direct in parts:
                t = sd[parts[direct]]
                sides.append(_Side((parts[direct],), tuple(t.shape)))
                dims.append(None)
            elif a_suf in parts and b_suf in parts:
                a, b = sd[parts[a_suf]], sd[parts[b_suf]]
                sides.append(_Side((parts[a_suf], parts[b_suf]), (a.shape[0], b.shape[1])))
                dims.append(b.shape[0])
            else:
                raise MergeError(f"{file.name}: incomplete LoKr tensors for {stem}")
        alpha_key = f"{stem}.alpha"
        consumed.add(alpha_key)
        dim = dims[0] if dims[0] is not None else dims[1]
        alpha_scale = file.scalars[alpha_key] / dim if alpha_key in file.scalars and dim is not None else 1.0
        out[stem].append(_Entry(
            lora=index, strength=lora.strength, source=file.name, alpha=alpha_scale, plain=False,
            w1=sides[0], w2=sides[1],
        ))
    leftover = [k for k in file.shapes if k not in consumed and not k.endswith(".alpha")]
    if leftover:
        raise MergeError(f"{file.name}: unsupported tensors that cannot be merged, e.g. {leftover[0]}")
    return out


def _index_model_mode(index: int, lora: LoraInput, file: _LoraFile, sd, skeleton: Skeleton) -> dict[str, list[_Entry]]:
    mapped, unmatched = map_lora_keys(sd, skeleton.module)
    if unmatched:
        raise MergeError(
            f"{file.name}: {len(unmatched)} key(s) do not match the {skeleton.family} model "
            f"(wrong family or shape), e.g. {unmatched[0]}"
        )
    out: dict[str, list[_Entry]] = {}
    for param, deltas in mapped.items():
        stem = param[: -len(".weight")]
        entries = []
        for d in deltas:
            if d.kron:
                entries.append(_lokr_entry(index, lora, file.name, d))
            else:
                row_slice = (d.target_slice[1], d.target_slice[2]) if d.target_slice is not None else None
                entries.append(_plain_entry(index, lora, file.name, d.up, d.down, d.alpha, row_slice))
        out[stem] = entries
    return out


def _lokr_final_rank(entry: _Entry, lokr_rank: int) -> int:
    products = min(entry.w1.shape) * min(entry.w2.shape)
    return max(1, min(lokr_rank, products))


class _Loader:
    def __init__(self, paths: Sequence[str]) -> None:
        self.paths = paths
        self.handles: list = []

    def __enter__(self):
        self.handles = [safe_open(p, framework="pt", device="cpu") for p in self.paths]
        return self

    def __exit__(self, *exc) -> None:
        for h in self.handles:
            h.__exit__(None, None, None)
        self.handles = []

    def side(self, entry: _Entry, side: _Side) -> torch.Tensor:
        h = self.handles[entry.lora]
        tensors = [h.get_tensor(k).to(torch.float32) for k in side.keys]
        return tensors[0] if len(tensors) == 1 else tensors[0] @ tensors[1]


def _stem_dims(stem: str, entries: list[_Entry], full_rows: int | None) -> tuple[int, int]:
    outs = {(e.out_dim if e.row_slice is None else full_rows) for e in entries}
    ins = {e.in_dim for e in entries}
    if len(outs) != 1 or len(ins) != 1:
        names = ", ".join(sorted({e.source for e in entries}))
        raise MergeError(f"{stem}: shape mismatch between {names} (in {sorted(ins)}, out {sorted(outs)})")
    return outs.pop(), ins.pop()


def _build_stem(
    loader: _Loader, entries: list[_Entry], full_rows: int | None, mode: str, rank: int | None, lokr_rank: int | None
) -> tuple[torch.Tensor, torch.Tensor, float | None, list[float]]:
    ups, downs, lokr_energies = [], [], []
    for e in entries:
        if e.plain:
            r = e.rank
            eff = e.strength * e.alpha / r
            up = loader.side(e, e.up) * eff
            down = loader.side(e, e.down)
            if e.row_slice is not None:
                start, length = e.row_slice
                padded = torch.zeros(full_rows, up.shape[1], dtype=up.dtype)
                padded[start:start + length] = up
                up = padded
        else:
            up, down, kept = kron_to_lora(loader.side(e, e.w1), loader.side(e, e.w2), e.strength * e.alpha, lokr_rank)
            lokr_energies.append(kept)
        ups.append(up)
        downs.append(down)
    up = torch.cat(ups, dim=1)
    down = torch.cat(downs, dim=0)
    energy = None
    if mode == "reduce":
        if up.shape[1] > rank:
            up, down, energy = _reduce(up, down, rank)
        else:
            energy = 1.0
    return up, down, energy, lokr_energies


def _final_rank(entries: list[_Entry], mode: str, rank: int | None, lokr_rank: int | None, out_dim: int, in_dim: int) -> int:
    total = sum(e.rank if e.plain else _lokr_final_rank(e, lokr_rank) for e in entries)
    if mode == "reduce" and total > rank:
        return max(1, min(rank, min(out_dim, total), min(in_dim, total)))
    return total


def merge_loras(
    loras: Sequence[LoraInput],
    mode: str = "concat",
    rank: int | None = None,
    min_energy: float = 0.9,
    *,
    dtype: str = "F32",
    family: str | None = None,
    model_path: str | Path | None = None,
) -> tuple[list[TensorSpec], Iterator[tuple[str, torch.Tensor]], MergeReport]:
    if mode not in MODES:
        raise MergeError(f"unknown merge mode {mode!r} (expected one of {', '.join(MODES)})")
    if dtype not in OUTPUT_DTYPES:
        raise MergeError(f"unsupported output dtype {dtype!r}")
    if not loras:
        raise MergeError("no LoRA was given")
    if mode == "reduce" and (rank is None or rank < 1):
        raise MergeError("reduce mode needs a target rank")
    if rank is not None and rank < 1:
        raise MergeError("rank must be positive")

    skeleton: Skeleton | None = None
    if model_path is not None:
        plan_in, metadata_in = read_header(model_path)
        skeleton = build_skeleton(plan_in, metadata_in, family)
    wanted_coarse = {_SPEC_TO_COARSE[f] for f in resolve_family(family)} if family else None
    detected_family = _SPEC_TO_COARSE[skeleton.family] if skeleton else None

    index: dict[str, list[_Entry]] = defaultdict(list)
    per_lora_stems: dict[str, set[str]] = {}
    coarse_seen: dict[str, str] = {}
    per_lora: dict[str, dict[str, int]] = {}
    for i, lora in enumerate(loras):
        file = _LoraFile(lora.path)
        refusal = refusal_for_unsupported_kind(file.name, file.shapes)
        if refusal is not None:
            raise MergeError(refusal)
        sd = file.meta_state(i)
        te = [s for s, _, _ in _iter_stems(sd) if is_text_encoder_stem(s)] + [
            s for s, _ in _iter_lokr_stems(sd) if is_text_encoder_stem(s)
        ]
        if te:
            raise MergeError(f"{file.name}: text-encoder LoRA keys are not supported ({len(te)} found, e.g. {te[0]})")
        if skeleton is not None:
            entries = _index_model_mode(i, lora, file, sd, skeleton)
        else:
            guess = detect_lora_family(sd)
            if guess is None:
                raise MergeError(f"{file.name}: cannot identify which model family this LoRA targets")
            if wanted_coarse is not None and guess not in wanted_coarse:
                raise MergeError(f"{file.name}: targets {guess} but family {family!r} was requested")
            coarse_seen[file.name] = guess
            entries = _index_stem_mode(i, lora, file, sd)
            per_lora_stems[file.name] = set(entries)
        per_lora[lora.path] = {
            "layers": len(entries),
            "lokr_layers": sum(1 for group in entries.values() for e in group if not e.plain),
        }
        for stem, group in entries.items():
            index[stem].extend(group)
    if len(set(coarse_seen.values())) > 1:
        detail = ", ".join(f"{n}: {f}" for n, f in coarse_seen.items())
        raise MergeError(f"LoRAs target different model families ({detail})")
    if skeleton is None:
        _check_dialects(per_lora_stems)
    resolved_family = detected_family or next(iter(coarse_seen.values()), None)

    plain_ranks = [e.rank for group in index.values() for e in group if e.plain]
    has_lokr = any(not e.plain for group in index.values() for e in group)
    lokr_rank = rank if rank is not None else (max(plain_ranks) if plain_ranks else None)
    warnings: list[str] = []
    if has_lokr:
        if lokr_rank is None:
            raise MergeError("LoKr inputs need an explicit rank: there is no LoRA input to take a default rank from")
        if rank is None:
            warnings.append(f"LoKr inputs converted by SVD at rank {lokr_rank}, the rank of the largest LoRA input")

    torch_dtype = DTYPE_TO_TORCH[dtype]
    stems = sorted(index)
    full_rows_of = {
        stem: (skeleton.param_shape(f"{stem}.weight")[0] if skeleton is not None else None) for stem in stems
    }
    dims = {stem: _stem_dims(stem, index[stem], full_rows_of[stem]) for stem in stems}
    ranks = {stem: _final_rank(index[stem], mode, rank, lokr_rank, *dims[stem]) for stem in stems}
    out_key = {stem: (f"{KEY_PREFIX}{stem}" if skeleton is not None else stem) for stem in stems}

    energy: dict[str, float] = {}
    lokr_energy: dict[str, float] = {}
    paths = [l.path for l in loras]
    check = [s for s in stems if mode == "reduce" or any(not e.plain for e in index[s])]
    with _Loader(paths) as loader:
        for stem in check:
            _, _, kept, lokr_kept = _build_stem(loader, index[stem], full_rows_of[stem], mode, rank, lokr_rank)
            if kept is not None:
                energy[out_key[stem]] = kept
            for n, value in enumerate(lokr_kept):
                lokr_energy[f"{out_key[stem]}#{n}"] = value
    low = {k: v for k, v in {**energy, **lokr_energy}.items() if v < min_energy}
    if low:
        worst = sorted(low.items(), key=lambda kv: kv[1])[:5]
        detail = ", ".join(f"{k} {v:.3f}" for k, v in worst)
        raise MergeError(
            f"{len(low)} layer(s) keep less than {min_energy:.2f} of the delta energy at this rank "
            f"(worst: {detail}); raise the rank or lower min_energy"
        )

    plan: list[TensorSpec] = []
    for stem in stems:
        out_dim, in_dim = dims[stem]
        r = ranks[stem]
        plan.append(TensorSpec(f"{out_key[stem]}.lora_up.weight", dtype, (out_dim, r)))
        plan.append(TensorSpec(f"{out_key[stem]}.lora_down.weight", dtype, (r, in_dim)))
        plan.append(TensorSpec(f"{out_key[stem]}.alpha", "F32", ()))

    report = MergeReport(
        mode=mode,
        family=resolved_family,
        key_format=(f"{KEY_PREFIX}<model module path>.lora_up|lora_down.weight + .alpha" if skeleton is not None
                    else "input key spelling, .lora_up|lora_down.weight + .alpha"),
        rank=max(ranks.values(), default=None),
        layers=len(stems),
        output_dtype=dtype,
        energy=energy,
        lokr_energy=lokr_energy,
        per_lora=per_lora,
        warnings=warnings,
        metadata={
            "potionui_tool": f"merge_loras:{mode}",
            "loras": json.dumps([{"name": Path(l.path).name, "strength": l.strength} for l in loras]),
        },
    )

    def generate() -> Iterator[tuple[str, torch.Tensor]]:
        with _Loader(paths) as loader:
            for stem in stems:
                up, down, _, _ = _build_stem(loader, index[stem], full_rows_of[stem], mode, rank, lokr_rank)
                key = out_key[stem]
                yield f"{key}.lora_up.weight", up.to(torch_dtype).contiguous()
                yield f"{key}.lora_down.weight", down.to(torch_dtype).contiguous()
                yield f"{key}.alpha", torch.tensor(float(up.shape[1]), dtype=torch.float32)

    return plan, generate(), report
