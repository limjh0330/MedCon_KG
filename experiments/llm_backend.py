"""LLM and embedding backends.

LocalLLM is a thin HuggingFace `transformers` wrapper for local instruction
models with deterministic decoding and JSON-safe parsing. It supports both
plain CausalLM checkpoints and PEFT LoRA adapters.

OpenAIEmbedder caches Stage-0 recommendation embeddings to disk (npz). Per-query
embeddings are also cached to amortize re-runs.
"""

import hashlib
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# LLM interface
# ──────────────────────────────────────────────────────────────────────

class BaseLLM(ABC):
    name: str = "base"

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        json_mode: bool = False,
    ) -> str: ...

    def generate_batch(
        self,
        batch_messages: list[list[dict]],
        max_new_tokens: int = 512,
        json_mode: bool = False,
    ) -> list[str]:
        """Default fallback: serial. Override for batched inference."""
        return [
            self.generate(m, max_new_tokens=max_new_tokens, json_mode=json_mode)
            for m in batch_messages
        ]


# ──────────────────────────────────────────────────────────────────────
# Local instruction-model backend (HF transformers)
# ──────────────────────────────────────────────────────────────────────

class LocalLLM(BaseLLM):
    """Instruction-tuned local models via HuggingFace transformers."""

    JSON_INSTRUCTION = (
        "Respond with a single valid JSON object. "
        "Do not include any text outside the JSON. Do not use markdown code fences."
    )

    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        attn_impl: str = "sdpa",
        device: str = "auto",
        hf_token: Optional[str] = None,
        deterministic: bool = True,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.deterministic = deterministic
        self._torch = torch

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)

        tok_kwargs = {"token": hf_token} if hf_token else {}
        load_name, base_name, is_peft_adapter = self._resolve_model_names(model_name)

        if is_peft_adapter:
            logger.info(f"Detected PEFT adapter model: {model_name}")
            logger.info(f"Using base model for tokenizer/model load: {base_name}")

        logger.info(f"Loading tokenizer: {load_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(load_name, **tok_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # Decoder-only models need LEFT padding for correct batched generation.
        self.tokenizer.padding_side = "left"

        logger.info(
            f"Loading model: {load_name} (dtype={dtype}, attn={attn_impl}, "
            f"device={device})"
        )
        device_map = device if device != "cpu" else None
        base_model = AutoModelForCausalLM.from_pretrained(
            load_name,
            torch_dtype=torch_dtype,
            device_map=device_map if device != "cpu" else None,
            attn_implementation=attn_impl,
            **tok_kwargs,
        )
        self.model = self._attach_peft_adapter(base_model, model_name, is_peft_adapter)
        self.model.eval()
        self.max_context_tokens = self._resolve_context_window()
        # Keep chat-safe terminators for Llama-style instruct models.
        self.eos_token_ids = self._resolve_eos_tokens()
        logger.info(
            f"Model loaded. context_window={self.max_context_tokens}, "
            f"EOS terminators={self.eos_token_ids}"
        )

    def _resolve_model_names(self, model_name: str) -> tuple[str, str, bool]:
        """Returns (load_name, base_name, is_peft_adapter)."""
        try:
            from peft import PeftConfig
            peft_cfg = PeftConfig.from_pretrained(model_name)
        except Exception:
            return model_name, model_name, False

        base_name = getattr(peft_cfg, "base_model_name_or_path", None) or model_name
        return base_name, base_name, True

    def _attach_peft_adapter(self, base_model, adapter_name: str, enabled: bool):
        if not enabled:
            return base_model
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(
                "`peft` is required to load LoRA adapters. Install with: pip install peft"
            ) from e
        logger.info(f"Attaching PEFT adapter: {adapter_name}")
        return PeftModel.from_pretrained(base_model, adapter_name)

    def _resolve_context_window(self) -> int:
        """Returns usable max sequence length for generation."""
        for src in (getattr(self.model, "config", None), self.tokenizer):
            if src is None:
                continue
            for attr in ("max_position_embeddings", "n_positions", "model_max_length"):
                val = getattr(src, attr, None)
                if isinstance(val, int) and 0 < val < 10_000_000:
                    return val
        return 4096

    def _resolve_eos_tokens(self) -> list[int]:
        ids = []
        if self.tokenizer.eos_token_id is not None:
            ids.append(self.tokenizer.eos_token_id)
        # Llama-style instruct models also terminate on <|eot_id|>
        for special in ("<|eot_id|>", "<|end_of_text|>"):
            tid = self.tokenizer.convert_tokens_to_ids(special)
            if isinstance(tid, int) and tid >= 0 and tid not in ids:
                ids.append(tid)
        return ids

    def _format_messages(self, messages: list[dict], json_mode: bool) -> list[dict]:
        """Inject JSON instruction into the system message when requested."""
        if not json_mode:
            return messages
        formatted = []
        injected = False
        for m in messages:
            if m.get("role") == "system" and not injected:
                formatted.append({
                    "role": "system",
                    "content": m["content"].rstrip() + "\n\n" + self.JSON_INSTRUCTION,
                })
                injected = True
            else:
                formatted.append(dict(m))
        if not injected:
            formatted.insert(0, {"role": "system", "content": self.JSON_INSTRUCTION})
        return formatted

    def _build_prompt(self, messages: list[dict], json_mode: bool) -> str:
        msgs = self._format_messages(messages, json_mode)
        return self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        json_mode: bool = False,
    ) -> str:
        return self.generate_batch([messages], max_new_tokens, json_mode)[0]

    # ── Token counting (used by runners to log retrieval + answer tokens) ──

    def count_tokens(self, text: str) -> int:
        """Number of tokens in plain text (no chat-template framing)."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_prompt_tokens(
        self,
        messages: list[dict],
        json_mode: bool = False,
    ) -> int:
        """Token count of the exact chat-templated prompt that .generate() sees.

        Mirrors `_build_prompt + tokenizer(..., add_special_tokens=False)` so the
        number matches the input length actually fed to model.generate.
        """
        prompt = self._build_prompt(messages, json_mode)
        return len(self.tokenizer.encode(prompt, add_special_tokens=False))

    def generate_batch(
        self,
        batch_messages: list[list[dict]],
        max_new_tokens: int = 512,
        json_mode: bool = False,
    ) -> list[str]:
        if not batch_messages:
            return []

        torch = self._torch
        prompts = [self._build_prompt(m, json_mode) for m in batch_messages]

        safe_max_new_tokens = max_new_tokens
        if safe_max_new_tokens >= self.max_context_tokens:
            safe_max_new_tokens = max(1, self.max_context_tokens - 1)
            logger.warning(
                "Requested max_new_tokens=%s exceeds context window=%s; clamped to %s",
                max_new_tokens,
                self.max_context_tokens,
                safe_max_new_tokens,
            )

        max_input_tokens = max(1, self.max_context_tokens - safe_max_new_tokens)
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
            add_special_tokens=False,
        ).to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": safe_max_new_tokens,
            "eos_token_id": self.eos_token_ids,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
        }
        if self.deterministic:
            gen_kwargs.update(do_sample=False, temperature=1.0, top_p=1.0)
        else:
            gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.95)

        with torch.inference_mode():
            out = self.model.generate(**enc, **gen_kwargs)

        # Strip the prompt prefix from each output row.
        outputs: list[str] = []
        input_lens = enc["input_ids"].shape[1]
        for row in out:
            gen_tokens = row[input_lens:].tolist()
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            outputs.append(text)
        return outputs


# ──────────────────────────────────────────────────────────────────────
# JSON-safe parsing helpers
# ──────────────────────────────────────────────────────────────────────

_FENCE_PREFIX_RE = re.compile(r"^```(?:json)?\s*")
_FENCE_SUFFIX_RE = re.compile(r"\s*```$")


def strip_to_json(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = _FENCE_PREFIX_RE.sub("", t)
    t = _FENCE_SUFFIX_RE.sub("", t).strip()
    return t


def _repair_truncated_json(text: str) -> str:
    """Close unclosed brackets/braces to recover JSON cut off mid-output."""
    t = text.rstrip()
    # Strip trailing comma or partial token that prevents parsing
    t = re.sub(r',\s*$', '', t)
    stack = []
    in_string = False
    escape_next = False
    for ch in t:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append('}' if ch == '{' else ']')
        elif ch in ('}', ']'):
            if stack and stack[-1] == ch:
                stack.pop()
    return t + ''.join(reversed(stack))


def parse_json_object(text: str) -> Optional[dict]:
    """Robust JSON object parse.

    Tries in order:
    1. Direct parse after stripping fences.
    2. First {…} greedy match.
    3. Truncation repair (close unclosed brackets) then parse.
    """
    t = strip_to_json(text)
    if not t:
        return None
    # 1. Direct
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # 2. First {…} match
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    # 3. Repair truncated JSON
    repaired = _repair_truncated_json(t)
    if repaired != t:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
        m2 = re.search(r"\{[\s\S]*\}", repaired)
        if m2:
            try:
                return json.loads(m2.group())
            except json.JSONDecodeError:
                pass
    return None


# ──────────────────────────────────────────────────────────────────────
# OpenAI embedder
# ──────────────────────────────────────────────────────────────────────

class OpenAIEmbedder:
    """text-embedding-3-large with simple disk caching."""

    def __init__(
        self,
        api_key: Optional[str],
        model: str = "text-embedding-3-large",
        batch_size: int = 128,
    ):
        if not api_key:
            raise ValueError("OpenAI API key is required for embeddings (Variant 2).")
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "Install OpenAI SDK: pip install 'openai>=1.70.0'"
            ) from e
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return an (N, D) float32 matrix. Input order is preserved."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend(d.embedding for d in resp.data)
        return np.asarray(vectors, dtype=np.float32)

    def embed_with_cache(
        self,
        texts: list[str],
        cache_path: str,
    ) -> np.ndarray:
        """Embed texts; cache the (texts_hash, matrix) bundle to npz.

        If the cached texts_hash matches, return the cached matrix instead of
        re-calling the API. The hash is over the concatenated SHA256 of texts so
        any change to the input invalidates the cache.
        """
        sig = self._sig(texts)
        if os.path.isfile(cache_path):
            try:
                cached = np.load(cache_path, allow_pickle=False)
                if str(cached["signature"]) == sig:
                    logger.info(
                        f"Loaded {cached['matrix'].shape[0]} cached embeddings "
                        f"from {cache_path}"
                    )
                    return cached["matrix"].astype(np.float32)
                logger.info("Embedding cache signature mismatch — recomputing")
            except Exception as e:
                logger.warning(f"Cache read failed ({e}); recomputing embeddings")

        logger.info(f"Computing {len(texts)} embeddings via {self.model}…")
        matrix = self.embed_texts(texts)
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path, matrix=matrix, signature=np.array(sig))
        logger.info(f"Saved embeddings to {cache_path}")
        return matrix

    @staticmethod
    def _sig(texts: list[str]) -> str:
        h = hashlib.sha256()
        h.update(str(len(texts)).encode())
        for t in texts:
            h.update(b"\x00")
            h.update(hashlib.sha256((t or "").encode("utf-8")).digest())
        return h.hexdigest()


def cosine_top_k(query_vec: np.ndarray, matrix: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """Return top-k (index, similarity) pairs by cosine. matrix must be (N, D)."""
    if matrix.size == 0 or query_vec.size == 0:
        return []
    q = query_vec.astype(np.float32)
    qn = np.linalg.norm(q)
    if qn == 0:
        return []
    mn = np.linalg.norm(matrix, axis=1)
    mn = np.where(mn == 0, 1e-12, mn)
    sims = (matrix @ q) / (mn * qn)
    k = min(top_k, sims.shape[0])
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]
