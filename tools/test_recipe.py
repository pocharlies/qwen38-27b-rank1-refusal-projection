#!/usr/bin/env python3
"""Guards on the sparkrun recipe, written after breaking it twice for the same person.

    python3 tools/test_recipe.py

Both failures had the same shape: the recipe was not what production runs, and the
only symptom the user got was sparkrun's

    Error: Server health check never passed at http://127.0.0.1:8000/v1/models

which points at the network or the image and is neither. No dependencies, so this
runs anywhere the repo is checked out.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECIPES = [ROOT / "recipes" / "qwen38-27b-nvfp4-refusal-dial.yaml",
           ROOT / "hf" / "qwen38-27b-nvfp4-refusal-dial.yaml"]

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


def command_block(text: str) -> str:
    """The literal block under `command:`, comments and all removed."""
    m = re.search(r"^command: \|\n((?:[ \t].*\n|\n)+)", text, re.M)
    return m.group(1) if m else ""


for path in RECIPES:
    tag = path.parent.name + "/"
    text = path.read_text(encoding="utf-8")
    body = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    cmd = command_block(text)

    check(bool(cmd), f"{tag}: no encuentro el bloque `command:`")

    # 1 -- A {placeholder} inside a JSON literal is unreachable.
    #
    # sparkrun substitutes with vpd's `\{(.*?)\}`: non-greedy and brace-blind, so in
    #     '{"method":"mtp","num_speculative_tokens":{spec_tokens}}'
    # the first `{` pairs with the first `}` and the whole thing is read as ONE key.
    # No default has that name, the text is put back verbatim, and vLLM receives a
    # literal `{spec_tokens}` -> JSONDecodeError during argparse, before it binds the
    # port. recipe.validate() returns [] either way.
    for line in cmd.splitlines():
        for blob in re.findall(r"'(\{.*?\})'", line):
            inner = re.findall(r"\{([a-z_][a-z0-9_]*)\}", blob[1:-1])
            check(
                not inner,
                f"{tag}: {inner} esta dentro de un literal JSON y sparkrun NUNCA lo "
                f"sustituira -> vLLM recibe la llave literal. Mete el JSON entero en "
                f"un default y sustituye el bloque completo: {line.strip()}",
            )

    # 2 -- Triton has to be named TWICE.
    #
    # The Qwen3.5 MTP drafter builds its own attention selector and ignores the
    # target's --attention-backend. Left alone it picks FlashInfer and dies on the
    # first prefill with the 19-vs-20 argument `plan()` ABI mismatch, which happens
    # during startup warmup -- so the server simply never becomes ready.
    check(
        "--attention-backend triton_attn" in cmd,
        f"{tag}: falta --attention-backend triton_attn para el modelo target",
    )
    spec = re.search(r"speculative_config:\s*'([^']+)'", body)
    check(bool(spec), f"{tag}: no encuentro el default speculative_config")
    if spec:
        blob = spec.group(1)
        check(
            '"attention_backend":"triton_attn"' in blob.replace(" ", ""),
            f"{tag}: --speculative-config no fuerza triton_attn en el drafter MTP. "
            f"Sin eso el drafter coge FlashInfer y el arranque muere sin llegar a "
            f"servir: {blob}",
        )
        import json
        try:
            json.loads(blob)
        except Exception as e:  # noqa: BLE001
            check(False, f"{tag}: speculative_config no es JSON valido: {e}")

    # 3 -- It boots censored. Uncensored is something you turn on.
    check(
        re.search(r'VLLM_REFUSAL_LAMBDA_INIT:\s*"0\.0"', body) is not None,
        f"{tag}: VLLM_REFUSAL_LAMBDA_INIT deberia ser 0.0 -- el dial se enciende, "
        f"no se hereda",
    )

# The two copies must not drift: hf/ is the one people wget.
if all(p.exists() for p in RECIPES):
    a, b = (p.read_text(encoding="utf-8") for p in RECIPES)
    check(a == b, "recipes/ y hf/ han divergido; hf/ es la que la gente se baja")

if failures:
    print("[recipe] FALLA:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print(f"[recipe] OK — {len(RECIPES)} ficheros, sin placeholders atrapados en JSON, "
      f"triton en target y drafter")
