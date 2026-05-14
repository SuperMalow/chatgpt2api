from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageBackendSpec:
    name: str
    use_legacy_prompt_hint: bool = False
    slug_mode: str = "native"
