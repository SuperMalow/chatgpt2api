from __future__ import annotations

from services.image_backends.base import ImageBackendSpec


NATIVE_IMAGE_BACKEND = ImageBackendSpec(
    name="native",
    use_legacy_prompt_hint=False,
    slug_mode="native",
)
