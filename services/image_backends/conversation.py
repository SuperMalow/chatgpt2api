from __future__ import annotations

from services.image_backends.base import ImageBackendSpec


CONVERSATION_IMAGE_BACKEND = ImageBackendSpec(
    name="conversation",
    use_legacy_prompt_hint=True,
    slug_mode="conversation",
)
