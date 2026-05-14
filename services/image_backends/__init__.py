from __future__ import annotations

from services.config import config
from services.image_backends.base import ImageBackendSpec
from services.image_backends.conversation import CONVERSATION_IMAGE_BACKEND
from services.image_backends.native import NATIVE_IMAGE_BACKEND

IMAGE_BACKENDS: dict[str, ImageBackendSpec] = {
    NATIVE_IMAGE_BACKEND.name: NATIVE_IMAGE_BACKEND,
    CONVERSATION_IMAGE_BACKEND.name: CONVERSATION_IMAGE_BACKEND,
}


def get_image_backend_specs() -> list[ImageBackendSpec]:
    default_name = config.image_backend_default
    fallback_enabled = config.image_backend_fallback_enabled
    primary = IMAGE_BACKENDS.get(default_name, NATIVE_IMAGE_BACKEND)
    if not fallback_enabled:
        return [primary]
    fallback = CONVERSATION_IMAGE_BACKEND if primary.name != CONVERSATION_IMAGE_BACKEND.name else NATIVE_IMAGE_BACKEND
    return [primary, fallback]


def is_image_backend_fallback_error(message: str) -> bool:
    text = str(message or "").lower()
    return any(
        needle in text
        for needle in (
            "upstream image connection failed",
            "image generation finished without image data",
            "unsupported image model",
            "unsupported image backend",
            "model_not_supported",
            "feature_not_supported",
        )
    )
