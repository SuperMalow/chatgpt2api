from __future__ import annotations

import io
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services import image_service
from services.openai_backend_api import OpenAIBackendAPI
from services.protocol import conversation
from PIL import Image


def png_1x1() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), (255, 255, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeImageConfig:
    def __init__(self, root: Path) -> None:
        self.images_dir = root / "images"
        self.image_thumbnails_dir = root / "image_thumbnails"
        self.base_url = "http://testserver"
        self.images_dir.mkdir(parents=True, exist_ok=False)
        self.image_thumbnails_dir.mkdir(parents=True, exist_ok=False)

    def cleanup_old_images(self) -> int:
        return 0


def make_root() -> Path:
    root = Path("/private/tmp") / f"chatgpt2api-image-validation-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    return root


class ImageFileValidationTests(unittest.TestCase):
    def test_image_listing_skips_files_that_are_not_real_images(self) -> None:
        config = FakeImageConfig(make_root())
        day_dir = config.images_dir / "2026" / "05" / "15"
        day_dir.mkdir(parents=True, exist_ok=False)
        (day_dir / "valid.png").write_bytes(png_1x1())
        (day_dir / "invalid.png").write_bytes(b"fake image bytes")

        with (
            mock.patch.object(image_service, "config", config),
            mock.patch.object(image_service, "load_tags", return_value={}),
        ):
            result = image_service.list_images("http://testserver")

        self.assertEqual([item["name"] for item in result["items"]], ["valid.png"])

    def test_save_image_bytes_rejects_non_image_bytes(self) -> None:
        config = FakeImageConfig(make_root())

        with mock.patch.object(conversation, "config", config):
            with self.assertRaises(ValueError):
                conversation.save_image_bytes(b"fake image bytes", "http://testserver")

    def test_download_image_bytes_rejects_non_image_response_body(self) -> None:
        class FakeResponse:
            status_code = 200
            content = b"fake image bytes"
            text = "fake image bytes"

            def json(self) -> dict[str, object]:
                return {}

        backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
        backend._request_with_local_retry = mock.Mock(return_value=FakeResponse())

        with self.assertRaisesRegex(RuntimeError, "non-image"):
            backend.download_image_bytes(["https://example.test/image.png"])


if __name__ == "__main__":
    unittest.main()
