from __future__ import annotations

import io
import zipfile
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from PIL import Image, ImageOps

from services.config import config
from services.image_tags_service import load_tags, remove_tags

THUMBNAIL_SIZE = (320, 320)
DEFAULT_IMAGE_PAGE_SIZE = 12
MAX_IMAGE_PAGE_SIZE = 100
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def _cleanup_empty_dirs(root: Path) -> None:
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def _safe_relative_path(path: str) -> str:
    value = str(path or "").strip().replace("\\", "/").lstrip("/")
    if not value:
        raise HTTPException(status_code=404, detail="image not found")
    parts = Path(value).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise HTTPException(status_code=404, detail="image not found")
    return Path(*parts).as_posix()


def _safe_image_path(relative_path: str) -> Path:
    rel = _safe_relative_path(relative_path)
    root = config.images_dir.resolve()
    path = (root / rel).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="image not found") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    return path


def _thumbnail_path(relative_path: str) -> Path:
    rel = _safe_relative_path(relative_path)
    return config.image_thumbnails_dir / f"{rel}.png"


def thumbnail_url(base_url: str, relative_path: str) -> str:
    return f"{base_url.rstrip('/')}/image-thumbnails/{_safe_relative_path(relative_path)}"


def _image_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            dimensions = image.size
            image.verify()
            return dimensions
    except Exception:
        return None


def ensure_thumbnail(relative_path: str) -> Path:
    source = _safe_image_path(relative_path)
    target = _thumbnail_path(relative_path)
    source_mtime = source.stat().st_mtime
    if target.exists() and target.stat().st_mtime >= source_mtime:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
            image.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            image.save(target, format="PNG", optimize=True)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail="failed to create thumbnail") from exc
    return target


def get_thumbnail_response(relative_path: str) -> FileResponse:
    return FileResponse(ensure_thumbnail(relative_path))


def get_image_download_response(relative_path: str) -> FileResponse:
    path = _safe_image_path(relative_path)
    return FileResponse(path, filename=path.name)


def cleanup_image_thumbnails() -> int:
    thumbnails_root = config.image_thumbnails_dir
    images_root = config.images_dir
    removed = 0
    for path in thumbnails_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(thumbnails_root).as_posix()
        if not rel.endswith(".png") or not (images_root / rel[:-4]).exists():
            path.unlink()
            removed += 1
    _cleanup_empty_dirs(thumbnails_root)
    return removed


def _has_image_signature(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            header = file.read(16)
    except OSError:
        return False
    suffix = path.suffix.lower()
    if suffix == ".png":
        return header.startswith(b"\x89PNG\r\n\x1a\n")
    if suffix in {".jpg", ".jpeg"}:
        return header.startswith(b"\xff\xd8\xff")
    if suffix == ".gif":
        return header.startswith((b"GIF87a", b"GIF89a"))
    if suffix == ".webp":
        return header.startswith(b"RIFF") and header[8:12] == b"WEBP"
    return False


def _image_records() -> list[dict[str, object]]:
    items = []
    root = config.images_dir
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if not _has_image_signature(path):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        rel = path.relative_to(root).as_posix()
        parts = rel.split("/")
        day = "-".join(parts[:3]) if len(parts) >= 4 else datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d")
        items.append({
            "rel": rel,
            "path": rel,
            "name": path.name,
            "date": day,
            "size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "mtime": stat.st_mtime,
        })
    items.sort(key=lambda item: (float(item["mtime"]), str(item["rel"])), reverse=True)
    return items


def _image_items(start_date: str = "", end_date: str = "") -> list[dict[str, object]]:
    items = []
    for record in _filter_records_by_date(_image_records(), start_date, end_date):
        public_item = _public_image_item(record, "", {})
        if public_item is not None:
            public_item.pop("url", None)
            public_item.pop("thumbnail_url", None)
            public_item.pop("tags", None)
            items.append(public_item)
    return items


def _filter_records_by_date(
    records: list[dict[str, object]],
    start_date: str = "",
    end_date: str = "",
) -> list[dict[str, object]]:
    return [
        item
        for item in records
        if (not start_date or str(item["date"]) >= start_date)
        and (not end_date or str(item["date"]) <= end_date)
    ]


def _normalize_tags(tags: list[str] | None = None) -> list[str]:
    return list(dict.fromkeys(str(tag or "").strip() for tag in tags or [] if str(tag or "").strip()))


def _filter_records_by_tags(
    records: list[dict[str, object]],
    all_tags: dict[str, list[str]],
    selected_tags: list[str],
) -> list[dict[str, object]]:
    if not selected_tags:
        return records
    return [
        item
        for item in records
        if all(tag in all_tags.get(str(item["path"]), []) for tag in selected_tags)
    ]


def _paginate_records(
    records: list[dict[str, object]],
    page: int = 1,
    page_size: int = DEFAULT_IMAGE_PAGE_SIZE,
) -> tuple[list[dict[str, object]], int, int, int, int]:
    page_size = min(MAX_IMAGE_PAGE_SIZE, max(1, int(page_size or DEFAULT_IMAGE_PAGE_SIZE)))
    total = len(records)
    pages = max(1, (total + page_size - 1) // page_size)
    page = min(max(1, int(page or 1)), pages)
    start = (page - 1) * page_size
    return records[start : start + page_size], total, page, page_size, pages


def _build_summary(
    all_records: list[dict[str, object]],
    date_records: list[dict[str, object]],
    filtered_records: list[dict[str, object]],
) -> dict[str, int]:
    return {
        "total": len(all_records),
        "date_total": len(date_records),
        "filtered": len(filtered_records),
        "size": sum(int(item.get("size") or 0) for item in all_records),
        "date_size": sum(int(item.get("size") or 0) for item in date_records),
        "filtered_size": sum(int(item.get("size") or 0) for item in filtered_records),
    }


def _public_image_item(
    item: dict[str, object],
    base_url: str,
    all_tags: dict[str, list[str]],
) -> dict[str, object] | None:
    rel = str(item["path"])
    dimensions = _image_dimensions(config.images_dir / rel)
    if dimensions is None:
        return None
    public_item = {
        key: value
        for key, value in item.items()
        if key != "mtime"
    }
    public_item.update({
        "width": dimensions[0],
        "height": dimensions[1],
    })
    if base_url:
        public_item.update({
            "url": f"{base_url.rstrip('/')}/images/{item['path']}",
            "thumbnail_url": thumbnail_url(base_url, rel),
            "tags": all_tags.get(rel, []),
        })
    return public_item


def list_images(
    base_url: str,
    start_date: str = "",
    end_date: str = "",
    page: int = 1,
    page_size: int = DEFAULT_IMAGE_PAGE_SIZE,
    tags: list[str] | None = None,
) -> dict[str, object]:
    config.cleanup_old_images()
    all_tags = load_tags()
    selected_tags = _normalize_tags(tags)
    all_records = _image_records()
    date_records = _filter_records_by_date(all_records, start_date, end_date)
    filtered_records = _filter_records_by_tags(date_records, all_tags, selected_tags)
    page_records, total, page, page_size, pages = _paginate_records(filtered_records, page, page_size)
    items = [
        public_item
        for item in page_records
        if (public_item := _public_image_item(item, base_url, all_tags)) is not None
    ]
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": pages,
        "summary": _build_summary(all_records, date_records, filtered_records),
        "selected_tags": selected_tags,
    }


def delete_images(paths: list[str] | None = None, start_date: str = "", end_date: str = "", all_matching: bool = False) -> dict[str, int]:
    root = config.images_dir.resolve()
    targets = [str(item["path"]) for item in _image_items(start_date, end_date)] if all_matching else (paths or [])
    removed = 0
    for item in targets:
        path = (root / item).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if path.is_file():
            path.unlink()
            for thumbnail in (_thumbnail_path(item), config.image_thumbnails_dir / _safe_relative_path(item)):
                if thumbnail.is_file():
                    thumbnail.unlink()
            remove_tags(item)
            removed += 1
    _cleanup_empty_dirs(root)
    _cleanup_empty_dirs(config.image_thumbnails_dir)
    return {"removed": removed}


def download_images_zip(paths: list[str]) -> io.BytesIO:
    root = config.images_dir.resolve()
    buf = io.BytesIO()
    added = 0
    used_names: set[str] = set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in paths:
            rel = _safe_relative_path(item)
            path = (root / rel).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                continue
            if not path.is_file():
                continue
            name = path.name
            if name in used_names:
                stem = path.stem
                suffix = path.suffix
                counter = 2
                while f"{stem}_{counter}{suffix}" in used_names:
                    counter += 1
                name = f"{stem}_{counter}{suffix}"
            used_names.add(name)
            zf.write(path, name)
            added += 1
    if added == 0:
        raise HTTPException(status_code=404, detail="no images found")
    buf.seek(0)
    return buf
