from __future__ import annotations

import re
from typing import Any, Optional, Tuple, Union

import numpy as np
from PIL import Image

from trident.wsi_objects.WSI import ReadMode, WSI


class FastSlideWSI(WSI):
    """
    WSI implementation backed by fastslide.

    FastSlide's Python API reads regions using coordinates native to the
    requested pyramid level. TRIDENT and OpenSlide use level-0 coordinates for
    all levels, so this adapter converts coordinates before every read.
    """

    _MPP_KEYS = (
        "mpp_x",
        "mpp-y",
        "mpp-x",
        "mpp_y",
        "mpp",
        "microns_per_pixel",
        "openslide.mpp-x",
        "openslide.mpp-y",
        "aperio.MPP",
    )
    _MAG_KEYS = (
        "objective_magnification",
        "objective-power",
        "objective_power",
        "openslide.objective-power",
        "aperio.AppMag",
    )

    def __init__(self, slide_path: str, apply_icc: bool = False, **kwargs: Any) -> None:
        self.apply_icc = apply_icc
        # FastSlide is already fast in-process, and TRIDENT's default worker
        # heuristic can create many DataLoader workers per GPU process. Those
        # workers receive initialized native reader state during segmentation
        # and feature extraction, which can hang or oversubscribe CPU/I/O.
        if kwargs.get("max_workers") is None:
            kwargs["max_workers"] = 0
        super().__init__(slide_path, **kwargs)

    def _lazy_initialize(self) -> None:
        super()._lazy_initialize()

        if self._initialized:
            return

        try:
            import fastslide
        except ImportError as e:
            raise ImportError(
                "fastslide is required for the `fastslide` reader. "
                "Build and install the FastSlide wheel before using reader_type='fastslide'."
            ) from e

        try:
            self.img = fastslide.FastSlide.from_file_path(self.slide_path, apply_icc=self.apply_icc)
            self.dimensions = self.get_dimensions()
            self.width, self.height = self.dimensions
            self.level_count = int(self.img.level_count)
            self.level_downsamples = tuple(float(d) for d in self.img.level_downsamples)
            self.level_dimensions = tuple((int(w), int(h)) for w, h in self.img.level_dimensions)
            self.properties = dict(getattr(self.img, "properties", {}) or {})
            if self.mpp is None:
                self.mpp = self._fetch_mpp(self.custom_mpp_keys)
            self.mag = self._fetch_magnification(self.custom_mpp_keys)
            self._initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize WSI with FastSlide: {e}") from e

    @staticmethod
    def _try_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            if isinstance(value, str):
                match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
                if match is not None:
                    try:
                        return float(match.group(0))
                    except ValueError:
                        return None
            return None

    def _fetch_mpp(self, custom_mpp_keys: Optional[list[str]] = None) -> float:
        mpp = getattr(self.img, "mpp", None)
        if mpp:
            mpp_values = [self._try_float(value) for value in mpp]
            mpp_values = [value for value in mpp_values if value is not None and value > 0]
            if mpp_values:
                return round(float(sum(mpp_values) / len(mpp_values)), 4)

        mpp_keys = list(self._MPP_KEYS)
        if custom_mpp_keys:
            mpp_keys.extend(custom_mpp_keys)

        for key in mpp_keys:
            if key in self.properties:
                parsed = self._try_float(self.properties[key])
                if parsed is not None and parsed > 0:
                    return round(parsed, 4)

        for key, value in self.properties.items():
            key_lower = str(key).lower()
            if "mpp" not in key_lower and "micron" not in key_lower:
                continue
            parsed = self._try_float(value)
            if parsed is not None and parsed > 0:
                return round(parsed, 4)

        raise ValueError(
            f"Unable to extract MPP from slide metadata: '{self.slide_path}'. "
            "Provide `mpp` explicitly or pass custom MPP metadata keys."
        )

    def _fetch_magnification(self, custom_mpp_keys: Optional[list[str]] = None) -> int:
        mag = super()._fetch_magnification(custom_mpp_keys)
        if mag is not None:
            return mag

        for key in self._MAG_KEYS:
            if key in self.properties:
                parsed = self._try_float(self.properties[key])
                if parsed is not None:
                    return int(parsed)

        raise ValueError(f"Unable to determine magnification from metadata for: {self.slide_path}")

    def _normalize_region(self, region: Any, size: Tuple[int, int]) -> Image.Image:
        if hasattr(region, "numpy"):
            array = np.asarray(region.numpy())
        elif isinstance(region, Image.Image):
            return region.convert("RGB")
        else:
            array = np.asarray(region)

        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        elif array.ndim == 3 and array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        elif array.ndim == 3 and array.shape[2] > 3:
            array = array[:, :, :3]

        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        image = Image.fromarray(array).convert("RGB")
        if image.size != size:
            image = image.resize(size)
        return image

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        read_as: ReadMode = "pil",
    ) -> Union[Image.Image, np.ndarray]:
        self._lazy_initialize()

        if level < 0 or level >= self.level_count:
            raise ValueError(f"Invalid pyramid level {level}; slide has {self.level_count} levels.")

        if level == 0:
            native_location = (int(location[0]), int(location[1]))
        else:
            native_location = tuple(
                int(v) for v in self.img.convert_level0_to_level_native(
                    int(location[0]), int(location[1]), int(level)
                )
            )

        requested_size = (int(size[0]), int(size[1]))
        level_w, level_h = self.level_dimensions[level]
        read_w = max(0, min(requested_size[0], level_w - native_location[0]))
        read_h = max(0, min(requested_size[1], level_h - native_location[1]))

        try:
            if read_w == 0 or read_h == 0:
                image = Image.new("RGB", requested_size, (255, 255, 255))
            else:
                region = self.img.read_region(
                    location=native_location,
                    level=int(level),
                    size=(read_w, read_h),
                )
                image = self._normalize_region(region, (read_w, read_h))
                if image.size != requested_size:
                    canvas = Image.new("RGB", requested_size, (255, 255, 255))
                    canvas.paste(image, (0, 0))
                    image = canvas
        except Exception as e:
            raise RuntimeError(
                f"FastSlide failed to read region at level-0 location {location}, "
                f"native location {native_location}, level {level}, size {size}: {e}"
            ) from e

        if read_as == "pil":
            return image
        if read_as == "numpy":
            return np.array(image)
        raise ValueError(f"Invalid `read_as` value: {read_as}. Must be 'pil', 'numpy'.")

    def get_dimensions(self) -> Tuple[int, int]:
        return tuple(int(v) for v in self.img.dimensions)

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        self._lazy_initialize()

        target_w, target_h = int(size[0]), int(size[1])
        if target_w <= 0 or target_h <= 0:
            raise ValueError(f"Invalid thumbnail size: {size}")

        requested_downsample = max(self.width / target_w, self.height / target_h)
        try:
            level = int(self.img.get_best_level_for_downsample(float(requested_downsample)))
        except Exception:
            level = self.get_best_level_and_custom_downsample(float(requested_downsample))[0]

        level_size = tuple(int(v) for v in self.level_dimensions[level])
        region = self.img.read_region(location=(0, 0), level=level, size=level_size)
        thumbnail = self._normalize_region(region, level_size)
        resample_mode = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        thumbnail.thumbnail((target_w, target_h), resample_mode)
        return thumbnail

    def close(self) -> None:
        try:
            if getattr(self, "img", None) is not None and hasattr(self.img, "close"):
                self.img.close()
        except Exception:
            pass
        self.img = None
