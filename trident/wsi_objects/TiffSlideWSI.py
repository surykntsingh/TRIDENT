from __future__ import annotations

import warnings
import re
from typing import Any, Optional, Tuple, Union

import numpy as np
from PIL import Image

from trident.wsi_objects.WSI import ReadMode, WSI


class TiffSlideWSI(WSI):
    """
    WSI implementation backed by tiffslide.
    """

    _MPP_KEYS = (
        "tiffslide.mpp-x",
        "tiffslide.mpp-y",
        "openslide.mpp-x",
        "openslide.mpp-y",
        "aperio.MPP",
        "openslide.mirax.MPP",
        "hamamatsu.XResolution",
        "mpp-x",
        "mpp_y",
        "mpp",
        "microns_per_pixel",
    )
    _MAG_KEYS = (
        "tiffslide.objective-power",
        "openslide.objective-power",
        "objective-power",
        "objective_power",
        "aperio.AppMag",
    )

    def __init__(self, slide_path: str, allow_openslide_fallback: bool = True, **kwargs: Any) -> None:
        self.allow_openslide_fallback = allow_openslide_fallback
        super().__init__(slide_path, **kwargs)

    @staticmethod
    def _is_entirely_black(image: Image.Image) -> bool:
        return not np.any(np.asarray(image))

    def _init_openslide_fallback(self) -> Any:
        if not self.allow_openslide_fallback:
            raise RuntimeError("OpenSlide fallback is disabled for this TiffSlideWSI instance.")
        from openslide import OpenSlide

        return OpenSlide(self.slide_path)

    def _read_region_from_backend(
        self,
        backend: Any,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
    ) -> Image.Image:
        return backend.read_region(location, level, size).convert("RGB")

    def _get_thumbnail_from_backend(self, backend: Any, size: tuple[int, int]) -> Image.Image:
        return backend.get_thumbnail(size).convert("RGB")

    def _lazy_initialize(self) -> None:
        super()._lazy_initialize()

        if self._initialized:
            return

        try:
            from tiffslide import TiffSlide
        except ImportError as e:
            raise ImportError(
                "tiffslide is required for the `tiffslide` reader. "
                "Install it with `pip install tiffslide`."
            ) from e

        self._fallback_img = None
        self.tiffslide_black_thumbnail_detected = False

        try:
            self.img = TiffSlide(self.slide_path)
        except Exception as tiffslide_init_error:
            if not self.allow_openslide_fallback:
                raise RuntimeError(f"Failed to initialize WSI with tiffslide: {tiffslide_init_error}") from tiffslide_init_error
            warnings.warn(
                f"tiffslide failed to initialize '{self.slide_path}': {tiffslide_init_error}. "
                "Falling back to OpenSlide."
            )
            try:
                self.img = self._init_openslide_fallback()
                self._fallback_img = self.img
            except Exception as fallback_init_error:
                raise RuntimeError(f"Failed to initialize WSI with tiffslide: {tiffslide_init_error}") from fallback_init_error

        try:
            self.dimensions = self.get_dimensions()
            self.width, self.height = self.dimensions
            self.level_count = self.img.level_count
            self.level_downsamples = self.img.level_downsamples
            self.level_dimensions = self.img.level_dimensions
            self.properties = self.img.properties
            if self.mpp is None:
                self.mpp = self._fetch_mpp(self.custom_mpp_keys)
            self.mag = self._fetch_magnification(self.custom_mpp_keys)
            self._initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize WSI with tiffslide: {e}") from e

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
        mpp_keys = list(self._MPP_KEYS)
        if custom_mpp_keys:
            mpp_keys.extend(custom_mpp_keys)

        for key in mpp_keys:
            if key in self.img.properties:
                parsed = self._try_float(self.img.properties[key])
                if parsed is not None:
                    return round(parsed, 4)

        for key, value in self.img.properties.items():
            key_lower = str(key).lower()
            if "mpp" not in key_lower and "micron" not in key_lower:
                continue
            parsed = self._try_float(value)
            if parsed is not None:
                return round(parsed, 4)

        x_resolution = self.img.properties.get("tiff.XResolution")
        unit = self.img.properties.get("tiff.ResolutionUnit")
        if x_resolution and unit:
            parsed_xres = self._try_float(x_resolution)
            if parsed_xres and parsed_xres > 0:
                unit_upper = str(unit).upper()
                if unit_upper == "CENTIMETER":
                    return round(10000 / parsed_xres, 4)
                if unit_upper == "INCH":
                    return round(25400 / parsed_xres, 4)

        raise ValueError(
            f"Unable to extract MPP from slide metadata: '{self.slide_path}'.\n"
            "Suggestions:\n"
            "- Provide `custom_mpp_keys` to specify metadata keys to look for.\n"
            "- Set the MPP explicitly via the class constructor.\n"
            "- If using the `run_batch_of_slides.py` script, pass the MPP via the "
            "`--custom_list_of_wsis` argument in a CSV file."
        )

    def _fetch_magnification(self, custom_mpp_keys: Optional[list[str]] = None) -> int:
        mag = super()._fetch_magnification(custom_mpp_keys)
        if mag is not None:
            return mag

        for key in self._MAG_KEYS:
            if key in self.img.properties:
                parsed = self._try_float(self.img.properties[key])
                if parsed is not None:
                    return int(parsed)

        raise ValueError(f"Unable to determine magnification from metadata for: {self.slide_path}")

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        read_as: ReadMode = "pil",
    ) -> Union[Image.Image, np.ndarray]:
        try:
            region = self._read_region_from_backend(self.img, location, level, size)
        except Exception as e:
            if not self.allow_openslide_fallback:
                raise RuntimeError(
                    f"tiffslide failed to read region at {location}, level {level}: {e}"
                ) from e
            warnings.warn(
                f"Corrupt region at {location}, level {level}: {e}. "
                "Attempting OpenSlide fallback."
            )
            try:
                if self._fallback_img is None:
                    self._fallback_img = self._init_openslide_fallback()
                region = self._read_region_from_backend(self._fallback_img, location, level, size)
            except Exception as fallback_e:
                warnings.warn(
                    f"Fallback read failed at {location}, level {level}: {fallback_e}. "
                    "Returning blank white image."
                )
                region = Image.new("RGB", size, (255, 255, 255))

        if read_as == "pil":
            return region
        if read_as == "numpy":
            return np.array(region)
        raise ValueError(f"Invalid `read_as` value: {read_as}. Must be 'pil', 'numpy'.")

    def get_dimensions(self) -> Tuple[int, int]:
        return self.img.dimensions

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        try:
            thumbnail = self._get_thumbnail_from_backend(self.img, size)
            if self._is_entirely_black(thumbnail):
                self.tiffslide_black_thumbnail_detected = True
                raise ValueError("tiffslide returned an entirely black thumbnail")
            return thumbnail
        except Exception as e:
            if not self.allow_openslide_fallback:
                raise RuntimeError(f"tiffslide failed to generate thumbnail: {e}") from e
            warnings.warn(
                f"Thumbnail generation failed for '{self.slide_path}' with tiffslide: {e}. "
                "Falling back to OpenSlide."
            )
            if self._fallback_img is None:
                self._fallback_img = self._init_openslide_fallback()
            thumbnail = self._get_thumbnail_from_backend(self._fallback_img, size)
            if self._is_entirely_black(thumbnail):
                raise ValueError(
                    f"Thumbnail is entirely black for '{self.slide_path}'; both tiffslide and "
                    "OpenSlide returned unreadable image data."
                )
            return thumbnail

    def close(self) -> None:
        backends = []
        for backend in (getattr(self, "_fallback_img", None), getattr(self, "img", None)):
            if backend is not None and all(id(backend) != id(existing) for existing in backends):
                backends.append(backend)

        for backend in backends:
            try:
                if backend is not None and hasattr(backend, "close"):
                    backend.close()
            except Exception:
                pass
        self._fallback_img = None
        self.img = None
