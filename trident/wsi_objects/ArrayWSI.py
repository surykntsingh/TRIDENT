from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from PIL import Image

from trident.wsi_objects.WSI import ReadMode, WSI


class ArrayWSI(WSI):
    """
    WSI adapter backed by an in-memory RGB array.

    This is intended for challenge inputs that are known to be single-level
    20x TIFF images. It preserves TRIDENT's read_region contract without
    repeatedly opening the TIFF backend.
    """

    def __init__(self, slide_path: str, wsi_array: np.ndarray, **kwargs) -> None:
        if kwargs.get("mpp") is None:
            raise ValueError("ArrayWSI requires `mpp` because array inputs do not carry reliable slide metadata.")
        self.array = self._normalize_array(wsi_array)
        super().__init__(slide_path, **kwargs)

    @staticmethod
    def _normalize_array(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        elif array.ndim == 3 and array.shape[0] in (3, 4):
            array = np.moveaxis(array[:3], 0, -1)
        elif array.ndim == 3 and array.shape[-1] in (3, 4):
            array = array[:, :, :3]
        else:
            raise ValueError(f"Unsupported WSI array shape: {array.shape}")

        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array

    def _lazy_initialize(self) -> None:
        super()._lazy_initialize()
        if self._initialized:
            return

        height, width = self.array.shape[:2]
        self.dimensions = (width, height)
        self.width = width
        self.height = height
        self.level_count = 1
        self.level_downsamples = [1.0]
        self.level_dimensions = [self.dimensions]
        self.properties = {}
        self.mag = self._fetch_magnification(self.custom_mpp_keys)
        self._initialized = True

    def get_dimensions(self) -> Tuple[int, int]:
        self._lazy_initialize()
        return self.dimensions

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        self._lazy_initialize()
        img = Image.fromarray(self.array)
        img.thumbnail(size)
        return img.convert("RGB")

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        read_as: ReadMode = "pil",
    ) -> Union[Image.Image, np.ndarray]:
        self._lazy_initialize()
        if level != 0:
            raise ValueError("ArrayWSI only supports level=0.")

        x, y = int(location[0]), int(location[1])
        width, height = int(size[0]), int(size[1])
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + width), min(self.height, y + height)

        tile = np.full((height, width, 3), 255, dtype=np.uint8)
        if x1 > x0 and y1 > y0:
            dst_x0, dst_y0 = x0 - x, y0 - y
            tile[dst_y0:dst_y0 + (y1 - y0), dst_x0:dst_x0 + (x1 - x0)] = self.array[y0:y1, x0:x1]

        if read_as == "numpy":
            return tile
        if read_as == "pil":
            return Image.fromarray(tile)
        raise ValueError(f"Invalid `read_as` value: {read_as}. Must be 'pil' or 'numpy'.")

    def release(self) -> None:
        self.array = None
        super().release()
