import sys
import types
import unittest
from unittest.mock import patch

from PIL import Image

from trident.wsi_objects.TiffSlideWSI import TiffSlideWSI


class _FakeTiffSlide:
    def __init__(self, slide_path):
        self.slide_path = slide_path
        self.dimensions = (1024, 512)
        self.level_count = 2
        self.level_downsamples = (1.0, 4.0)
        self.level_dimensions = ((1024, 512), (256, 128))
        self.properties = {
            "tiffslide.mpp-x": "0.5",
            "tiffslide.objective-power": "20",
        }

    def read_region(self, location, level, size):
        return Image.new("RGB", size, (255, 255, 255))

    def get_thumbnail(self, size):
        return Image.new("RGB", size, (0, 0, 0))

    def close(self):
        return None


class TestTiffSlideWSI(unittest.TestCase):
    def test_lazy_initialize_reads_tiffslide_metadata(self):
        fake_module = types.SimpleNamespace(TiffSlide=_FakeTiffSlide)
        with patch.dict(sys.modules, {"tiffslide": fake_module}):
            wsi = TiffSlideWSI(slide_path="/tmp/sample.tiff", lazy_init=False)
            self.assertEqual(wsi.mpp, 0.5)
            self.assertEqual(wsi.mag, 20)
            self.assertEqual(wsi.dimensions, (1024, 512))

    def test_fetch_mpp_scans_generic_mpp_keys(self):
        wsi = TiffSlideWSI(slide_path="/tmp/sample.tiff", lazy_init=True)
        wsi.img = types.SimpleNamespace(
            properties={
                "vendor.physical.microns_per_pixel_x": "0.2428",
                "vendor.objective_power": "40",
            }
        )
        self.assertEqual(wsi._fetch_mpp(), 0.2428)

    def test_fetch_mpp_extracts_numeric_value_from_comment_style_metadata(self):
        wsi = TiffSlideWSI(slide_path="/tmp/sample.tiff", lazy_init=True)
        wsi.img = types.SimpleNamespace(
            properties={
                "vendor.comment.mpp": "MPP = 0.5031 um/px",
            }
        )
        self.assertEqual(wsi._fetch_mpp(), 0.5031)

    def test_fetch_mpp_falls_back_to_tiff_resolution_tags(self):
        wsi = TiffSlideWSI(slide_path="/tmp/sample.tiff", lazy_init=True)
        wsi.img = types.SimpleNamespace(
            properties={
                "tiff.XResolution": "50800",
                "tiff.ResolutionUnit": "INCH",
            }
        )
        self.assertEqual(wsi._fetch_mpp(), 0.5)

    def test_missing_tiffslide_dependency_raises_clear_error(self):
        import builtins

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "tiffslide":
                raise ImportError("missing tiffslide")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            wsi = TiffSlideWSI(slide_path="/tmp/sample.tiff", lazy_init=True)
            with self.assertRaises(ImportError):
                wsi._lazy_initialize()


if __name__ == "__main__":
    unittest.main()
