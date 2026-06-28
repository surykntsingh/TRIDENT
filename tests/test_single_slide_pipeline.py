import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from trident.single_slide_pipeline import extract_conch_v15_features_for_wsi


class _SlideContext:
    def __init__(self, slide):
        self.slide = slide

    def __enter__(self):
        return self.slide

    def __exit__(self, exc_type, exc, tb):
        return False


class _SegModel:
    def __init__(self, target_mag=10):
        self.target_mag = target_mag


class TestSingleSlidePipeline(unittest.TestCase):
    def test_reuses_existing_feature_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            feature_dir = root / "20x_512px_0px_overlap" / "features_conch_v15"
            feature_dir.mkdir(parents=True)
            feature_path = feature_dir / "case123.h5"
            feature_path.write_bytes(b"ok")

            returned = extract_conch_v15_features_for_wsi(
                wsi_path="/tmp/case123.tiff",
                job_dir=root,
            )

            self.assertEqual(returned, feature_path)

    def test_generates_coords_and_features(self):
        slide = MagicMock()
        slide.extract_patch_features.return_value = "/tmp/job/20x_512px_0px_overlap/features_conch_v15/case123.h5"
        seg_model = _SegModel(target_mag=10)
        encoder = MagicMock()
        encoder.enc_name = "conch_v15"

        with patch("trident.single_slide_pipeline.load_wsi", return_value=_SlideContext(slide)) as mock_load_wsi, \
             patch("trident.single_slide_pipeline.segmentation_model_factory", return_value=seg_model), \
             patch("trident.single_slide_pipeline.encoder_factory", return_value=encoder) as mock_encoder_factory:
            returned = extract_conch_v15_features_for_wsi(
                wsi_path="/tmp/case123.tiff",
                job_dir="/tmp/job",
                patch_encoder_weights_path="/tmp/conch/pytorch_model_vision.bin",
                device="cuda:0",
                mpp=0.5,
                reader_type="tiffslide",
                segmenter="hest",
                seg_conf_thresh=0.5,
                mag=20,
                patch_size=512,
                batch_size=256,
                remove_artifacts=True,
                remove_holes=True,
            )

        self.assertEqual(str(returned), "/tmp/job/20x_512px_0px_overlap/features_conch_v15/case123.h5")
        slide.segment_tissue.assert_called_once()
        slide.extract_tissue_coords.assert_called_once()
        slide.extract_patch_features.assert_called_once()
        _, load_wsi_kwargs = mock_load_wsi.call_args
        self.assertEqual(load_wsi_kwargs["reader_type"], "tiffslide")
        self.assertEqual(load_wsi_kwargs["mpp"], 0.5)
        _, encoder_kwargs = mock_encoder_factory.call_args
        self.assertEqual(encoder_kwargs["weights_path"], "/tmp/conch/pytorch_model_vision.bin")
        extract_kwargs = slide.extract_patch_features.call_args.kwargs
        self.assertEqual(extract_kwargs["coords_path"], "/tmp/job/20x_512px_0px_overlap/patches/case123_patches.h5")
        self.assertEqual(extract_kwargs["save_features"], "/tmp/job/20x_512px_0px_overlap/features_conch_v15")


if __name__ == "__main__":
    unittest.main()
