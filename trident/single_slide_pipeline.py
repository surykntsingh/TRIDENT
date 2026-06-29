from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Sequence

import h5py

from trident.patch_encoder_models.load import encoder_factory
from trident.segmentation_models.load import segmentation_model_factory
from trident.wsi_objects.WSIFactory import WSIReaderType, load_wsi


def _resolve_path(root: Path, path: str | Path | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return root / resolved


def _default_segmentation_device(segmenter: str, device: str) -> str:
    return "cpu" if segmenter == "otsu" else device


def _coords_count(coords_path: Path) -> int:
    if not coords_path.exists():
        return 0
    with h5py.File(coords_path, "r") as h5_file:
        if "coords" not in h5_file:
            return 0
        return int(h5_file["coords"].shape[0])


def _feature_count(feature_path: Path) -> int:
    if not feature_path.exists():
        return 0
    with h5py.File(feature_path, "r") as h5_file:
        if "features" not in h5_file:
            return 0
        return int(h5_file["features"].shape[0])


def _clear_slide_contours(slide) -> None:
    if hasattr(slide, "gdf_contours"):
        delattr(slide, "gdf_contours")
    if hasattr(slide, "tissue_seg_path"):
        slide.tissue_seg_path = None


def _is_black_thumbnail_failure(slide, exc: Exception) -> bool:
    if getattr(slide, "tiffslide_black_thumbnail_detected", False):
        return True
    message = str(exc).lower()
    return "black thumbnail" in message or "entirely black thumbnail" in message


def _extract_coords_with_segmenter(
    *,
    slide,
    coords_path: Path,
    coords_root: Path,
    wsi_name: str,
    segmenter: str,
    seg_conf_thresh: float,
    mag: int,
    patch_size: int,
    overlap: int,
    min_tissue_proportion: float,
    remove_holes: bool,
    dataloader_workers: int,
    device: str,
) -> int:
    segmentation_model = segmentation_model_factory(
        model_name=segmenter,
        confidence_thresh=seg_conf_thresh,
    )
    slide.segment_tissue(
        segmentation_model=segmentation_model,
        target_mag=segmentation_model.target_mag,
        job_dir=str(coords_root.parent),
        device=_default_segmentation_device(segmenter, device),
        holes_are_tissue=not remove_holes,
        num_workers=dataloader_workers,
    )
    slide.extract_tissue_coords(
        target_mag=mag,
        patch_size=patch_size,
        save_coords=str(coords_root),
        overlap=overlap,
        min_tissue_proportion=min_tissue_proportion,
    )
    count = _coords_count(coords_path)
    if count == 0:
        warnings.warn(
            f"No tissue coordinates found for '{wsi_name}' using segmenter='{segmenter}'."
        )
    return count


def extract_wsi_patch_features(
    *,
    wsi_path: str | Path,
    wsi_array=None,
    job_dir: str | Path,
    patch_encoder: str = "conch_v15",
    patch_encoder_weights_path: str | Path | None = None,
    segmenter: str = "otsu",
    seg_conf_thresh: float = 0.5,
    mag: int = 20,
    patch_size: int = 512,
    overlap: int = 0,
    batch_size: int = 32,
    dataloader_workers: int = 0,
    device: str = "cuda:0",
    reader_type: Optional[WSIReaderType] = None,
    reader_type_fallbacks: Sequence[WSIReaderType] | None = None,
    mpp: float | None = None,
    custom_mpp_keys: Optional[list[str]] = None,
    min_tissue_proportion: float = 0.0,
    remove_holes: bool = False,
    remove_artifacts: bool = False,
    remove_penmarks: bool = False,
    fallback_segmenters: bool = True,
    saveas: str = "h5",
    features_dir: str | Path | None = None,
    reuse_existing: bool = True,
) -> Path:
    """
    Compute patch features for one WSI and return the saved feature path.

    This wraps the same stages used by TRIDENT's single-slide script:
    segmentation -> patch coordinates -> patch features.
    """
    wsi_path = Path(wsi_path)
    job_dir = Path(job_dir)
    mag_str = f"{float(mag):g}"
    coords_root = job_dir / f"{mag_str}x_{patch_size}px_{overlap}px_overlap"
    coords_path = coords_root / "patches" / f"{wsi_path.stem}_patches.h5"
    resolved_features_dir = _resolve_path(job_dir, features_dir) or (coords_root / f"features_{patch_encoder}")
    feature_path = resolved_features_dir / f"{wsi_path.stem}.{saveas}"

    if reuse_existing and feature_path.exists() and _feature_count(feature_path) > 0:
        return feature_path

    patch_encoder_weights = str(patch_encoder_weights_path) if patch_encoder_weights_path is not None else None

    reader_candidates = list(reader_type_fallbacks or [])
    if reader_type is not None:
        reader_candidates.insert(0, reader_type)
    if not reader_candidates:
        reader_candidates = [None]

    last_reader_error: Exception | None = None
    for reader_candidate in reader_candidates:
        load_kwargs = {
            "slide_path": str(wsi_path),
            "reader_type": reader_candidate,
            "lazy_init": False,
            "mpp": mpp,
            "custom_mpp_keys": custom_mpp_keys,
        }
        if wsi_array is not None:
            load_kwargs["wsi_array"] = wsi_array
        if reader_candidate == "tiffslide":
            load_kwargs["allow_openslide_fallback"] = False
        try:
            slide_cm = load_wsi(**load_kwargs)
        except Exception as exc:
            last_reader_error = exc
            warnings.warn(
                f"Failed to initialize WSI reader '{reader_candidate}' for '{wsi_path.name}': {exc}"
            )
            continue

        try:
            with slide_cm as slide:
                return _extract_wsi_patch_features_from_slide(
                    slide=slide,
                    wsi_path=wsi_path,
                    job_dir=job_dir,
                    coords_root=coords_root,
                    coords_path=coords_path,
                    resolved_features_dir=resolved_features_dir,
                    patch_encoder=patch_encoder,
                    patch_encoder_weights=patch_encoder_weights,
                    segmenter=segmenter,
                    seg_conf_thresh=seg_conf_thresh,
                    mag=mag,
                    patch_size=patch_size,
                    overlap=overlap,
                    batch_size=batch_size,
                    dataloader_workers=dataloader_workers,
                    device=device,
                    min_tissue_proportion=min_tissue_proportion,
                    remove_holes=remove_holes,
                    remove_artifacts=remove_artifacts,
                    remove_penmarks=remove_penmarks,
                    fallback_segmenters=fallback_segmenters,
                    saveas=saveas,
                )
        except Exception as exc:
            last_reader_error = exc
            warnings.warn(
                f"WSI reader '{reader_candidate}' failed while processing '{wsi_path.name}': {exc}"
            )
            continue

    raise RuntimeError(
        f"All WSI readers failed for '{wsi_path.name}'. Tried {reader_candidates}."
    ) from last_reader_error


def _extract_wsi_patch_features_from_slide(
    *,
    slide,
    wsi_path: Path,
    job_dir: Path,
    coords_root: Path,
    coords_path: Path,
    resolved_features_dir: Path,
    patch_encoder: str,
    patch_encoder_weights: str | None,
    segmenter: str,
    seg_conf_thresh: float,
    mag: int,
    patch_size: int,
    overlap: int,
    batch_size: int,
    dataloader_workers: int,
    device: str,
    min_tissue_proportion: float,
    remove_holes: bool,
    remove_artifacts: bool,
    remove_penmarks: bool,
    fallback_segmenters: bool,
    saveas: str,
) -> Path:
    # Submission containers tend to have very limited /dev/shm. Force
    # sequential DataLoader execution by default for reliability.
    slide.max_workers = dataloader_workers
    skip_segmentation = False
    if coords_path.exists() and _coords_count(coords_path) == 0:
        warnings.warn(
            f"Existing coordinate file for '{wsi_path.name}' contains no patches; regenerating it."
        )
        coords_path.unlink()
    if not coords_path.exists():
        try:
            count = _extract_coords_with_segmenter(
                slide=slide,
                coords_path=coords_path,
                coords_root=coords_root,
                wsi_name=wsi_path.name,
                segmenter=segmenter,
                seg_conf_thresh=seg_conf_thresh,
                mag=mag,
                patch_size=patch_size,
                overlap=overlap,
                min_tissue_proportion=min_tissue_proportion,
                remove_holes=remove_holes,
                dataloader_workers=dataloader_workers,
                device=device,
            )
        except Exception as exc:
            if not _is_black_thumbnail_failure(slide, exc):
                raise
            warnings.warn(
                f"Skipping tissue segmentation for '{wsi_path.name}' because the selected "
                "reader produced an entirely black thumbnail. Using unmasked whole-slide "
                "coordinates instead."
            )
            count = 0
            skip_segmentation = True
        if not skip_segmentation and fallback_segmenters and count == 0 and segmenter != "otsu":
            if segmenter == "hest" and seg_conf_thresh > 0.1:
                warnings.warn(
                    f"Retrying segmenter='hest' with seg_conf_thresh=0.1 for '{wsi_path.name}'."
                )
                _clear_slide_contours(slide)
                count = _extract_coords_with_segmenter(
                    slide=slide,
                    coords_path=coords_path,
                    coords_root=coords_root,
                    wsi_name=wsi_path.name,
                    segmenter="hest",
                    seg_conf_thresh=0.1,
                    mag=mag,
                    patch_size=patch_size,
                    overlap=overlap,
                    min_tissue_proportion=min_tissue_proportion,
                    remove_holes=remove_holes,
                    dataloader_workers=dataloader_workers,
                    device=device,
                )
        if not skip_segmentation and fallback_segmenters and count == 0 and segmenter != "otsu":
            warnings.warn(
                f"Falling back to segmenter='otsu' for '{wsi_path.name}' before using unmasked coordinates."
            )
            _clear_slide_contours(slide)
            count = _extract_coords_with_segmenter(
                slide=slide,
                coords_path=coords_path,
                coords_root=coords_root,
                wsi_name=wsi_path.name,
                segmenter="otsu",
                seg_conf_thresh=seg_conf_thresh,
                mag=mag,
                patch_size=patch_size,
                overlap=overlap,
                min_tissue_proportion=min_tissue_proportion,
                remove_holes=remove_holes,
                dataloader_workers=dataloader_workers,
                device=device,
            )
        if count == 0:
            warnings.warn(
                f"Falling back to unmasked whole-slide coordinates for '{wsi_path.name}'."
            )
            _clear_slide_contours(slide)
            slide.extract_tissue_coords(
                target_mag=mag,
                patch_size=patch_size,
                save_coords=str(coords_root),
                overlap=overlap,
                min_tissue_proportion=0.0,
            )
        if remove_artifacts or remove_penmarks:
            artifact_remover_model = segmentation_model_factory(
                "grandqc_artifact",
                remove_penmarks_only=remove_penmarks and not remove_artifacts,
            )
        else:
            artifact_remover_model = None
        if artifact_remover_model is not None:
            slide.segment_tissue(
                segmentation_model=artifact_remover_model,
                target_mag=artifact_remover_model.target_mag,
                holes_are_tissue=False,
                job_dir=str(job_dir),
                num_workers=dataloader_workers,
            )
            slide.extract_tissue_coords(
                target_mag=mag,
                patch_size=patch_size,
                save_coords=str(coords_root),
                overlap=overlap,
                min_tissue_proportion=0.0,
            )

    encoder = encoder_factory(
        patch_encoder,
        weights_path=patch_encoder_weights,
    )
    generated_path = slide.extract_patch_features(
        patch_encoder=encoder,
        coords_path=str(coords_path),
        save_features=str(resolved_features_dir),
        device=device,
        saveas=saveas,
        batch_limit=batch_size,
    )

    return Path(generated_path)


def extract_conch_v15_features_for_wsi(
    *,
    wsi_path: str | Path,
    wsi_array=None,
    job_dir: str | Path,
    patch_encoder_weights_path: str | Path | None = None,
    segmenter: str = "otsu",
    seg_conf_thresh: float = 0.5,
    mag: int = 20,
    patch_size: int = 512,
    overlap: int = 0,
    batch_size: int = 32,
    dataloader_workers: int = 0,
    device: str = "cuda:0",
    reader_type: Optional[WSIReaderType] = None,
    reader_type_fallbacks: Sequence[WSIReaderType] | None = None,
    mpp: float | None = None,
    custom_mpp_keys: Optional[list[str]] = None,
    min_tissue_proportion: float = 0.0,
    remove_holes: bool = False,
    remove_artifacts: bool = False,
    remove_penmarks: bool = False,
    fallback_segmenters: bool = True,
    reuse_existing: bool = True,
) -> Path:
    return extract_wsi_patch_features(
        wsi_path=wsi_path,
        wsi_array=wsi_array,
        job_dir=job_dir,
        patch_encoder="conch_v15",
        patch_encoder_weights_path=patch_encoder_weights_path,
        segmenter=segmenter,
        seg_conf_thresh=seg_conf_thresh,
        mag=mag,
        patch_size=patch_size,
        overlap=overlap,
        batch_size=batch_size,
        dataloader_workers=dataloader_workers,
        device=device,
        reader_type=reader_type,
        reader_type_fallbacks=reader_type_fallbacks,
        mpp=mpp,
        custom_mpp_keys=custom_mpp_keys,
        min_tissue_proportion=min_tissue_proportion,
        remove_holes=remove_holes,
        remove_artifacts=remove_artifacts,
        remove_penmarks=remove_penmarks,
        fallback_segmenters=fallback_segmenters,
        saveas="h5",
        reuse_existing=reuse_existing,
    )
