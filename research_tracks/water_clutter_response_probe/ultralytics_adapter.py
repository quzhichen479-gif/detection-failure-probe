from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from probe_core import InferenceResult, scale_letterbox_boxes, xywh_to_xyxy


@dataclass
class RawCapture:
    boxes_xywh: np.ndarray | None = None
    class_scores: np.ndarray | None = None
    input_hw: tuple[int, int] | None = None


class UltralyticsYOLOAdapter:
    """Inference-only adapter for Ultralytics YOLO11.

    It exposes ordinary post-NMS predictions and, when possible, decoded pre-NMS
    candidate scores captured from the final Detect module. Failure to capture raw
    candidates never changes the primary post-NMS probe.
    """

    def __init__(
        self,
        weights: str | Path,
        imgsz: int,
        conf: float,
        iou: float,
        device: str | int | None,
        raw_candidate_mode: str = "auto",
    ) -> None:
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Run this probe inside the existing YOLO11/Ultralytics environment. "
                "It intentionally does not vendor or reinstall Ultralytics."
            ) from exc

        self.torch = torch
        self.weights = str(weights)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self.raw_candidate_mode = raw_candidate_mode
        self.model = YOLO(self.weights)
        self.model.model.eval()
        self._raw_capture = RawCapture()
        self._hook = None
        self._detect = None
        if raw_candidate_mode != "off":
            self._install_detect_hook(strict=raw_candidate_mode == "required")

    def _find_detect_module(self) -> Any | None:
        found = None
        for module in self.model.model.modules():
            name = module.__class__.__name__.lower()
            if name == "detect" or name.endswith("detect"):
                found = module
        return found

    @staticmethod
    def _decoded_from_output(output: Any):
        if isinstance(output, tuple) and len(output) >= 1:
            return output[0]
        if isinstance(output, list) and output and hasattr(output[0], "shape"):
            return None
        if hasattr(output, "shape"):
            return output
        return None

    def _install_detect_hook(self, strict: bool) -> None:
        detect = self._find_detect_module()
        if detect is None:
            if strict:
                raise RuntimeError("Could not locate final Ultralytics Detect module.")
            return
        self._detect = detect

        def hook(module, inputs, output):
            try:
                decoded = self._decoded_from_output(output)
                if decoded is None or not hasattr(decoded, "detach") or decoded.ndim != 3:
                    self._raw_capture = RawCapture()
                    return

                arr = decoded.detach().float().cpu()
                if arr.shape[0] != 1:
                    self._raw_capture = RawCapture()
                    return
                if arr.shape[1] > arr.shape[2] and arr.shape[2] <= 512:
                    arr = arr.transpose(1, 2)
                if arr.shape[1] < 5:
                    self._raw_capture = RawCapture()
                    return

                nc = int(getattr(module, "nc", arr.shape[1] - 4))
                if arr.shape[1] == 4 + nc:
                    sample = arr[0].numpy()
                    boxes = sample[:4].T
                    cls = sample[4 : 4 + nc].T
                elif arr.shape[2] == 4 + nc:
                    sample = arr[0].numpy()
                    boxes = sample[:, :4]
                    cls = sample[:, 4 : 4 + nc]
                else:
                    self._raw_capture = RawCapture()
                    return

                input_hw = None
                if inputs:
                    feature_arg = inputs[0]
                    if isinstance(feature_arg, (list, tuple)) and feature_arg:
                        f0 = feature_arg[0]
                        if hasattr(f0, "shape"):
                            stride = getattr(module, "stride", None)
                            if stride is not None and len(stride):
                                s0 = float(
                                    stride[0].detach().cpu()
                                    if hasattr(stride[0], "detach")
                                    else stride[0]
                                )
                                input_hw = (
                                    int(round(float(f0.shape[-2]) * s0)),
                                    int(round(float(f0.shape[-1]) * s0)),
                                )
                self._raw_capture = RawCapture(
                    boxes_xywh=np.asarray(boxes, dtype=np.float32),
                    class_scores=np.asarray(cls, dtype=np.float32),
                    input_hw=input_hw,
                )
            except Exception:
                if strict:
                    raise
                self._raw_capture = RawCapture()

        self._hook = detect.register_forward_hook(hook)

    def close(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def predict(self, image_rgb: np.ndarray) -> InferenceResult:
        image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
        self._raw_capture = RawCapture()
        with self.torch.inference_mode():
            results = self.model.predict(
                source=image,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
                save=False,
                stream=False,
            )
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            classes = np.zeros((0,), dtype=np.int64)
        else:
            boxes = result.boxes.xyxy.detach().float().cpu().numpy().astype(np.float32)
            scores = result.boxes.conf.detach().float().cpu().numpy().astype(np.float32)
            classes = result.boxes.cls.detach().long().cpu().numpy().astype(np.int64)

        raw_boxes = None
        raw_scores = None
        cap = self._raw_capture
        if cap.boxes_xywh is not None and cap.class_scores is not None and cap.input_hw is not None:
            orig_h, orig_w = image_rgb.shape[:2]
            raw_boxes = xywh_to_xyxy(cap.boxes_xywh)
            raw_boxes = scale_letterbox_boxes(raw_boxes, cap.input_hw, (orig_h, orig_w))
            raw_scores = cap.class_scores

        return InferenceResult(
            boxes=boxes,
            scores=scores,
            classes=classes,
            raw_boxes=raw_boxes,
            raw_scores=raw_scores,
        )
