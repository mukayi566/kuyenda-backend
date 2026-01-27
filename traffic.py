# traffic.py (UPDATED)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import sys
import datetime as dt

import pandas as pd


# ----------------------------
# Utilities
# ----------------------------
def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    # coords are (lng, lat)
    lng1, lat1 = a
    lng2, lat2 = b
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def now_lusaka_hour_and_dow() -> Tuple[int, int]:
    # Lusaka is UTC+2 (Africa/Lusaka)
    now = dt.datetime.utcnow() + dt.timedelta(hours=2)
    return now.hour, now.weekday()  # Mon=0..Sun=6


def is_rush_hour(hour: int, dow: int) -> bool:
    if dow <= 4:
        return (7 <= hour <= 9) or (16 <= hour <= 19)
    return False


def congestion_from_ratio(r: float) -> str:
    if r < 1.20:
        return "low"
    if r < 1.60:
        return "moderate"
    if r < 2.20:
        return "heavy"
    return "severe"


def class_to_level(c: int) -> str:
    return {0: "low", 1: "moderate", 2: "heavy", 3: "severe"}.get(int(c), "low")


def normalize_level(level: Any) -> str:
    """
    Mapbox congestion values can be: "low" "moderate" "heavy" "severe"
    Sometimes also: "unknown" or None.
    """
    if not level:
        return "low"
    s = str(level).lower().strip()
    if s in ("low", "moderate", "heavy", "severe"):
        return s
    if s == "unknown":
        return "low"
    return "low"


def level_to_color(level: str) -> str:
    # Optional helper if you want backend to supply colors
    return {
        "low": "#1E90FF",       # blue
        "moderate": "#FFD400",  # yellow
        "heavy": "#FF7A00",     # orange
        "severe": "#FF0000",    # red
    }.get(level, "#1E90FF")


# ----------------------------
# Model loader (robust)
# ----------------------------
def _install_numpy_compat_shim() -> None:
    try:
        import numpy.core as npcore  # type: ignore
        sys.modules.setdefault("numpy._core", npcore)
        sys.modules.setdefault("numpy._core._multiarray_umath", npcore._multiarray_umath)  # type: ignore
        sys.modules.setdefault("numpy._core.multiarray", npcore.multiarray)  # type: ignore
        sys.modules.setdefault("numpy._core.numeric", npcore.numeric)  # type: ignore
        sys.modules.setdefault("numpy._core.fromnumeric", npcore.fromnumeric)  # type: ignore
    except Exception:
        pass


def _install_custom_class_shim() -> None:
    if hasattr(sys.modules.get("__main__"), "BehavioralFeatureExtractor"):
        return

    class BehavioralFeatureExtractor:  # noqa: N801
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

        def fit_transform(self, X, y=None):
            return self.transform(X)

    sys.modules["__main__"].BehavioralFeatureExtractor = BehavioralFeatureExtractor  # type: ignore


@dataclass
class TrafficModelResult:
    using_ml: bool
    segments: List[Dict[str, Any]]
    breakdown: Dict[str, int]


class TrafficPredictor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        _install_numpy_compat_shim()
        _install_custom_class_shim()
        import joblib
        self.model = joblib.load(self.model_path)

    def _build_segments_from_annotation(
        self,
        route_coordinates: List[List[float]],
        annotation: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Mapbox route.legs[0].annotation typically includes:
          - durations: seconds, length = len(coords)-1
          - distances: meters,  length = len(coords)-1
          - congestion: string array, length = len(coords)-1
        """
        durs = annotation.get("durations") or []
        dists = annotation.get("distances") or []
        cong = annotation.get("congestion") or []

        n = len(route_coordinates) - 1
        segments: List[Dict[str, Any]] = []
        for i in range(n):
            seg = {}
            if i < len(durs) and durs[i] is not None:
                seg["duration_s"] = float(durs[i])
            if i < len(dists) and dists[i] is not None:
                seg["distance_m"] = float(dists[i])
            if i < len(cong) and cong[i] is not None:
                seg["congestion"] = normalize_level(cong[i])
            segments.append(seg)
        return segments

    def predict_route(
        self,
        route_coordinates: List[List[float]],
        *,
        # Preferred: already-built per-segment info (len = len(coords)-1)
        route_segments: Optional[List[Dict[str, Any]]] = None,
        # Optional alternative: raw mapbox annotation object
        annotation: Optional[Dict[str, Any]] = None,
        route_id: str = "route_0",
        origin: str = "",
        destination: str = "",
        hour: Optional[int] = None,
        day_of_week: Optional[int] = None,
        reports_count_default: int = 0,
        city_center_lnglat: Tuple[float, float] = (28.3228, -15.3875),
        hotspot_radius_km: float = 3.0,
        include_colors: bool = True,
    ) -> TrafficModelResult:
        """
        route_coordinates: list of [lng, lat]
        route_segments: optional list, length = len(route_coordinates)-1:
          - duration_s (or duration)
          - distance_m (or distance)
          - congestion ("low/moderate/heavy/severe")
          - reports_count
        annotation: optional Mapbox annotation object to auto-build segments.
        """
        if len(route_coordinates) < 2:
            return TrafficModelResult(using_ml=False, segments=[], breakdown={})

        if hour is None or day_of_week is None:
            h, dow = now_lusaka_hour_and_dow()
            hour = h if hour is None else hour
            day_of_week = dow if day_of_week is None else day_of_week

        if not origin:
            origin = f"{route_coordinates[0][0]},{route_coordinates[0][1]}"
        if not destination:
            destination = f"{route_coordinates[-1][0]},{route_coordinates[-1][1]}"

        # If route_segments not provided, try build from annotation
        if route_segments is None and annotation is not None:
            route_segments = self._build_segments_from_annotation(route_coordinates, annotation)

        seg_rows: List[Dict[str, Any]] = []
        seg_meta: List[Dict[str, Any]] = []

        n = len(route_coordinates) - 1
        for i in range(n):
            a = (float(route_coordinates[i][0]), float(route_coordinates[i][1]))
            b = (float(route_coordinates[i + 1][0]), float(route_coordinates[i + 1][1]))

            length_km = max(haversine_km(a, b), 0.001)

            # Normal speed heuristic (km/h)
            normal_speed = 40.0
            if is_rush_hour(int(hour), int(day_of_week)):
                normal_speed = 35.0
            normal_travel_min = (length_km / normal_speed) * 60.0

            seg_info = route_segments[i] if (route_segments and i < len(route_segments)) else {}

            duration_s = seg_info.get("duration_s", seg_info.get("duration", None))
            if duration_s is not None:
                current_travel_min = max(float(duration_s) / 60.0, 0.05)
            else:
                slowdown = 1.0
                if is_rush_hour(int(hour), int(day_of_week)):
                    slowdown = 1.5
                current_travel_min = normal_travel_min * slowdown

            ratio = current_travel_min / max(normal_travel_min, 0.05)

            # prefer provided congestion; else infer
            congestion_level = normalize_level(seg_info.get("congestion")) if seg_info.get("congestion") else congestion_from_ratio(ratio)

            # hotspot heuristic: segment midpoint near city center
            mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
            dist_to_center = haversine_km(mid, city_center_lnglat)
            hotspot = "yes" if dist_to_center <= hotspot_radius_km else "no"

            row = {
                "current_travel_min": float(current_travel_min),
                "normal_travel_min": float(normal_travel_min),
                "reports_count": int(seg_info.get("reports_count", reports_count_default)),
                "route_id": str(route_id),
                "origin": str(origin),
                "destination": str(destination),
                "hotspot": str(hotspot),
                "day_of_week": str(int(day_of_week)),
                "congestion_level": str(congestion_level),
            }
            seg_rows.append(row)

            seg_meta.append({
                "index": i,
                "from": {"lng": a[0], "lat": a[1]},
                "to": {"lng": b[0], "lat": b[1]},
                "segment_km": float(length_km),
                "mapbox_congestion": congestion_level,  # what we used as input label
                "duration_s": float(duration_s) if duration_s is not None else None,
                "distance_m": float(seg_info.get("distance_m", seg_info.get("distance", None))) if seg_info.get("distance_m", seg_info.get("distance", None)) is not None else None,
            })

        # Try ML
        try:
            self.load()
            df = pd.DataFrame(seg_rows)

            y = self.model.predict(df)  # type: ignore
            proba = None
            if hasattr(self.model, "predict_proba"):  # type: ignore
                proba = self.model.predict_proba(df)  # type: ignore

            breakdown = {"low": 0, "moderate": 0, "heavy": 0, "severe": 0}
            segments_out: List[Dict[str, Any]] = []

            for i, c in enumerate(y):
                level = class_to_level(int(c))
                conf = float(max(proba[i])) if proba is not None else None
                breakdown[level] += 1

                out = {
                    **seg_meta[i],
                    "level": level,              # ✅ what your UI should color by
                    "confidence": conf,
                    "source": "ml",
                }
                if include_colors:
                    out["color"] = level_to_color(level)
                segments_out.append(out)

            return TrafficModelResult(using_ml=True, segments=segments_out, breakdown=breakdown)

        except Exception as e:
            # Fallback heuristic
            breakdown = {"low": 0, "moderate": 0, "heavy": 0, "severe": 0}
            segments_out: List[Dict[str, Any]] = []

            for i, row in enumerate(seg_rows):
                ratio = row["current_travel_min"] / max(row["normal_travel_min"], 0.05)
                level = congestion_from_ratio(ratio)
                breakdown[level] += 1

                out = {
                    **seg_meta[i],
                    "level": level,
                    "confidence": None,
                    "source": f"heuristic ({type(e).__name__})",
                }
                if include_colors:
                    out["color"] = level_to_color(level)
                segments_out.append(out)

            return TrafficModelResult(using_ml=False, segments=segments_out, breakdown=breakdown)
