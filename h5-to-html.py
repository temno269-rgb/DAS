#!/usr/bin/env python3
"""
Интерактивный 3D/2D-график активности из .h5 артефактов.
Читает activity.h5 (или любой .h5 с данными windows x channels).
Открывает отдельную HTML-страницу в браузере.

Оси:
  X - Абсолютное время (из .h5: start_time + time_start_sample / sample_rate_hz)
  Y - Каналы / расстояние в метрах (из .h5: distance_m)
  Z - Активность / мощность (mean / max / min / m2) из .h5 матрицы

Оптимизирован для сверхбольших матриц (например, 11471 x 3057) за счёт:
  1. Единого мастер-датасета с динамическим прореживанием прямо на странице.
  2. Переноса тяжелых вычислений meshgrid на клиентскую сторону JS.
  3. Устранения ошибок сериализации (циклических ссылок) в Plotly.react/redraw.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import h5py
import plotly.graph_objects as go


# ═══════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════

def _parse_flexible_timestamp(raw_time: str) -> datetime:
    """Парсит timestamp вида '20260713T112525151040' с переменной длиной дробной части секунд.

    Поддерживаемые форматы дробной части:
      - 0 цифр:  '20260713T112525'        → 11:25:25.000000
      - 1-6 цифр: '20260713T112525151'    → 11:25:25.151000  (добиваем до 6)
      - 6+ цифр:  '20260713T112525151040' → 11:25:25.151040  (обрезаем до 6)
    """
    if "T" not in raw_time:
        raise ValueError(f"Expected ISO-compact format with 'T': {raw_time!r}")

    date_part, time_part = raw_time.split("T", 1)

    if len(time_part) <= 6:
        # Только HHMMSS без дробной части — формат без %f
        return datetime.strptime(f"{date_part}T{time_part}", "%Y%m%dT%H%M%S")
    else:
        hhmmss = time_part[:6]
        frac = time_part[6:]
        # Добиваем до 6 цифр (микросекунды) или обрезаем
        frac = frac.ljust(6, "0")[:6]
        return datetime.strptime(f"{date_part}T{hhmmss}{frac}", "%Y%m%dT%H%M%S%f")


def _fmt_time_axis(val: float) -> str:
    """Форматирует значение оси времени для тиков Python-рендера."""
    if val > 1e9:
        dt = datetime.fromtimestamp(val, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return f"{val:.3f}"


def _build_wall_surface(
    xw: np.ndarray,
    yw: np.ndarray,
    zw: np.ndarray,
    cdw: np.ndarray,
    scw: np.ndarray,
    cmin: float,
    cmax: float,
) -> go.Surface:
    """Создаёт поверхность-стенку / пол для 3D-сцены с единым цветовым диапазоном."""
    return go.Surface(
        x=xw, y=yw, z=zw, customdata=cdw, surfacecolor=scw,
        colorscale="Turbo", showscale=False, hoverinfo="skip",
        cauto=False, cmin=cmin, cmax=cmax,
        lighting=dict(ambient=0.5, diffuse=0.7, roughness=0.8),
    )


# ═══════════════════════════════════════════════════════════════════════
# Core I/O
# ═══════════════════════════════════════════════════════════════════════

def read_activity_h5(path: Path | str) -> dict:
    """Читает .h5 артефакт и возвращает словарь массивов + атрибуты.

    Временные массивы (time_start_sample, time_stop_sample) конвертируются
    из отсчётов в секунды делением на sample_rate_hz.  Атрибут start_time
    парсится в Unix-timestamp (float64) и сохраняется в arrays["start_time"].
    """
    with h5py.File(str(path), "r") as f:
        attrs = {k: f.attrs[k] for k in f.attrs.keys()}

        sr = float(attrs.get("sample_rate_hz", 1.0))
        if sr <= 0:
            sr = 1.0

        arrays: dict = {}
        for k in f.keys():
            if "count" in k or "offset" in k:
                arrays[k] = np.asarray(f[k], dtype=np.uint32)
            elif k in ("time_start_sample", "time_stop_sample"):
                # Конвертируем отсчёты → секунды прямо при чтении
                arrays[k] = np.asarray(f[k], dtype=np.float64) / sr
            else:
                arrays[k] = np.asarray(f[k], dtype=np.float32)

        if "distance_m" in f:
            arrays["distance_m"] = np.asarray(f["distance_m"], dtype=np.float64)

        # Парсинг start_time → Unix timestamp
        start_time_unix = 0.0
        if "start_time" in f.attrs:
            raw_time = f.attrs["start_time"]
            if isinstance(raw_time, bytes):
                raw_time = raw_time.decode("utf-8")
            if isinstance(raw_time, str):
                try:
                    dt = _parse_flexible_timestamp(raw_time)
                    start_time_unix = dt.timestamp()
                except (ValueError, OverflowError):
                    try:
                        start_time_unix = float(raw_time)
                    except (ValueError, TypeError):
                        start_time_unix = 0.0
            else:
                try:
                    start_time_unix = float(raw_time)
                except (ValueError, TypeError):
                    start_time_unix = 0.0

        arrays["start_time"] = np.asarray(start_time_unix, dtype=np.float64)
        attrs["start_time"] = start_time_unix  # Унифицируем: и в attrs, и в arrays

    return {"arrays": arrays, "attrs": attrs}



def _read_start_time_attr(attrs: dict) -> float:
    raw_time = attrs.get("start_time", 0.0)
    if isinstance(raw_time, bytes):
        raw_time = raw_time.decode("utf-8")
    if isinstance(raw_time, str):
        try:
            return _parse_flexible_timestamp(raw_time).timestamp()
        except (ValueError, OverflowError):
            try:
                return float(raw_time)
            except (ValueError, TypeError):
                return 0.0
    try:
        return float(raw_time)
    except (ValueError, TypeError):
        return 0.0


def _pool_time_percentile(data: np.ndarray, edges: np.ndarray, percentile: float) -> np.ndarray:
    out = np.empty((len(edges) - 1, data.shape[1]), dtype=np.float32)
    for i in range(len(out)):
        out[i] = np.percentile(data[edges[i]:edges[i + 1]], percentile, axis=0)
    return out


def read_activity_h5_collection(
    paths: list[Path | str],
    time_stride: int = 4,
    channel_stride: int = 5,
    target_time_rows: int = 6000,
    percentile: float = 95.0,
) -> dict:
    """Сортирует H5 по start_time, прореживает 4×5 и объединяет через P95."""
    if time_stride < 1 or channel_stride < 1 or target_time_rows < 1:
        raise ValueError("Шаги прореживания и target_time_rows должны быть >= 1")

    infos = []
    for raw_path in paths:
        path = Path(raw_path)
        with h5py.File(str(path), "r") as f:
            attrs = {k: f.attrs[k] for k in f.attrs.keys()}
            keys = {
                k for k in f.keys()
                if isinstance(f[k], h5py.Dataset)
                and f[k].ndim == 2
                and "offset" not in k
            }
            if not keys:
                raise ValueError(f"Не найдено 2D матриц в файле {path}")
            infos.append({
                "path": path,
                "attrs": attrs,
                "start": _read_start_time_attr(attrs),
                "keys": keys,
                "shapes": {k: tuple(f[k].shape) for k in keys},
            })

    infos.sort(key=lambda x: (
        x["start"] <= 0,
        x["start"] if x["start"] > 0 else float("inf"),
        x["path"].name,
    ))
    common_keys = set.intersection(*(x["keys"] for x in infos))
    if not common_keys:
        raise ValueError("В файлах нет общей 2D-метрики")

    preferred = [k for k in common_keys if "count" not in k]
    shape_key = "mean" if "mean" in common_keys else sorted(preferred or common_keys)[0]
    channel_count = infos[0]["shapes"][shape_key][1]
    for info in infos:
        if info["shapes"][shape_key][1] != channel_count:
            raise ValueError("Файлы содержат разное число каналов")

    print(f"[INFO] Объединение {len(infos)} H5-файлов по start_time:")
    for i, info in enumerate(infos, 1):
        print(f"  {i:02d}. {info['start']:.6f} | {info['path'].name}")

    sampled_times = []
    total_original_windows = 0
    expected_rows = []
    for info in infos:
        rows = info["shapes"][shape_key][0]
        expected_rows.append(len(range(0, rows, time_stride)))
        total_original_windows += rows
        attrs = info["attrs"]
        sr = float(attrs.get("sample_rate_hz", 1.0))
        if sr <= 0:
            sr = 1.0
        with h5py.File(str(info["path"]), "r") as f:
            if "time_start_sample" in f:
                t0 = np.asarray(f["time_start_sample"][::time_stride], dtype=np.float64) / sr
                if "time_stop_sample" in f:
                    t1 = np.asarray(f["time_stop_sample"][::time_stride], dtype=np.float64) / sr
                    tx = (t0 + t1) / 2.0
                else:
                    tx = t0
            else:
                win_sec = float(attrs.get("window_seconds", 1.0))
                tx = np.arange(0, rows, time_stride, dtype=np.float64) * win_sec
        if info["start"] != 0.0:
            tx += info["start"]
        sampled_times.append(tx)

    time_concat = np.concatenate(sampled_times)
    n_out = min(target_time_rows, len(time_concat))
    edges = np.linspace(0, len(time_concat), n_out + 1, dtype=np.int64)
    time_axis = np.empty(n_out, dtype=np.float64)
    for i in range(n_out):
        time_axis[i] = float(np.mean(time_concat[edges[i]:edges[i + 1]]))

    arrays: dict[str, np.ndarray] = {}
    for key in sorted(common_keys):
        parts = []
        compatible = True
        for info, sampled_rows in zip(infos, expected_rows):
            shape = info["shapes"].get(key)
            if shape is None or shape[0] != info["shapes"][shape_key][0] or shape[1] != channel_count:
                compatible = False
                break
            with h5py.File(str(info["path"]), "r") as f:
                part = np.asarray(f[key][::time_stride, ::channel_stride], dtype=np.float32)
            if part.shape[0] != sampled_rows:
                compatible = False
                break
            parts.append(part)
        if not compatible:
            continue
        combined = np.concatenate(parts, axis=0)
        np.nan_to_num(combined, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        arrays[key] = _pool_time_percentile(combined, edges, percentile)
        del combined, parts
        print(f"[INFO] P{percentile:g}: {key} -> {arrays[key].shape[0]} x {arrays[key].shape[1]}")

    first = infos[0]
    with h5py.File(str(first["path"]), "r") as f:
        arrays["distance_m"] = (
            np.asarray(f["distance_m"][::channel_stride], dtype=np.float64)
            if "distance_m" in f
            else np.arange(0, channel_count, channel_stride, dtype=np.float64)
        )

    arrays["time_start_sample"] = time_axis
    arrays["time_stop_sample"] = time_axis
    arrays["start_time"] = np.asarray(0.0, dtype=np.float64)

    attrs = dict(first["attrs"])
    attrs["start_time"] = 0.0
    attrs["sample_rate_hz"] = 1.0
    attrs["format"] = f"combined-{len(infos)}-files-p{percentile:g}-uint16"
    attrs["source_files_count"] = len(infos)
    attrs["original_windows_count"] = total_original_windows
    attrs["time_stride"] = time_stride
    attrs["channel_stride"] = channel_stride
    attrs["aggregation_percentile"] = percentile
    return {"arrays": arrays, "attrs": attrs}

def inject_to_html(html_str: str, css_content: str, js_content: str) -> str:
    """Надёжно внедряет кастомные стили и JavaScript-код в разметку Plotly HTML."""
    idx_head = html_str.lower().find("</head>")
    if idx_head != -1:
        html_str = html_str[:idx_head] + css_content + html_str[idx_head:]
    else:
        html_str = css_content + html_str

    idx_body = html_str.lower().find("</body>")
    if idx_body != -1:
        html_str = html_str[:idx_body] + js_content + html_str[idx_body:]
    else:
        html_str = html_str + js_content
    return html_str


# ═══════════════════════════════════════════════════════════════════════
# Main builder
# ═══════════════════════════════════════════════════════════════════════

def build_interactive_3d(
    h5_path: Path | str | list[Path | str],
    output_html: Path | str = "activity_3d.html",
    metric_key: str = "mean",
    time_stride: int = 4,
    channel_stride: int = 5,
    target_time_rows: int = 6000,
    percentile: float = 95.0,
) -> Path:
    if isinstance(h5_path, (list, tuple, set)):
        paths = [Path(p) for p in h5_path]
        print(f"[INFO] Загрузка коллекции из {len(paths)} файлов...")
        result = read_activity_h5_collection(
            paths, time_stride=time_stride, channel_stride=channel_stride,
            target_time_rows=target_time_rows, percentile=percentile,
        )
    else:
        print(f"[INFO] Загрузка файла {h5_path}...")
        result = read_activity_h5(h5_path)
    arrays = result["arrays"]
    attrs = result["attrs"]

    # Автоопределение 2D матриц метрик в .h5
    available_metrics_list = [
        k for k in arrays.keys()
        if isinstance(arrays[k], np.ndarray)
        and arrays[k].ndim == 2
        and "count" not in k
        and "offset" not in k
        and k not in ("time_start_sample", "time_stop_sample")
    ]
    if not available_metrics_list:
        available_metrics_list = [
            k for k in arrays.keys()
            if isinstance(arrays[k], np.ndarray) and arrays[k].ndim == 2
        ]

    if not available_metrics_list:
        raise ValueError(
            f"Не найдено 2D матриц в файле {h5_path}. "
            f"Доступные ключи: {list(arrays.keys())}"
        )

    if metric_key not in arrays and available_metrics_list:
        metric_key = available_metrics_list[0]

    n_windows, n_channels = arrays[metric_key].shape
    print(f"[INFO] Исходный размер матрицы активности: {n_windows} окон x {n_channels} каналов")

    # ═══ Производные метрики (вычисляются из существующих данных) ═══
    print("[INFO] Расчёт производных метрик...")
    _mean_arr = arrays.get("mean")
    _max_arr = arrays.get("max")
    _min_arr = arrays.get("min")
    _m2_arr = arrays.get("m2")
    _count_arr = arrays.get("valid_count")

    # RMS = sqrt(m2) — среднеквадратичное значение
    if _m2_arr is not None and _m2_arr.shape == (n_windows, n_channels):
        arrays["rms"] = np.sqrt(np.maximum(_m2_arr, 0)).astype(np.float32)
        print("[INFO]   + rms (sqrt(m2))")

    # Dynamic Range = max - min — диапазон значений в окне
    if _max_arr is not None and _min_arr is not None and \
       _max_arr.shape == (n_windows, n_channels) and _min_arr.shape == (n_windows, n_channels):
        arrays["dynamic_range"] = (_max_arr - _min_arr).astype(np.float32)
        print("[INFO]   + dynamic_range (max - min)")

    # Activity Index = log(1 + mean) — логарифмическая активность
    if _mean_arr is not None and _mean_arr.shape == (n_windows, n_channels):
        arrays["activity_index"] = np.log1p(np.maximum(_mean_arr, 0)).astype(np.float32)
        print("[INFO]   + activity_index (log(1+mean))")

    # SNR и CV — вычисляем variance/std_dev один раз
    if _mean_arr is not None and _m2_arr is not None and _count_arr is not None and \
       _mean_arr.shape == (n_windows, n_channels) and _m2_arr.shape == (n_windows, n_channels) and \
       _count_arr.shape == (n_windows, n_channels):
        with np.errstate(divide="ignore", invalid="ignore"):
            variance = np.where(_count_arr > 1, _m2_arr / _count_arr - _mean_arr ** 2, 0)
            variance = np.maximum(variance, 0)
            std_dev = np.sqrt(variance)
            # SNR (Signal-to-Noise Ratio) = |mean| / std_dev
            snr = np.where(std_dev > 1e-10, np.abs(_mean_arr) / std_dev, 0)
            arrays["snr"] = np.nan_to_num(snr, nan=0, posinf=0, neginf=0).astype(np.float32)
            print("[INFO]   + snr (mean / std_dev)")
            # Coefficient of Variation = std_dev / |mean|
            cv = np.where(np.abs(_mean_arr) > 1e-10, std_dev / np.abs(_mean_arr), 0)
            arrays["cv"] = np.nan_to_num(cv, nan=0, posinf=100, neginf=0).astype(np.float32)
            print("[INFO]   + cv (std_dev / |mean|)")

    # Percentile Range = p98 - p2 (по каналам) — мера разброса
    if _mean_arr is not None and _mean_arr.shape == (n_windows, n_channels):
        with np.errstate(invalid="ignore"):
            p98_per_win = np.nanpercentile(np.where(np.isfinite(_mean_arr), _mean_arr, np.nan), 98, axis=1)
            p02_per_win = np.nanpercentile(np.where(np.isfinite(_mean_arr), _mean_arr, np.nan), 2, axis=1)
            spread = (p98_per_win - p02_per_win)[:, None] * np.ones((1, n_channels))
        arrays["spread"] = np.nan_to_num(spread, nan=0).astype(np.float32)
        print("[INFO]   + spread (p98-p2 per window)")

    # Обновляем список метрик с учётом производных
    available_metrics_list = [
        k for k in arrays.keys()
        if isinstance(arrays[k], np.ndarray)
        and arrays[k].ndim == 2
        and "count" not in k
        and "offset" not in k
        and k not in ("time_start_sample", "time_stop_sample")
    ]
    if not available_metrics_list:
        available_metrics_list = [
            k for k in arrays.keys()
            if isinstance(arrays[k], np.ndarray) and arrays[k].ndim == 2
        ]

    if not available_metrics_list:
        raise ValueError(
            f"Не найдено 2D матриц в файле {h5_path}. "
            f"Доступные ключи: {list(arrays.keys())}"
        )

    if metric_key not in arrays and available_metrics_list:
        metric_key = available_metrics_list[0]

    # Статистические показатели для HUD и цветового диапазона
    print("[INFO] Расчёт статистических показателей для HUD...")
    metric_stats: dict[str, dict[str, float]] = {}
    for m in available_metrics_list:
        arr = np.asarray(arrays[m], dtype=np.float32)
        if arr.shape == (n_windows, n_channels):
            clean_arr = arr[np.isfinite(arr)]
            if clean_arr.size > 0:
                metric_stats[m] = {
                    "min": float(clean_arr.min()),
                    "max": float(clean_arr.max()),
                    "mean": float(clean_arr.mean()),
                    "p2": float(np.percentile(clean_arr, 2)),
                    "p5": float(np.percentile(clean_arr, 5)),
                    "p95": float(np.percentile(clean_arr, 95)),
                    "p98": float(np.percentile(clean_arr, 98)),
                    "std": float(np.std(clean_arr)),
                    "median": float(np.median(clean_arr)),
                }
            else:
                metric_stats[m] = {"min": 0.0, "max": 0.0, "mean": 0.0,
                                   "p2": 0.0, "p5": 0.0, "p95": 0.0, "p98": 0.0,
                                   "std": 0.0, "median": 0.0}

    stats = metric_stats.get(metric_key, {"min": 0.0, "max": 1.0, "mean": 0.0,
                                          "p2": 0.0, "p5": 0.0, "p95": 0.0, "p98": 0.0})
    # Начальный рендер использует перцентильный диапазон (как JS по умолчанию)
    stats_min, stats_max = float(stats.get("p2", stats["min"])), float(stats.get("p98", stats["max"]))

    # ===== Ось времени (X) — АБСОЛЮТНОЕ ВРЕМЯ =====
    sr = float(attrs.get("sample_rate_hz", 1.0))
    if sr <= 0:
        sr = 1.0

    if "time_start_sample" in arrays and "time_stop_sample" in arrays:
        # Уже в секундах (конвертировано при чтении)
        time_start = arrays["time_start_sample"]
        time_stop = arrays["time_stop_sample"]
        time_axis = (time_start + time_stop) / 2.0
    elif "time_start_sample" in arrays:
        time_axis = arrays["time_start_sample"]
    elif "window_seconds" in attrs:
        win_sec = float(attrs["window_seconds"])
        time_axis = np.arange(n_windows, dtype=np.float64) * win_sec
    else:
        time_axis = np.arange(n_windows, dtype=np.float64)

    if len(time_axis) != n_windows:
        time_axis = np.linspace(float(time_axis[0]), float(time_axis[-1]), n_windows)

    # Прибавляем абсолютное время начала записи
    start_time_offset = float(arrays.get("start_time", 0.0))
    if start_time_offset != 0.0:
        time_axis = time_axis + start_time_offset
        print(f"[INFO] Абсолютное время: {time_axis[0]:.3f} ... {time_axis[-1]:.3f} с "
              f"(start_time={start_time_offset})")

    # ===== Ось расстояний (Y) =====
    if "distance_m" in arrays:
        channels_axis = np.asarray(arrays["distance_m"], dtype=np.float64)
    else:
        channels_axis = np.arange(n_channels, dtype=np.float64)
    if len(channels_axis) != n_channels:
        channels_axis = np.linspace(0, max(1, n_channels - 1), n_channels)

    # ===== МАСТЕР-ДАТАСЕТ =====
    MASTER_MAX_TIME = 6000
    MASTER_MAX_CHANNELS = 1000
    step_win_m = max(1, int(np.ceil(n_windows / MASTER_MAX_TIME)))
    step_chan_m = max(1, int(np.ceil(n_channels / MASTER_MAX_CHANNELS)))
    master_n_win = len(range(0, n_windows, step_win_m))
    master_n_chan = len(range(0, n_channels, step_chan_m))
    print(f"[INFO] Мастер-датасет: {master_n_win} x {master_n_chan} "
          f"(шаг: {step_win_m} x {step_chan_m})")

    all_metrics_master: dict[str, str] = {}
    for m in available_metrics_list:
        arr = np.asarray(arrays[m], dtype=np.float32)
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.shape == (n_windows, n_channels):
            sub = np.ascontiguousarray(arr[::step_win_m, ::step_chan_m], dtype=np.float32)
            q_min = float(np.min(sub))
            q_max = float(np.max(sub))
            if q_max > q_min:
                quantized = np.rint((sub - q_min) * (65535.0 / (q_max - q_min))).clip(0, 65535).astype("<u2")
            else:
                quantized = np.zeros(sub.shape, dtype="<u2")
            b64 = base64.b64encode(quantized.tobytes()).decode("ascii")
            all_metrics_master[m] = f"u16|{sub.shape[0]}|{sub.shape[1]}|{q_min:.17g}|{q_max:.17g}|{b64}"

    master_time = time_axis[::step_win_m].tolist()
    master_channels = channels_axis[::step_chan_m].tolist()

    # ===== Начальный 3D-рендер =====
    INIT_3D_TARGET = 200
    step_w3 = max(1, master_n_win // INIT_3D_TARGET)
    step_c3 = max(1, master_n_chan // INIT_3D_TARGET)

    def _decode_master_array(encoded: str) -> np.ndarray:
        """Декодирует uint16-квантизованный мастер-массив."""
        parts = encoded.split("|", 5)
        if parts[0] == "u16":
            rows, cols = int(parts[1]), int(parts[2])
            lo, hi = float(parts[3]), float(parts[4])
            raw = base64.b64decode(parts[5])
            q = np.frombuffer(raw, dtype="<u2").reshape(rows, cols)
            if hi <= lo:
                return np.full((rows, cols), lo, dtype=np.float32)
            return (lo + q.astype(np.float32) * ((hi - lo) / 65535.0)).astype(np.float32)
        rows, cols = int(parts[0]), int(parts[1])
        raw = base64.b64decode(parts[2])
        return np.frombuffer(raw, dtype=np.float32).reshape(rows, cols)

    time_3d = master_time[::step_w3]
    chan_3d = master_channels[::step_c3]

    Z_init_full = _decode_master_array(all_metrics_master[metric_key])
    Z_3d = Z_init_full[::step_w3, ::step_c3]
    z_floor = float(Z_3d.min())

    # Центрирование 3D оси времени
    t0_3d = time_3d[0]
    shift_time_3d = abs(t0_3d) > 1000000.0
    time_3d_plot = [t - t0_3d for t in time_3d] if shift_time_3d else list(time_3d)

    # Тиковые метки для начального 3D
    n3 = len(time_3d)
    n_ticks_3d = min(8, n3)
    tick_idx_3d = [int(i * (n3 - 1) / max(1, n_ticks_3d - 1)) for i in range(n_ticks_3d)]
    time_tickvals_3d = [time_3d_plot[i] for i in tick_idx_3d]
    time_ticktext_3d = [_fmt_time_axis(time_3d[i]) for i in tick_idx_3d]

    # NumPy meshgrid для начального 3D-рендера
    X_plot_3d, Y_3d = np.meshgrid(time_3d_plot, chan_3d, indexing="ij")
    X_abs_3d, _ = np.meshgrid(time_3d, chan_3d, indexing="ij")

    # Форматируем время для hover (чтобы отображалось читаемо, а не как число)
    X_hover_3d = np.vectorize(lambda v: _fmt_time_axis(float(v)))(X_abs_3d)

    # Начальный colorbar с реальными значениями данных (не нормализованными 0-1)
    n_cb_ticks = 10
    cb_tickvals = [stats_min + (stats_max - stats_min) * i / n_cb_ticks for i in range(n_cb_ticks + 1)]
    cb_ticktext = [str(round(v)) if v >= 100 else f"{v:.2f}" for v in cb_tickvals]

    traces: list[go.Surface] = [
        go.Surface(
            x=X_plot_3d, y=Y_3d, z=Z_3d,
            customdata=X_hover_3d,
            colorscale="Turbo",
            cauto=False, cmin=stats_min, cmax=stats_max,
            colorbar=dict(
                title="Мощность", thickness=20, len=0.6, y=0.5,
                tickmode="array", tickvals=cb_tickvals, ticktext=cb_ticktext,
            ),
            contours=dict(
                x=dict(show=False, highlight=False),
                y=dict(show=False, highlight=False),
                z=dict(show=False, highlight=False),
            ),
            hovertemplate=(
                "<b>Время:</b> %{customdata}<br>"
                "<b>Расстояние:</b> %{y:.2f} м<br>"
                "<b>Значение:</b> %{z:.6f}<extra></extra>"
            ),
        )
    ]

    # Стены (4 шт.) + пол — с единым cmin/cmax для стабильности цветовой палитры
    for ri, ci in [
        (0, slice(None)),
        (-1, slice(None)),
        (slice(None), 0),
        (slice(None), -1),
    ]:
        if isinstance(ri, int):
            xw = np.array([X_plot_3d[ri, :], X_plot_3d[ri, :]])
            yw = np.array([Y_3d[ri, :], Y_3d[ri, :]])
            zw = np.array([Z_3d[ri, :], np.full(Z_3d.shape[1], z_floor)])
            cdw = np.array([X_hover_3d[ri, :], X_hover_3d[ri, :]])
            scw = np.array([Z_3d[ri, :], np.full(Z_3d.shape[1], z_floor)])
        else:
            xw = np.array([X_plot_3d[:, ci], X_plot_3d[:, ci]])
            yw = np.array([Y_3d[:, ci], Y_3d[:, ci]])
            zw = np.array([Z_3d[:, ci], np.full(Z_3d.shape[0], z_floor)])
            cdw = np.array([X_hover_3d[:, ci], X_hover_3d[:, ci]])
            scw = np.array([Z_3d[:, ci], np.full(Z_3d.shape[0], z_floor)])
        traces.append(_build_wall_surface(xw, yw, zw, cdw, scw, stats_min, stats_max))

    # Пол
    traces.append(
        _build_wall_surface(
            X_plot_3d, Y_3d,
            np.full_like(Z_3d, z_floor),
            X_hover_3d,
            np.full_like(Z_3d, z_floor),
            stats_min, stats_max,
        )
    )

    # Вычисляем центр данных для начальной камеры
    data_center_x = float(np.mean(time_3d_plot))
    data_center_y = float(np.mean(chan_3d))
    data_center_z = float(np.mean(Z_3d))

    # Маркер версии для идентификации генератора
    SCRIPT_VERSION = "v4.5-multi-p95-uint16"

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=f"Загрузка графической сцены... [{SCRIPT_VERSION}]", x=0.5),
        scene=dict(
            xaxis=dict(
                title=dict(text="Абсолютное время", font=dict(size=14, color="#fff")),
                tickvals=time_tickvals_3d,
                ticktext=time_ticktext_3d,
                showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                showline=True, linecolor="rgba(255,255,255,0.18)",
            ),
            yaxis=dict(
                title=dict(text="Расстояние (м)", font=dict(size=14, color="#fff")),
                showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                showline=True, linecolor="rgba(255,255,255,0.18)",
            ),
            zaxis=dict(
                title=dict(text="Мощность", font=dict(size=14, color="#fff")),
                showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                showline=True, linecolor="rgba(255,255,255,0.18)",
            ),
            camera=dict(
                eye=dict(x=1.6, y=-1.4, z=1.0),
                center=dict(x=0, y=0, z=0),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.6, y=1.4, z=1.2),
        ),
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    out_path = Path(output_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("[INFO] Экспорт Plotly HTML...")
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)

    html_content = out_path.read_text(encoding="utf-8")
    print("[INFO] Сериализация мастер-датасета и генерация JS...")

    # ===== JS: данные (f-string — здесь внедряются данные из Python) =====
    # Определяются ДО функций, чтобы гарантировать доступность в HUD
    js_vars = f"""
  window.masterData = {json.dumps(all_metrics_master)};
  window.masterTimeAbs = {json.dumps(master_time)};
  window.masterChannels = {json.dumps(master_channels)};
  window.masterMaxDim = {min(1600, max(master_n_win, master_n_chan))};
  window.metricStats = {json.dumps(metric_stats)};
  window.currentMetricKey = "{metric_key}";
  window.nWindowsOriginal = {n_windows};
  window.nChannelsOriginal = {n_channels};
  window.timeAxisStart = {float(time_axis[0])};
  window.timeAxisEnd = {float(time_axis[-1])};
  window.distanceMin = {float(channels_axis[0])};
  window.distanceMax = {float(channels_axis[-1])};
  window.formatName = "{str(attrs.get('format', 'activity'))}";
  window.defaultRes3D = 200;
  window.defaultRes2D = {min(1200, max(master_n_win, master_n_chan))};
  window.startTimeAbs = {start_time_offset};
  window.sampleRateHz = {sr};
  window.isLargeTimestamp = {str(time_axis[0] > 1e9).lower()};
  window.scriptVersion = "{SCRIPT_VERSION}";
  window.metricUnits = "{str(attrs.get('metric_units', ''))}";
  window.distanceUnits = "{str(attrs.get('distance_units', 'м'))}";
"""

    # ===== JS: утилитарные функции (raw string — без f-интерполяции) =====
    js_utils = r"""<script>
(function(){
  var baseSpeed = 0.025;
  var speedMultiplier = 1.0;
  var keysPressed = {};
  var isRelayouting = false;
  var pendingCamera = null;

  window.currentViewMode = "3d";
  window.currentRes = 200;
  window.currentResampled = null;
  window.filterNonPositive = true;

  // === Декодирование Base64 uint16 с восстановлением реальных значений ===
  function decodeMasterArray(encoded) {
    var parts = encoded.split('|');
    var isU16 = parts[0] === 'u16';
    var rows = +(isU16 ? parts[1] : parts[0]);
    var cols = +(isU16 ? parts[2] : parts[1]);
    var lo = isU16 ? +parts[3] : 0;
    var hi = isU16 ? +parts[4] : 0;
    var binary = atob(isU16 ? parts[5] : parts[2]);
    var buf = new ArrayBuffer(binary.length);
    var view = new Uint8Array(buf);
    for (var i = 0; i < binary.length; i++) view[i] = binary.charCodeAt(i);
    var typed = isU16 ? new Uint16Array(buf) : new Float32Array(buf);
    var scale = isU16 && hi > lo ? (hi - lo) / 65535.0 : 0;
    var result = [];
    for (var r = 0; r < rows; r++) {
      var row = new Array(cols);
      var off = r * cols;
      for (var c = 0; c < cols; c++) {
        row[c] = isU16 ? lo + typed[off + c] * scale : typed[off + c];
      }
      result.push(row);
    }
    return result;
  }

  // Декодируем все метрики из Base64 в 2D-массивы при загрузке
  function decodeMasterData() {
    var keys = Object.keys(window.masterData);
    for (var ki = 0; ki < keys.length; ki++) {
      var key = keys[ki];
      var val = window.masterData[key];
      if (typeof val === 'string' && val.indexOf('|') > 0) {
        window.masterData[key] = decodeMasterArray(val);
      }
    }
    console.log('[h5-to-html] Мастер-данные декодированы: ' + keys.length + ' метрик');
  }

  // === Форматирование времени ===
  function formatTimeAxis(val) {
    if (window.isLargeTimestamp && val > 1e9) {
      var d = new Date(val * 1000);
      var YYYY = d.getUTCFullYear();
      var MM = String(d.getUTCMonth()+1).padStart(2,'0');
      var DD = String(d.getUTCDate()).padStart(2,'0');
      var hh = String(d.getUTCHours()).padStart(2,'0');
      var mm = String(d.getUTCMinutes()).padStart(2,'0');
      var ss = String(d.getUTCSeconds()).padStart(2,'0');
      return YYYY+'-'+MM+'-'+DD+' '+hh+':'+mm+':'+ss;
    }
    return val.toFixed(3);
  }
  function formatHoverTime(val) {
    if (window.isLargeTimestamp && val > 1e9) {
      var d = new Date(val * 1000);
      var YYYY = d.getUTCFullYear();
      var MM = String(d.getUTCMonth()+1).padStart(2,'0');
      var DD = String(d.getUTCDate()).padStart(2,'0');
      var hh = String(d.getUTCHours()).padStart(2,'0');
      var mm = String(d.getUTCMinutes()).padStart(2,'0');
      var ss = String(d.getUTCSeconds()).padStart(2,'0');
      var ms = String(d.getUTCMilliseconds()).padStart(3,'0');
      return YYYY+'-'+MM+'-'+DD+' '+hh+':'+mm+':'+ss+'.'+ms+' UTC';
    }
    return val.toFixed(3) + ' с';
  }

  // === Ресэмплинг из мастер-датасета ===
  function getMasterDims() {
    var firstKey = Object.keys(window.masterData)[0];
    if (!firstKey) return {nWin: 0, nChan: 0};
    var Z = window.masterData[firstKey];
    return {nWin: Z.length, nChan: Z[0].length};
  }

  function resampleData(targetMaxDim) {
    var dims = getMasterDims();
    var nWin = dims.nWin, nChan = dims.nChan;
    var stepW = Math.max(1, Math.round(nWin / targetMaxDim));
    var stepC = Math.max(1, Math.round(nChan / targetMaxDim));

    var time = [], chan = [];
    for (var i = 0; i < nWin; i += stepW) time.push(window.masterTimeAbs[i]);
    for (var j = 0; j < nChan; j += stepC) chan.push(window.masterChannels[j]);

    var z = {};
    var keys = Object.keys(window.masterData);
    for (var ki = 0; ki < keys.length; ki++) {
      var mk = keys[ki];
      var src = window.masterData[mk];
      var dst = [];
      for (var i = 0; i < nWin; i += stepW) {
        var row = [];
        for (var j = 0; j < nChan; j += stepC) row.push(src[i][j]);
        dst.push(row);
      }
      z[mk] = dst;
    }

    var t0 = time[0];
    var shiftTime = Math.abs(t0) > 1000000;
    var timeCentered = [];
    for (var i = 0; i < time.length; i++) {
      timeCentered.push(shiftTime ? (time[i] - t0) : time[i]);
    }

    return {
      z: z, timeAbs: time, timeCentered: timeCentered,
      channels: chan, shiftTime: shiftTime, t0: t0,
      stepW: stepW, stepC: stepC,
      nWin: time.length, nChan: chan.length
    };
  }

  function computeTicks(timeArr, maxTicks) {
    var n = timeArr.length;
    if (n <= maxTicks) {
      return {
        vals: timeArr.slice(),
        text: timeArr.map(function(t){ return formatTimeAxis(t); })
      };
    }
    var indices = [];
    for (var i = 0; i < maxTicks; i++) {
      indices.push(Math.round(i * (n - 1) / (maxTicks - 1)));
    }
    return {
      vals: indices.map(function(i){ return timeArr[i]; }),
      text: indices.map(function(i){ return formatTimeAxis(timeArr[i]); })
    };
  }

  function ensureResampled(targetRes) {
    if (window.currentResampled && window.currentResampled._targetRes === targetRes)
      return window.currentResampled;
    window.currentResampled = resampleData(targetRes);
    window.currentResampled._targetRes = targetRes;
    return window.currentResampled;
  }

  function transpose2D(Z) {
    if (!Z || !Z.length) return [];
    var nRows = Z.length, nCols = Z[0].length;
    var Zt = [];
    for (var j = 0; j < nCols; j++) {
      var row = [];
      for (var i = 0; i < nRows; i++) row.push(Z[i][j]);
      Zt.push(row);
    }
    return Zt;
  }

  // === Явные палитры (не полагаемся на Plotly — он может подменить строку) ===
  var COLORSCALES = {
    "Inferno": [
      [0.00, '#000004'], [0.07, '#0c0926'], [0.13, '#1b0c41'], [0.20, '#2d0e59'],
      [0.27, '#3f0f71'], [0.33, '#51127c'], [0.40, '#641a80'], [0.47, '#782281'],
      [0.53, '#8c2980'], [0.60, '#a1307e'], [0.67, '#b73779'], [0.73, '#cb416d'],
      [0.80, '#de4f5d'], [0.87, '#ed6925'], [0.93, '#f7a51a'], [1.00, '#fcffa4']
    ],
    "Viridis": [
      [0.00, '#440154'], [0.07, '#482777'], [0.13, '#3e4989'], [0.20, '#31688e'],
      [0.27, '#26828e'], [0.33, '#1f9e89'], [0.40, '#35b779'], [0.47, '#6ece58'],
      [0.53, '#86d549'], [0.60, '#a5dc36'], [0.67, '#c2e327'], [0.73, '#d4e71f'],
      [0.80, '#e2e418'], [0.87, '#eddf13'], [0.93, '#f5d90e'], [1.00, '#fde725']
    ],
    "Plasma": [
      [0.00, '#0d0887'], [0.07, '#3b049a'], [0.13, '#5c01a6'], [0.20, '#7e03a8'],
      [0.27, '#9c179e'], [0.33, '#b62f8a'], [0.40, '#cb4679'], [0.47, '#db5c68'],
      [0.53, '#e87254'], [0.60, '#f18941'], [0.67, '#f7a232'], [0.73, '#fbbb24'],
      [0.80, '#f7d13d'], [0.87, '#f2e768'], [0.93, '#fcffa4'], [1.00, '#f0f921']
    ],
    "Magma": [
      [0.00, '#000004'], [0.07, '#0c0926'], [0.13, '#1f0c48'], [0.20, '#360f6e'],
      [0.27, '#4e0f6e'], [0.33, '#6a176e'], [0.40, '#86216b'], [0.47, '#a02e5f'],
      [0.53, '#b8404e'], [0.60, '#cf553d'], [0.67, '#e26d2e'], [0.73, '#f08c21'],
      [0.80, '#f9af17'], [0.87, '#fcd319'], [0.93, '#fcffa4'], [1.00, '#fcffa4']
    ],
    "Cividis": [
      [0.00, '#00224e'], [0.07, '#002960'], [0.13, '#17356d'], [0.20, '#254176'],
      [0.27, '#324d7b'], [0.33, '#3f597e'], [0.40, '#4c6581'], [0.47, '#597283'],
      [0.53, '#667f85'], [0.60, '#748c87'], [0.67, '#839988'], [0.73, '#95a682'],
      [0.80, '#a8b37a'], [0.87, '#bcc06f'], [0.93, '#d2cd5f'], [1.00, '#fde725']
    ],
    "Turbo": [
      [0.00, '#30123b'], [0.07, '#4145ab'], [0.13, '#4675ed'], [0.20, '#39a2fc'],
      [0.27, '#1bd0d5'], [0.33, '#24f0a0'], [0.40, '#5dfc6a'], [0.47, '#a4fc3c'],
      [0.53, '#d1e834'], [0.60, '#f0c531'], [0.67, '#fe9b2d'], [0.73, '#f56b19'],
      [0.80, '#e03e11'], [0.87, '#b91a09'], [0.93, '#820902'], [1.00, '#300901']
    ],
    "Hot": [
      [0.00, '#000000'], [0.13, '#5a0000'], [0.25, '#b30000'], [0.38, '#ff0000'],
      [0.50, '#ff6600'], [0.63, '#ffaa00'], [0.75, '#ffdd00'], [0.88, '#ffff66'],
      [1.00, '#ffffff']
    ],
    "YlOrRd": [
      [0.00, '#ffffcc'], [0.13, '#ffeda0'], [0.25, '#fed976'], [0.38, '#feb24c'],
      [0.50, '#fd8d3c'], [0.63, '#fc4e2a'], [0.75, '#e31a1c'], [0.88, '#bd0026'],
      [1.00, '#800026']
    ],
    "Blues": [
      [0.00, '#f7fbff'], [0.13, '#deebf7'], [0.25, '#c6dbef'], [0.38, '#9ecae1'],
      [0.50, '#6baed6'], [0.63, '#4292c6'], [0.75, '#2171b5'], [0.88, '#08519c'],
      [1.00, '#08306b']
    ],
    "Greens": [
      [0.00, '#f7fcf5'], [0.13, '#e5f5e0'], [0.25, '#c7e9c0'], [0.38, '#a1d99b'],
      [0.50, '#74c476'], [0.63, '#41ab5d'], [0.75, '#238b45'], [0.88, '#006d2c'],
      [1.00, '#00441b']
    ],
    "Reds": [
      [0.00, '#fff5f0'], [0.13, '#fee0d2'], [0.25, '#fcbba1'], [0.38, '#fc9272'],
      [0.50, '#fb6a4a'], [0.63, '#ef3b2c'], [0.75, '#cb181d'], [0.88, '#a50f15'],
      [1.00, '#67000d']
    ],
    "RdBu": [
      [0.00, '#67001f'], [0.10, '#b2182b'], [0.20, '#d6604d'], [0.30, '#f4a582'],
      [0.40, '#fddbc7'], [0.50, '#f7f7f7'], [0.60, '#d1e5f0'], [0.70, '#92c5de'],
      [0.80, '#4393c3'], [0.90, '#2166ac'], [1.00, '#053061']
    ],
    "Spectral": [
      [0.00, '#9e0142'], [0.10, '#d53e4f'], [0.20, '#f46d43'], [0.30, '#fdae61'],
      [0.40, '#fee08b'], [0.50, '#ffffbf'], [0.60, '#e6f598'], [0.70, '#abdda4'],
      [0.80, '#66c2a5'], [0.90, '#3288bd'], [1.00, '#5e4fa2']
    ],
    "Rainbow": [
      [0.00, '#ff0000'], [0.13, '#ff8800'], [0.25, '#ffff00'], [0.38, '#00ff00'],
      [0.50, '#00ffff'], [0.63, '#0000ff'], [0.75, '#8800ff'], [0.88, '#ff00ff'],
      [1.00, '#ff0088']
    ],
    "Earth": [
      [0.00, '#000033'], [0.13, '#1a4d80'], [0.25, '#339966'], [0.38, '#66cc33'],
      [0.50, '#99cc00'], [0.63, '#cccc00'], [0.75, '#cc9933'], [0.88, '#996633'],
      [1.00, '#663300']
    ],
    "Electric": [
      [0.00, '#000000'], [0.13, '#1a0033'], [0.25, '#660099'], [0.38, '#cc0066'],
      [0.50, '#ff0033'], [0.63, '#ff6600'], [0.75, '#ffcc00'], [0.88, '#ffff66'],
      [1.00, '#ffffff']
    ],
    "Blackbody": [
      [0.00, '#000000'], [0.13, '#1a0000'], [0.25, '#660000'], [0.38, '#cc3300'],
      [0.50, '#ff6600'], [0.63, '#ffaa00'], [0.75, '#ffdd44'], [0.88, '#ffffaa'],
      [1.00, '#ffffff']
    ],
    "Portland": [
      [0.00, '#0a0a1a'], [0.13, '#1a3a6a'], [0.25, '#2a7a5a'], [0.38, '#4aba3a'],
      [0.50, '#8ada2a'], [0.63, '#caca1a'], [0.75, '#da9a1a'], [0.88, '#ca5a1a'],
      [1.00, '#ba1a1a']
    ]
  };
  window.currentColorscale = "Turbo";

  // === Фиксированный цветовой диапазон из предвычисленной статистики ===
  // Использует перцентильную обрезку (p2–p98), чтобы 96% данных
  // занимали всю палитру — центр не будет бесцветным.
  window.colorMode = 'percentile';  // 'full' | 'percentile' | 'per_channel'
  window.colorScaleType = 'linear';  // 'linear' | 'log'
  window.zAxisPower = 1.0;  // Степень сжатия Z-оси: 1.0=линейная, 0.5=корень, 0.25=4-й корень
  window.colorGamma = 1.0;  // gamma < 1 — детализация высоких значений, > 1 — низких

  // Получить текущую палитру с учётом гамма-коррекции.
  // Гамма применяется к позициям стопов цветов: pow(pos, gamma).
  // gamma < 1 — растягивает палитру на высоких значениях,
  // gamma > 1 — растягивает на низких значениях.
  function getActiveColorscale() {
    var base = COLORSCALES[window.currentColorscale];
    var gamma = window.colorGamma || 1.0;
    if (gamma === 1.0) return base;
    return base.map(function(stop) {
      return [Math.pow(stop[0], gamma), stop[1]];
    });
  }

  // === Преобразование цветового пространства ===
  // В log-режиме surfacecolor проходит через toColorSpace, а cmin/cmax — тоже.
  // Colorbar tickvals в цветовом пространстве, ticktext — реальные значения.
  var LOG_EPS = 1e-2;  // защита от log(0)

  function toColorSpace(val) {
    if (window.colorScaleType === 'log') {
      if (val <= 0) console.warn('[h5-to-html] toColorSpace: log10(val<=0) — clamped to LOG_EPS, val=', val);
      return Math.log10(Math.max(val, LOG_EPS));
    }
    return val;
  }
  function fromColorSpace(val) {
    if (window.colorScaleType === 'log') {
      return Math.pow(10, val);
    }
    return val;
  }

  // Безопасная конвертация в цветовое пространство с клампингом результата.
  // Гарантирует, что результат ∈ [csMin, csMax].
  // В log-режиме: если val <= 0 или val < dataMin, поднимаем до dataMin
  // (который в log-режиме всегда > 0 благодаря getColorRange).
  // Это предотвращает surfacecolor = -10 при val=0, что ломало WebGL-рендер.
  function toColorSpaceClamped(val, dataMin, csMin, csMax) {
    if (window.colorScaleType === 'log') {
      // В log-режиме dataMin > 0 (гарантируется getColorRange)
      var safeVal = val < dataMin ? dataMin : val;
      if (safeVal <= 0) safeVal = LOG_EPS;
      var result = Math.log10(safeVal);
      return result < csMin ? csMin : (result > csMax ? csMax : result);
    }
    // Линейный режим: просто клампим
    return val < csMin ? csMin : (val > csMax ? csMax : val);
  }

  // Быстрый клампинг в цветовом пространстве (для интерполированных значений)
  function clampCS(val, csMin, csMax) {
    return val < csMin ? csMin : (val > csMax ? csMax : val);
  }

  // === Преобразование Z-оси (активность) ===
  // Степенное сжатие: sign(x) * |x|^p, где p = window.zAxisPower
  // p = 1.0 — линейная (без сжатия)
  // p = 0.5 — корень (умеренное сжатие)
  // p = 0.25 — корень 4-й степени (сильное сжатие)
  // Никогда не уходит в минус для x >= 0.
  // Чем меньше p, тем больше визуального пространства получают малые значения.

  function toZAxis(val) {
    // Конвертирует реальное значение Z в координату оси
    var p = window.zAxisPower;
    if (p >= 0.999) return val;  // линейная — без вычислений
    var s = val < 0 ? -1 : 1;
    return s * Math.pow(Math.abs(val), p);
  }
  function fromZAxis(val) {
    // Конвертирует координату оси Z обратно в реальное значение
    var p = window.zAxisPower;
    if (p >= 0.999) return val;  // линейная
    var s = val < 0 ? -1 : 1;
    return s * Math.pow(Math.abs(val), 1.0 / p);
  }

  // === Гауссово сглаживание Z-матрицы ===
  // sigma = 0 — без сглаживания (острые углы)
  // sigma = 1..5 — степень размытия (скругление пиков и впадин)
  window.zSmoothSigma = 0;

  // Генерация 1D ядра Гаусса
  function gaussianKernel1D(sigma) {
    if (sigma <= 0) return [1];
    var radius = Math.ceil(sigma * 3);
    var kernel = [];
    var sum = 0;
    for (var i = -radius; i <= radius; i++) {
      var w = Math.exp(-(i * i) / (2 * sigma * sigma));
      kernel.push(w);
      sum += w;
    }
    // Нормализация
    for (var k = 0; k < kernel.length; k++) kernel[k] /= sum;
    return kernel;
  }

  // Свертка 1D массива с ядром (края — продление)
  function convolve1D(arr, kernel) {
    var n = arr.length;
    var r = kernel.length >> 1;  // radius
    var result = [];
    for (var i = 0; i < n; i++) {
      var val = 0;
      for (var k = 0; k < kernel.length; k++) {
        var idx = i + k - r;
        // Продление края (clamp)
        if (idx < 0) idx = 0;
        if (idx >= n) idx = n - 1;
        val += arr[idx] * kernel[k];
      }
      result.push(val);
    }
    return result;
  }

  // 2D Гауссово размытие (разделимое: строки → транспозиция → строки → транспозиция)
  // Сложность O(rows × cols × kernelSize) вместо O(rows² × cols × kernelSize)
  function gaussianBlur2D(matrix, sigma) {
    if (sigma <= 0) return matrix;
    var kernel = gaussianKernel1D(sigma);
    var rows = matrix.length;
    var cols = matrix[0].length;
    // Проход по строкам
    var temp = [];
    for (var i = 0; i < rows; i++) {
      temp.push(convolve1D(matrix[i], kernel));
    }
    // Транспозиция
    var transposed = [];
    for (var j = 0; j < cols; j++) {
      var col = [];
      for (var i = 0; i < rows; i++) col.push(temp[i][j]);
      transposed.push(col);
    }
    // Свёртка по столбцам (теперь — строки транспонированной матрицы)
    var blurredT = [];
    for (var j = 0; j < cols; j++) {
      blurredT.push(convolve1D(transposed[j], kernel));
    }
    // Обратная транспозиция
    var result = [];
    for (var i = 0; i < rows; i++) {
      var row = [];
      for (var j = 0; j < cols; j++) row.push(blurredT[j][i]);
      result.push(row);
    }
    return result;
  }

  // Применить сглаживание к Z-матрице с учётом настроек
  function applySmoothing(Z) {
    return gaussianBlur2D(Z, window.zSmoothSigma);
  }

  function formatRealValue(val) {
    // Красивое форматирование реального значения для colorbar/hover
    if (Math.abs(val) >= 1000) return val.toExponential(1);
    if (Math.abs(val) >= 100) return Math.round(val).toString();
    if (Math.abs(val) >= 1) return val.toFixed(2);
    if (Math.abs(val) >= 0.01) return val.toFixed(4);
    return val.toExponential(2);
  }

  function getColorRange(metricKey) {
    var stats = window.metricStats[metricKey];
    if (!stats) { console.warn('[h5-to-html] getColorRange: no stats for', metricKey, '— fallback {0,1}'); return {min: 0, max: 1}; }
    var lo, hi;
    if (window.colorMode === 'percentile' && stats.p2 !== undefined) {
      lo = stats.p2; hi = stats.p98;
    } else {
      lo = stats.min; hi = stats.max;
    }
    if (window.filterNonPositive && lo < 0) {
      lo = 0;
    }
    // В log-режиме lo должен быть > 0 и ОСМЫСЛЕННЫМ (основанным на данных)
    // Никогда не используем LOG_EPS для диапазона — только для отдельных точек данных!
    // LOG_EPS = 1e-10 растягивает лог-шкалу на 10 порядков, и все реальные данные
    // оказываются сжатыми в красный конец палитры.
    if (window.colorScaleType === 'log' && lo <= 0) {
      var origLo = lo;
      if (stats.p2 > 0) lo = stats.p2;
      else if (stats.p5 > 0) lo = stats.p5;
      else if (stats.min > 0) lo = stats.min;
      else if (hi > 0) lo = hi * 1e-2;
      else lo = 1e-2;                      // абсолютный минимум для пустых метрик
      console.warn('[h5-to-html] getColorRange: log mode — lo was', origLo, '→ clamped to', lo);
    }
    if (hi <= lo) { hi = lo + 1; }  // защита от пустого диапазона
    return {min: lo, max: hi};
  }
"""

    # ===== JS: приложение (raw string — HUD, трассировки, навигация) =====
    js_app = r"""
  // Декодируем Base64 данные (window.masterData уже определён в js_vars)
  if (typeof decodeMasterData === 'function') decodeMasterData();

  // === Генератор заголовков с мета-статистикой ===
  function getTitleText(metricKey, modeName) {
    var n_rows = window.nWindowsOriginal;
    var n_cols = window.nChannelsOriginal;
    var stats = window.metricStats[metricKey] || {min: 0, max: 0, mean: 0};
    var time_start = window.timeAxisStart;
    var time_end = window.timeAxisEnd;
    var time_duration = time_end - time_start;
    var dist_min = window.distanceMin;
    var dist_max = window.distanceMax;
    var startLabel = formatHoverTime(time_start);

    return "<b>Мониторинг активности (" + window.formatName + ")</b> — " + modeName +
           " | Метрика: <b style='color:#60a5fa;'>" + metricKey.toUpperCase() +
           "</b> | Временных окон: " + n_rows + "<br>" +
           "<span style='font-size:13px; color:#ccc;'>Старт: <b>" + startLabel +
           "</b> (длительность: " + time_duration.toFixed(1) + " с) | " +
           "Расстояние: <b>" + dist_min.toFixed(1) + " ... " + dist_max.toFixed(1) +
           " м</b> (" + n_cols + " каналов) | " +
           "Значения (Z): <b>[" + stats.min.toFixed(2) + " ... " + stats.max.toFixed(2) +
           "]</b> (ср: " + stats.mean.toFixed(2) + ")</span>";
  }

  // === 3D layout (используется и для 3D, и для плоского 2D) ===
  function get3DLayout(titleText, rs, isFlat, savedCamera) {
    var ticks = computeTicks(rs.timeAbs, 8);
    var tickValsCentered = [];
    for (var i = 0; i < ticks.vals.length; i++) {
      tickValsCentered.push(rs.shiftTime ? (ticks.vals[i] - rs.t0) : ticks.vals[i]);
    }
    // Камера: при переключении сохраняем текущую позицию в 3D,
    // чтобы при смене прореживания/метрики ракурс не сбрасывался
    var defaultCam3D = { eye: {x:1.6, y:-1.4, z:1.0}, center: {x:0, y:0, z:0}, up: {x:0, y:0, z:1} };
    var cam;
    if (isFlat) {
      cam = { eye: {x: 0, y: 0, z: 2.5}, center: {x: 0, y: 0, z: 0}, up: {x: 0, y: 1, z: 0} };
    } else if (savedCamera) {
      cam = savedCamera;
    } else {
      cam = defaultCam3D;
    }
    return {
      title: { text: titleText, font: { color: "#ffffff", size: 16 }, x: 0.5 },
      scene: {
        xaxis: {
          title: { text: "Абсолютное время", font: { size: 14, color: "#ffffff" } },
          tickvals: tickValsCentered, ticktext: ticks.text,
          tickfont: { size: 11, color: "#ccc" },
          showspikes: false, spikesides: false,
          showgrid: true, gridcolor: "rgba(255,255,255,0.06)", gridwidth: 1,
          showline: true, linecolor: "rgba(255,255,255,0.18)", linewidth: 1
        },
        yaxis: {
          title: { text: "Расстояние (м)", font: { size: 14, color: "#ffffff" } },
          tickfont: { size: 11, color: "#ccc" },
          showspikes: false, spikesides: false,
          showgrid: true, gridcolor: "rgba(255,255,255,0.06)", gridwidth: 1,
          showline: true, linecolor: "rgba(255,255,255,0.18)", linewidth: 1
        },
        zaxis: (function() {
          var isCompressed = window.zAxisPower < 0.999;
          var zCfg = {
            title: isFlat ? "" : { text: isCompressed ? "Мощность (x^" + window.zAxisPower.toFixed(2) + ")" + (window.metricUnits ? ' ' + window.metricUnits : '') : "Мощность" + (window.metricUnits ? ' (' + window.metricUnits + ')' : ''), font: { size: 14, color: "#ffffff" } },
            tickfont: { size: 11, color: "#ccc" },
            showspikes: false, spikesides: false,
            showticklabels: !isFlat,
            range: isFlat ? [-1, 1] : undefined,
            showgrid: true, gridcolor: "rgba(255,255,255,0.06)", gridwidth: 1,
            showline: true, linecolor: "rgba(255,255,255,0.18)", linewidth: 1
          };
          // При сжатии Z — кастомные тики: позиции в осевых координатах,
          // подписи — реальные значения (чтобы ось была читаемой)
          if (isCompressed && !isFlat) {
            var stats = window.metricStats[window.currentMetricKey];
            if (stats) {
              var zMin = Math.min(0, stats.min);
              var zMax = stats.max;
              // Генерируем «красивые» реальные значения для тиков
              var niceSteps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];
              var step = 1;
              for (var si = 0; si < niceSteps.length; si++) {
                if (niceSteps[si] * 8 >= zMax - zMin) { step = niceSteps[si]; break; }
              }
              var tickVals = [], tickText = [];
              for (var rv = 0; rv <= zMax + step * 0.5; rv += step) {
                tickVals.push(toZAxis(rv));
                tickText.push(rv >= 10 ? Math.round(rv).toString() : rv.toFixed(1));
              }
              // Добавляем отрицательные тики если есть
              if (zMin < 0) {
                for (var rv = -step; rv >= zMin - step * 0.5; rv -= step) {
                  tickVals.push(toZAxis(rv));
                  tickText.push(rv >= 10 || rv <= -10 ? Math.round(rv).toString() : rv.toFixed(1));
                }
              }
              zCfg.tickmode = "array";
              zCfg.tickvals = tickVals;
              zCfg.ticktext = tickText;
            } else {
              zCfg.nticks = 10;
              zCfg.tickformat = ".3f";
            }
          } else {
            zCfg.nticks = 10;
            zCfg.tickformat = ".3f";
          }
          return zCfg;
        })(),
        camera: cam,
        aspectmode: "manual",
        aspectratio: isFlat ? {x: 1.6, y: 1.4, z: 0.01} : {x: 1.6, y: 1.4, z: 1.2},
      },
      paper_bgcolor: "#0a0a14", plot_bgcolor: "#0a0a14",
      margin: { l: 0, r: 0, b: 0, t: 40 }, autosize: true
    };
  }

  // === Общий помощник для mesh-массивов ===
  function buildMeshArrays(rs) {
    var nWin = rs.timeCentered.length, nChan = rs.channels.length;
    var X_plot = [], Y = [], X_hover = [];
    // Предвычислить hover-строки один раз на строку (одинаковы для всех j в строке)
    var hoverCache = [];
    for (var i = 0; i < nWin; i++) {
      hoverCache.push(formatHoverTime(rs.timeAbs[i]));
    }
    for (var i = 0; i < nWin; i++) {
      var xr = [], yr = [], xhr = [];
      var tc = rs.timeCentered[i];
      var ht = hoverCache[i];
      for (var j = 0; j < nChan; j++) {
        xr.push(tc);
        yr.push(rs.channels[j]);
        xhr.push(ht);
      }
      X_plot.push(xr); Y.push(yr); X_hover.push(xhr);
    }
    return {X_plot: X_plot, Y: Y, X_hover: X_hover, nWin: nWin, nChan: nChan};
  }

  function fillArr(len, val) {
    var a = []; for(var k=0; k<len; k++) a.push(val); return a;
  }

  // === 3D surfaces & walls ===
  function get3DTraces(metricKey, rs) {
    var Z = rs.z[metricKey];
    var colorRange = getColorRange(metricKey);
    var mesh = buildMeshArrays(rs);
    var X_plot = mesh.X_plot, Y = mesh.Y, X_hover = mesh.X_hover;
    var nWin = mesh.nWin, nChan = mesh.nChan;

    // z_floor: пол поверхности
    var z_floor = Infinity;
    for (var i = 0; i < nWin; i++)
      for (var j = 0; j < nChan; j++)
        if (isFinite(Z[i][j]) && Z[i][j] < z_floor) z_floor = Z[i][j];
    if (window.filterNonPositive && z_floor < 0) z_floor = 0;

    // Обнуление отрицательных значений → 0 (видимые, нулевые тоже показываются)
    var Zd = Z;
    if (window.filterNonPositive) {
      Zd = [];
      for (var i = 0; i < nWin; i++) {
        var row = [];
        for (var j = 0; j < nChan; j++) {
          row.push(Z[i][j] < 0 ? 0 : Z[i][j]);
        }
        Zd.push(row);
      }
    }

    // ═══ Гауссово сглаживание ═══
    // Применяется к реальным данным до Z-преобразования,
    // чтобы скруглить углы поверхности
    if (window.zSmoothSigma > 0) {
      Zd = applySmoothing(Zd);
      // Пересчитать z_floor после сглаживания
      z_floor = Infinity;
      for (var i = 0; i < nWin; i++)
        for (var j = 0; j < nChan; j++)
          if (isFinite(Zd[i][j]) && Zd[i][j] < z_floor) z_floor = Zd[i][j];
      if (window.filterNonPositive && z_floor < 0) z_floor = 0;
    }

    // ═══ Z-ось: преобразование координат ═══
    // ZdReal — реальные значения (для hover/customdata)
    // ZdAxis — значения в координатах оси (toZAxis), для рендеринга поверхности
    var ZdReal = Zd;  // реальные значения (или Zd если не фильтровали)
    var ZdAxis = [];
    for (var i = 0; i < nWin; i++) {
      var row = [];
      for (var j = 0; j < nChan; j++) {
        row.push(toZAxis(Zd[i][j]));
      }
      ZdAxis.push(row);
    }
    var z_floor_axis = toZAxis(z_floor);

    var traces = [];

    // ═══ Главная поверхность — простая одинарная сетка ═══
    // Раньше использовался N_LEVELS-подраздел для градиента по высоте,
    // но это создавало дегенеративные треугольники (одинаковые x,y на разных z),
    // которые WebGL рендерер Plotly отбрасывал → видны были только стенки.
    // Теперь: одна поверхность, surfacecolor = toColorSpaceClamped(Zd).

    // ═══ Глобальная нормализация цвета (единая шкала на весь датасет) ═══
    // Используем getColorRange() — глобальный min/max (или p2/p98) по ВСЕМ данным.
    // Одинаковое значение → одинаковый цвет во всех каналах.
    // В log10-режиме: log10(val) перед нормализацией, colorbar показывает реальные значения.
    var cRange = getColorRange(metricKey);
    var isLogColor = window.colorScaleType === 'log';

    // Цветовой диапазон в пространстве нормализации (линейном или лог-пространстве)
    var cMin, cMax;
    if (isLogColor) {
      cMin = Math.log10(Math.max(cRange.min, LOG_EPS));
      cMax = Math.log10(Math.max(cRange.max, LOG_EPS));
      if (cMax <= cMin) cMax = cMin + 1;
      // Sanity check: ограничиваем лог-диапазон ≤ 4 порядков
      // Если диапазон шире — данные будут сжаты в красный конец (как было с LOG_EPS)
      var MAX_LOG_SPAN = 2;
      if ((cMax - cMin) > MAX_LOG_SPAN) {
        var origCMin = cMin;
        cMin = cMax - MAX_LOG_SPAN;
        console.warn('[h5-to-html] cMin clamped: log range too wide, was', origCMin, '→', cMin, '(span capped at', MAX_LOG_SPAN, 'orders)');
      }
    } else {
      cMin = cRange.min;
      cMax = cRange.max;
    }
    var cSpan = cMax - cMin;
    if (cSpan <= 0) cSpan = 1;

    // Нормализованные значения [0..1] для surfacecolor
    var ZdNorm = [];
    for (var i = 0; i < nWin; i++) {
      var row = [];
      for (var j = 0; j < nChan; j++) {
        var val = Zd[i][j];
        if (isLogColor) {
          val = Math.log10(Math.max(val, LOG_EPS));
        }
        var norm = (val - cMin) / cSpan;
        row.push(norm < 0 ? 0 : (norm > 1 ? 1 : norm));
      }
      ZdNorm.push(row);
    }

    // Цветовой диапазон для нормализованных данных: всегда [0, 1]
    var cRangeCS = {min: 0, max: 1};

    // ═══ Явные подписи colorbar — реальные значения данных ═══
    var nColorTicks = 10;
    var colorTickVals = [], colorTickText = [];
    for (var ct = 0; ct <= nColorTicks; ct++) {
      var frac = ct / nColorTicks;
      colorTickVals.push(frac);
      // Обратно из нормализованного [0,1] → реальное значение
      var realVal;
      if (isLogColor) {
        realVal = Math.pow(10, cMin + frac * cSpan);
      } else {
        realVal = cMin + frac * cSpan;
      }
      colorTickText.push(formatRealValue(realVal));
    }

    // surfacecolor в цветовом пространстве для каждой ячейки
    var mainSC = [];
    for (var i = 0; i < nWin; i++) {
      var scr = [];
      for (var j = 0; j < nChan; j++) {
        // ZdNorm уже [0,1], просто клампим
        var v = ZdNorm[i][j];
        scr.push(v < 0 ? 0 : (v > 1 ? 1 : v));
      }
      mainSC.push(scr);
    }

    // customdata: [время, реальное_значение] для hover (Z на оси может быть log)
    var mainCD = [];
    for (var i = 0; i < nWin; i++) {
      var cdr = [];
      for (var j = 0; j < nChan; j++) {
        cdr.push([X_hover[i][j], ZdReal[i][j]]);
      }
      mainCD.push(cdr);
    }

    traces.push({
      type: "surface",
      x: X_plot, y: Y, z: ZdAxis,
      surfacecolor: mainSC,
      customdata: mainCD,
      colorscale: getActiveColorscale(),
      cauto: false,
      cmin: cRangeCS.min, cmax: cRangeCS.max,
      colorbar: {
        title: { text: (isLogColor ? "log10 " : "") + metricKey.toUpperCase() + (window.metricUnits ? " (" + window.metricUnits + ")" : ""), font: { color: "#fff", size: 12 } },
        thickness: 25, len: 0.7, y: 0.5,
        tickfont: { color: "#ddd", size: 11 }, titlefont: { color: "#fff" },
        tickmode: "array",
        tickvals: colorTickVals,
        ticktext: colorTickText,
        outlinewidth: 1, outlinecolor: "rgba(255,255,255,0.15)"
      },
      contours: {
        x: { show: false, highlight: false },
        y: { show: false, highlight: false },
        z: { show: false, highlight: false }
      },
      hovertemplate: "<b>Расстояние:</b> %{y:.2f} м<br><b>Значение:</b> %{customdata[1]:.4f}<extra></extra>",
      hoverlabel: { bgcolor: "rgba(10,10,20,0.9)", font: { color: "#e0e0e0", size: 11 }, bordercolor: "#333333" },
      lighting: { ambient: 0.75, diffuse: 0.4, specular: 0.02, roughness: 0.9, fresnel: 0.05 }
    });

    // Стенки и пол — цвет по ВЫСОТЕ (Z-координате)
    // Каждая высота показывает свой цвет из палитры — как топографическая карта на стенах.
    // Z-координата → fromZAxis → нормализация через цветовой пайплайн → цвет.
    var wallOpts = {
      colorscale: getActiveColorscale(), showscale: false, hoverinfo: "skip",
      cauto: false, cmin: 0, cmax: 1,
      lighting: { ambient: 0.7, diffuse: 0.4, roughness: 0.8, specular: 0.05, fresnel: 0.05 }
    };

    // Вспомогательная функция: Z-координата → нормализованный цвет [0..1]
    // Проходим тот же пайплайн, что и для основной поверхности
    function zAxisToColorNorm(zAxisVal) {
      var dataVal = fromZAxis(zAxisVal);
      if (isLogColor) dataVal = Math.log10(Math.max(dataVal, LOG_EPS));
      var norm = (dataVal - cMin) / cSpan;
      return norm < 0 ? 0 : (norm > 1 ? 1 : norm);
    }

    var N_WALL_LEVELS = 10;  // больше уровней → плавнее градиент по высоте

    // Передняя стена (i=0)
    var fwX = [], fwY = [], fwZ = [], fwSC = [];
    for (var lv = 0; lv < N_WALL_LEVELS; lv++) {
      var t = lv / (N_WALL_LEVELS - 1);
      fwX.push(X_plot[0]); fwY.push(Y[0]);
      var zr = [], sr = [];
      for (var j = 0; j < nChan; j++) {
        var z = ZdAxis[0][j] * (1 - t) + z_floor_axis * t;
        zr.push(z);
        sr.push(zAxisToColorNorm(z));
      }
      fwZ.push(zr); fwSC.push(sr);
    }
    traces.push(Object.assign({ type: "surface", x: fwX, y: fwY, z: fwZ, surfacecolor: fwSC }, wallOpts));

    // Задняя стена (i=nWin-1)
    var bwX = [], bwY = [], bwZ = [], bwSC = [];
    for (var lv = 0; lv < N_WALL_LEVELS; lv++) {
      var t = lv / (N_WALL_LEVELS - 1);
      bwX.push(X_plot[nWin-1]); bwY.push(Y[nWin-1]);
      var zr = [], sr = [];
      for (var j = 0; j < nChan; j++) {
        var z = ZdAxis[nWin-1][j] * (1 - t) + z_floor_axis * t;
        zr.push(z);
        sr.push(zAxisToColorNorm(z));
      }
      bwZ.push(zr); bwSC.push(sr);
    }
    traces.push(Object.assign({ type: "surface", x: bwX, y: bwY, z: bwZ, surfacecolor: bwSC }, wallOpts));

    // Левая стена (j=0)
    var lwX = [], lwY = [], lwZ = [], lwSC = [];
    for (var lv = 0; lv < N_WALL_LEVELS; lv++) {
      var t = lv / (N_WALL_LEVELS - 1);
      lwX.push(X_plot.map(function(r){return r[0];}));
      lwY.push(Y.map(function(r){return r[0];}));
      var zr = [], sr = [];
      for (var i = 0; i < nWin; i++) {
        var z = ZdAxis[i][0] * (1 - t) + z_floor_axis * t;
        zr.push(z);
        sr.push(zAxisToColorNorm(z));
      }
      lwZ.push(zr); lwSC.push(sr);
    }
    traces.push(Object.assign({ type: "surface", x: lwX, y: lwY, z: lwZ, surfacecolor: lwSC }, wallOpts));

    // Правая стена (j=nChan-1)
    var rwX = [], rwY = [], rwZ = [], rwSC = [];
    for (var lv = 0; lv < N_WALL_LEVELS; lv++) {
      var t = lv / (N_WALL_LEVELS - 1);
      rwX.push(X_plot.map(function(r){return r[nChan-1];}));
      rwY.push(Y.map(function(r){return r[nChan-1];}));
      var zr = [], sr = [];
      for (var i = 0; i < nWin; i++) {
        var z = ZdAxis[i][nChan-1] * (1 - t) + z_floor_axis * t;
        zr.push(z);
        sr.push(zAxisToColorNorm(z));
      }
      rwZ.push(zr); rwSC.push(sr);
    }
    traces.push(Object.assign({ type: "surface", x: rwX, y: rwY, z: rwZ, surfacecolor: rwSC }, wallOpts));

    // Пол — цвет по высоте пола (z_floor)
    var floorColorNorm = zAxisToColorNorm(z_floor_axis);
    var floorZ = [], floorSC = [];
    for (var i=0; i<nWin; i++) {
      floorZ.push(fillArr(nChan, z_floor_axis));
      floorSC.push(fillArr(nChan, floorColorNorm));
    }
    traces.push(Object.assign({
      type: "surface",
      x: X_plot, y: Y, z: floorZ,
      customdata: X_hover,
      surfacecolor: floorSC
    }, wallOpts));

    return traces;
  }

  // === Плоский 3D (2D-режим как поверхность сверху) ===
  function getFlat3DTraces(metricKey, rs) {
    var Z = rs.z[metricKey];
    var mesh = buildMeshArrays(rs);
    var X_plot = mesh.X_plot, Y = mesh.Y, X_hover = mesh.X_hover;
    var nWin = mesh.nWin, nChan = mesh.nChan;

    // Фильтрация отрицательных
    var Zf = Z;
    if (window.filterNonPositive) {
      Zf = [];
      for (var i = 0; i < nWin; i++) {
        var row = [];
        for (var j = 0; j < nChan; j++) {
          row.push(Z[i][j] < 0 ? 0 : Z[i][j]);
        }
        Zf.push(row);
      }
    }

    // Гауссово сглаживание
    if (window.zSmoothSigma > 0) {
      Zf = applySmoothing(Zf);
    }

    // Глобальная нормализация (единая шкала на весь датасет)
    var cRange = getColorRange(metricKey);
    var isLogColor = window.colorScaleType === 'log';

    var cMin, cMax;
    if (isLogColor) {
      cMin = Math.log10(Math.max(cRange.min, LOG_EPS));
      cMax = Math.log10(Math.max(cRange.max, LOG_EPS));
      if (cMax <= cMin) cMax = cMin + 1;
      var MAX_LOG_SPAN = 4;
      if ((cMax - cMin) > MAX_LOG_SPAN) {
        cMin = cMax - MAX_LOG_SPAN;
        console.warn('[h5-to-html] flat3D: cMin clamped, log span capped at', MAX_LOG_SPAN);
      }
    } else {
      cMin = cRange.min;
      cMax = cRange.max;
    }
    var cSpan = cMax - cMin;
    if (cSpan <= 0) cSpan = 1;

    // Colorbar: реальные значения
    var nColorTicks = 10;
    var colorTickVals = [], colorTickText = [];
    for (var ct = 0; ct <= nColorTicks; ct++) {
      var frac = ct / nColorTicks;
      colorTickVals.push(frac);
      var realVal;
      if (isLogColor) {
        realVal = Math.pow(10, cMin + frac * cSpan);
      } else {
        realVal = cMin + frac * cSpan;
      }
      colorTickText.push(formatRealValue(realVal));
    }

    // Плоская поверхность z=0, цвет глобально нормализованный
    var flatZ = [], sc = [], cdArr = [];
    for (var i = 0; i < nWin; i++) {
      var fzr = [], scr = [], cdr = [];
      for (var j = 0; j < nChan; j++) {
        fzr.push(0);
        var val = Zf[i][j];
        if (isLogColor) {
          val = Math.log10(Math.max(val, LOG_EPS));
        }
        var norm = (val - cMin) / cSpan;
        scr.push(norm < 0 ? 0 : (norm > 1 ? 1 : norm));
        cdr.push([X_hover[i][j], Zf[i][j]]);
      }
      flatZ.push(fzr);
      sc.push(scr);
      cdArr.push(cdr);
    }

    return [{
      type: "surface",
      x: X_plot, y: Y, z: flatZ,
      surfacecolor: sc,
      customdata: cdArr,
      colorscale: getActiveColorscale(),
      cauto: false,
      cmin: 0, cmax: 1,
      colorbar: {
        title: { text: (isLogColor ? "log10 " : "") + metricKey.toUpperCase() + (window.metricUnits ? " (" + window.metricUnits + ")" : ""), font: { color: "#fff", size: 12 } },
        thickness: 20, len: 0.6, y: 0.5,
        tickfont: { color: "#ccc", size: 10 }, titlefont: { color: "#fff" },
        tickmode: "array",
        tickvals: colorTickVals,
        ticktext: colorTickText
      },
      contours: {
        x: { show: false, highlight: false },
        y: { show: false, highlight: false },
        z: { show: false, highlight: false }
      },
      hovertemplate: "<b>Расстояние:</b> %{y:.2f} м<br><b>Значение:</b> %{customdata[1]:.4f}<extra></extra>",
      hoverlabel: { bgcolor: "rgba(10,10,20,0.9)", font: { color: "#e0e0e0", size: 11 }, bordercolor: "#333333" },
      lighting: { ambient: 1.0, diffuse: 0.0, specular: 0.0, roughness: 1.0, fresnel: 0.0 }
    }];
  }

  // === Камера (общие функции, доступны и из renderCurrent, и из init) ===
  window._savedCamera3D = null;

  function _getGd() {
    return document.querySelector(".plotly-graph-div");
  }

  function getCam(){
    var gd = _getGd();
    try{
      if(gd && gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene._scene && typeof gd._fullLayout.scene._scene.getCamera === "function"){
        var c = gd._fullLayout.scene._scene.getCamera();
        if(c && c.eye && c.center) return {eye:{x:c.eye.x,y:c.eye.y,z:c.eye.z}, center:{x:c.center.x,y:c.center.y,z:c.center.z}, up:{x:(c.up?c.up.x:0),y:(c.up?c.up.y:0),z:(c.up?c.up.z:1)}};
      }
      if(gd && gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene.camera){
        var c = gd._fullLayout.scene.camera;
        return {eye:{x:c.eye.x,y:c.eye.y,z:c.eye.z}, center:{x:c.center.x,y:c.center.y,z:c.center.z}, up:{x:(c.up?c.up.x:0),y:(c.up?c.up.y:0),z:(c.up?c.up.z:1)}};
      }
      if(gd && gd.layout && gd.layout.scene && gd.layout.scene.camera){
        var c = gd.layout.scene.camera;
        return {eye:{x:c.eye.x,y:c.eye.y,z:c.eye.z}, center:{x:c.center.x,y:c.center.y,z:c.center.z}, up:{x:(c.up?c.up.x:0),y:(c.up?c.up.y:0),z:(c.up?c.up.z:1)}};
      }
    }catch(e){}
    return {eye:{x:1.6,y:-1.4,z:1.0},center:{x:0,y:0,z:0},up:{x:0,y:0,z:1}};
  }

  function applyCamera(eye, center, up){
    var gd = _getGd();
    if(!gd) return;
    if(gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene._scene && gd._fullLayout.scene._scene.camera){
      try {
        var c = gd._fullLayout.scene._scene.camera;
        if(c.eye) c.eye = [eye.x, eye.y, eye.z];
        if(c.center) c.center = [center.x, center.y, center.z];
        if(up && c.up) c.up = [up.x, up.y, up.z];
        if(gd._fullLayout.scene._scene.glplot && typeof gd._fullLayout.scene._scene.glplot.setCamera === "function"){
          gd._fullLayout.scene._scene.glplot.setCamera(c);
        }
      } catch(e){}
    }
    var camDict = {eye:{x:eye.x,y:eye.y,z:eye.z}, center:{x:center.x,y:center.y,z:center.z}};
    if(up) camDict.up = {x:up.x,y:up.y,z:up.z};
    if(isRelayouting){ pendingCamera = camDict; return; }
    isRelayouting = true;
    try {
      Plotly.relayout(gd, {"scene.camera": camDict}).then(function(){
        isRelayouting = false;
        if(pendingCamera){ var next = pendingCamera; pendingCamera = null; applyCamera(next.eye, next.center, next.up); }
      }).catch(function(){ isRelayouting = false; });
    } catch(err) { isRelayouting = false; }
  }

  // === Web Worker для Гауссова сглаживания (off main thread) ===
  var _smoothWorker = null;
  var _smoothPending = null;

  function _initSmoothWorker() {
    if (_smoothWorker) return;
    try {
      var workerFn = function() {
        self.onmessage = function(e) {
          var d = e.data;
          if (d.cmd === 'gaussianBlur') {
            var matrix = d.matrix, sigma = d.sigma;
            if (sigma <= 0) { self.postMessage({cmd:'blurResult', result: matrix, id: d.id}); return; }
            var radius = Math.ceil(sigma * 3);
            var kernel = [], sum = 0;
            for (var i = -radius; i <= radius; i++) {
              var w = Math.exp(-(i*i)/(2*sigma*sigma));
              kernel.push(w); sum += w;
            }
            for (var k = 0; k < kernel.length; k++) kernel[k] /= sum;
            function conv1D(arr, kern) {
              var n = arr.length, r = kern.length >> 1, res = [];
              for (var i = 0; i < n; i++) {
                var v = 0;
                for (var k = 0; k < kern.length; k++) {
                  var idx = i + k - r;
                  if (idx < 0) idx = 0; if (idx >= n) idx = n - 1;
                  v += arr[idx] * kern[k];
                }
                res.push(v);
              }
              return res;
            }
            var rows = matrix.length, cols = matrix[0].length;
            // Проход по строкам
            var temp = [];
            for (var i = 0; i < rows; i++) temp.push(conv1D(matrix[i], kernel));
            // Транспозиция → свёртка по столбцам → обратная транспозиция
            var transposed = [];
            for (var j = 0; j < cols; j++) {
              var col = [];
              for (var i = 0; i < rows; i++) col.push(temp[i][j]);
              transposed.push(col);
            }
            var blurredT = [];
            for (var j = 0; j < cols; j++) blurredT.push(conv1D(transposed[j], kernel));
            var result = [];
            for (var i = 0; i < rows; i++) {
              var row = [];
              for (var j = 0; j < cols; j++) row.push(blurredT[j][i]);
              result.push(row);
            }
            self.postMessage({cmd:'blurResult', result: result, id: d.id});
          }
        };
      };
      var blob = new Blob(['(' + workerFn.toString() + ')()'], {type: 'application/javascript'});
      var url = URL.createObjectURL(blob);
      _smoothWorker = new Worker(url);
      URL.revokeObjectURL(url);
      _smoothWorker.onmessage = function(e) {
        if (e.data.cmd === 'blurResult' && _smoothPending && _smoothPending.id === e.data.id) {
          _smoothPending.callback(e.data.result);
          _smoothPending = null;
        }
      };
    } catch(err) {
      console.warn('[h5-to-html] Web Worker не доступен, сглаживание в главном потоке:', err);
    }
  }

  // Асинхронное сглаживание через Worker (fallback — синхронное в главном потоке)
  function smoothAsync(Z, sigma, callback) {
    if (sigma <= 0) { callback(Z); return; }
    if (_smoothWorker) {
      var id = Date.now() + Math.random();
      _smoothPending = {id: id, callback: callback};
      _smoothWorker.postMessage({cmd: 'gaussianBlur', matrix: Z, sigma: sigma, id: id});
    } else {
      // Fallback: синхронно в главном потоке
      callback(gaussianBlur2D(Z, sigma));
    }
  }

  // === Производительность: таймер рендера ===
  var _renderStart = 0;
  var _lastRenderMs = 0;
  var _dataPoints = 0;

  // === Рендер текущего состояния (Plotly.react для повторных вызовов) ===
  var _firstRender = true;

  function renderCurrent() {
    var gd = _getGd();
    if (!gd) return;

    _renderStart = performance.now();

    console.log('[h5-to-html] renderCurrent: colorScale=' + window.colorScaleType + ' zAxisPower=' + window.zAxisPower + ' gamma=' + window.colorGamma + ' range=' + window.colorMode + ' filter=' + window.filterNonPositive);

    // Сохраняем текущую камеру ПЕРЕД перерисовкой
    var currentCam = getCam();
    if (currentCam && window.currentViewMode !== "flat") {
      window._savedCamera3D = currentCam;
    }

    // НЕ обнуляем currentResampled — кэш сохраняется до реального изменения данных
    var rs = ensureResampled(window.currentRes);
    var isFlat = window.currentViewMode === "flat";
    var titleText = getTitleText(
      window.currentMetricKey,
      isFlat ? "2D (сверху)" : "3D"
    );

    var data, layout;
    if (isFlat) {
      data = getFlat3DTraces(window.currentMetricKey, rs);
      layout = get3DLayout(titleText, rs, true, null);
    } else {
      data = get3DTraces(window.currentMetricKey, rs);
      layout = get3DLayout(titleText, rs, false, window._savedCamera3D);
    }

    _dataPoints = (rs.nWin || 0) * (rs.nChan || 0);

    // Plotly.react() вместо newPlot() — переиспользуем WebGL-контекст
    if (_firstRender || !gd.data || gd.data.length === 0) {
      _firstRender = false;
      Plotly.newPlot(gd, data, layout, {scrollZoom: false, responsive: true}).then(function() {
        _onRenderDone();
      });
    } else {
      Plotly.react(gd, data, layout, {scrollZoom: false, responsive: true}).then(function() {
        _onRenderDone();
      });
    }
  }

  function _onRenderDone() {
    _lastRenderMs = (performance.now() - _renderStart).toFixed(0);
    var perfEl = document.getElementById('perf-render');
    var dataEl = document.getElementById('perf-data');
    if (perfEl) perfEl.textContent = _lastRenderMs;
    if (dataEl) dataEl.textContent = (_dataPoints || 0).toLocaleString();
  }

  // === Инициализация ===
  function init(){
    var gd = document.querySelector(".plotly-graph-div");
    if(!gd || typeof Plotly === "undefined" || !gd._fullLayout) {
      setTimeout(init, 100);
      return;
    }

    console.log('[h5-to-html] init: colorScaleType=' + window.colorScaleType + ', zAxisPower=' + window.zAxisPower + ', gamma=' + window.colorGamma + ', filterNonPositive=' + window.filterNonPositive);

    // Проверка WebGL
    var canvas = document.createElement('canvas');
    var gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (!gl) {
      console.warn("[h5-to-html] WebGL недоступен — рендеринг может быть ограничен");
    } else {
      // Сообщаем браузеру о предпочтении GPU-рендеринга
      var dbgInfo = gl.getExtension('WEBGL_debug_renderer_info');
      if (dbgInfo) {
        var renderer = gl.getParameter(dbgInfo.UNMASKED_RENDERER_WEBGL);
        console.log('[h5-to-html] GPU:', renderer);
      }
    }

    // Инициализируем Web Worker для сглаживания
    _initSmoothWorker();

    // Внедряем панель HUD
    var hud = document.createElement("div");
    hud.id = "editor-hud";
    hud.innerHTML = `
      <div class="hud-header" onclick="document.getElementById('editor-hud').classList.toggle('minimized')">
        <span>DAS 3D Управление и Экспорт</span>
        <button class="toggle-btn" onclick="event.stopPropagation(); document.getElementById('editor-hud').classList.toggle('minimized')">_ / []</button>
      </div>
      <div class="hud-content">
        <div class="hud-section-title" style="color:#60a5fa;">Метрика активности (.h5)</div>
        <div class="hud-btn-group" id="metric-buttons"></div>
        <div class="hud-btn-group" id="derived-metric-buttons" style="margin-top:4px;"></div>

        <div class="hud-section-title">Прореживание (разрешение)</div>
        <div class="hud-slider-row">
          <span style="font-size:11px;color:#9aa0af;min-width:16px;">Lo</span>
          <input type="range" id="resolution-slider" min="30" max="` + window.masterMaxDim + `" value="` + window.currentRes + `" oninput="window.changeResolution(+this.value)">
          <span style="font-size:11px;color:#9aa0af;min-width:20px;">Hi</span>
        </div>
        <div class="hud-slider-label" id="resolution-label">` + window.currentRes + `</div>

        <div class="hud-section-title">Режим просмотра и ракурсы</div>
        <div class="hud-btn-group">
          <button class="hud-btn toggle-mode-btn" style="width:100%;" onclick="window.toggle3D2D(this)">Переключить в 2D (вид сверху)</button>
          <button class="hud-btn" onclick="window.hudSetView('reset')">Сброс 3D</button>
          <button class="hud-btn" onclick="window.hudSetView('top2d')">Вид сверху</button>
          <button class="hud-btn" onclick="window.hudSetView('iso')">Изометрия</button>
          <button class="hud-btn" onclick="window.hudSetView('side')">Сбоку</button>
        </div>

        <div class="hud-section-title">Скорость полёта (клавиши + / -)</div>
        <div class="hud-btn-group" id="speed-buttons">
          <button class="hud-btn" onclick="window.hudSetSpeed(0.5, this)">0.5x</button>
          <button class="hud-btn active" onclick="window.hudSetSpeed(1.0, this)">1x</button>
          <button class="hud-btn" onclick="window.hudSetSpeed(2.0, this)">2x</button>
          <button class="hud-btn" onclick="window.hudSetSpeed(4.0, this)">4x</button>
        </div>

        <div class="hud-section-title">Цветовая палитра</div>
        <div class="hud-btn-group" id="colorscale-buttons" style="max-height:140px;overflow-y:auto;flex-wrap:wrap;"></div>

        <div class="hud-section-title">Шкала цвета</div>
        <div class="hud-btn-group">
          <button class="hud-btn active" id="btn-scale-linear" onclick="window.setColorScaleType('linear', this)">Линейная</button>
          <button class="hud-btn" id="btn-scale-log" onclick="window.setColorScaleType('log', this)">Log10</button>
        </div>

        <div class="hud-section-title">Диапазон цвета</div>
        <div class="hud-btn-group">
          <button class="hud-btn active" id="btn-color-percentile" onclick="window.setColorMode('percentile', this)">P2–P98</button>
          <button class="hud-btn" id="btn-color-full" onclick="window.setColorMode('full', this)">Min–Max</button>
        </div>

        <div class="hud-section-title">Сжатие Z-оси (&lt;1 — сжать пики, 1 — линейная)</div>
        <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
          <span style="font-size:11px;color:#9aa0af;">p =</span>
          <input type="text" id="zpower-input" value="1.0" style="
            width:70px; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.2);
            border-radius:5px; padding:4px 8px; font-size:13px; color:#60a5fa;
            font-variant-numeric:tabular-nums; text-align:center; outline:none;
          " onkeydown="if(event.key==='Enter'){window.setZAxisPower(+this.value);this.blur();}" onblur="window.setZAxisPower(+this.value)">
        </div>
        <div style="display:flex;gap:4px;margin-top:4px;flex-wrap:wrap;">
          <button class="hud-btn" style="font-size:10px;padding:3px 6px;" onclick="window.setZAxisPower(1.0);document.getElementById('zpower-input').value='1.0'">1.0 (лин.)</button>
          <button class="hud-btn" style="font-size:10px;padding:3px 6px;" onclick="window.setZAxisPower(0.5);document.getElementById('zpower-input').value='0.5'">0.5 (√)</button>
          <button class="hud-btn" style="font-size:10px;padding:3px 6px;" onclick="window.setZAxisPower(0.33);document.getElementById('zpower-input').value='0.33'">0.33 (∛)</button>
          <button class="hud-btn" style="font-size:10px;padding:3px 6px;" onclick="window.setZAxisPower(0.25);document.getElementById('zpower-input').value='0.25'">0.25 (⁴√)</button>
          <button class="hud-btn" style="font-size:10px;padding:3px 6px;" onclick="window.setZAxisPower(0.15);document.getElementById('zpower-input').value='0.15'">0.15</button>
        </div>

        <div class="hud-section-title">Сглаживание поверхности (Гаусс)</div>
        <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
          <span style="font-size:11px;color:#9aa0af;">sigma =</span>
          <input type="text" id="zsmooth-input" value="0" style="
            width:70px; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.2);
            border-radius:5px; padding:4px 8px; font-size:13px; color:#60a5fa;
            font-variant-numeric:tabular-nums; text-align:center; outline:none;
          " onkeydown="if(event.key==='Enter'){window.setZSmoothSigma(+this.value);this.blur();}" onblur="window.setZSmoothSigma(+this.value)">
        </div>
        <div style="display:flex;gap:4px;margin-top:4px;flex-wrap:wrap;">
          <button class="hud-btn" style="font-size:10px;padding:3px 6px;" onclick="window.setZSmoothSigma(0);document.getElementById('zsmooth-input').value='0'">0 (нет)</button>
          <button class="hud-btn" style="font-size:10px;padding:3px 6px;" onclick="window.setZSmoothSigma(0.5);document.getElementById('zsmooth-input').value='0.5'">0.5</button>
          <button class="hud-btn" style="font-size:10px;padding:3px 6px;" onclick="window.setZSmoothSigma(1);document.getElementById('zsmooth-input').value='1'">1</button>
          <button class="hud-btn" style="font-size:10px;padding:3px 6px;" onclick="window.setZSmoothSigma(2);document.getElementById('zsmooth-input').value='2'">2</button>
          <button class="hud-btn" style="font-size:10px;padding:3px 6px;" onclick="window.setZSmoothSigma(3);document.getElementById('zsmooth-input').value='3'">3</button>
          <button class="hud-btn" style="font-size:10px;padding:3px 6px;" onclick="window.setZSmoothSigma(5);document.getElementById('zsmooth-input').value='5'">5</button>
        </div>

        <div class="hud-section-title">Гамма-контраст (&lt;1 — деталь высоких, &gt;1 — низких)</div>
        <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
          <span style="font-size:11px;color:#9aa0af;">gamma =</span>
          <input type="text" id="gamma-input" value="1.0" style="
            width:70px; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.2);
            border-radius:5px; padding:4px 8px; font-size:13px; color:#60a5fa;
            font-variant-numeric:tabular-nums; text-align:center; outline:none;
          " onkeydown="if(event.key==='Enter'){window.setColorGamma(+this.value);this.blur();}" onblur="window.setColorGamma(+this.value)">
        </div>

        <div class="hud-section-title">Фильтр значений</div>
        <div class="hud-btn-group">
          <label style="display:flex;align-items:center;gap:6px;cursor:pointer;font-size:12px;color:#e0e0e0;width:100%;padding:4px 0;">
            <input type="checkbox" id="filter-nonpositive" style="accent-color:#3b82f6;width:16px;height:16px;cursor:pointer;" checked onchange="window.toggleFilterNonPositive(this.checked)">
            Отрицательные → 0 (&lt; 0)
          </label>
        </div>

        <div class="hud-section-title" style="color:#34d399;">Экспорт 2D-графика (Время x Каналы)</div>
        <div class="hud-btn-group">
          <button class="hud-btn download-btn" style="width:100%;" onclick="window.exportFlat2DPNG(this)">Скачать 2D (PNG 1080p)</button>
        </div>

        <button class="hud-btn" style="width:100%; margin-top:10px; background:rgba(255,255,255,0.05);" onclick="document.getElementById('hud-help').classList.toggle('show')">Справка по управлению</button>
        <div id="hud-help" class="hud-help-box">
          <b>W / S / стрелки вверх/вниз</b> — Полёт вперёд / назад<br>
          <b>A / D / стрелки влево/вправо</b> — Стрейф<br>
          <b>E / Space</b> — Вверх по оси Z<br>
          <b>Q / C</b> — Вниз по оси Z<br>
          <b>Shift + WASD/QE</b> — Турбо x3<br>
          <b>+ / -</b> — Скорость полёта<br>
          <b>ЛКМ (Drag)</b> — Вращение<br>
          <b>ПКМ / Колесо</b> — Зум<br>
          <b>Shift + ЛКМ / СКМ</b> — Панорама
        </div>
        <div class="hud-perf-bar" id="hud-perf">Рендер: <span id="perf-render">—</span> мс | Данных: <span id="perf-data">—</span> точек</div>
      </div>
    `;
    document.body.appendChild(hud);

    // ═══ Гарантия видимости HUD ═══
    // Принудительно применяем стили после добавления в DOM
    hud.style.cssText = 'position:fixed!important;top:15px!important;left:15px!important;z-index:99999!important;';

    // ═══ Диагностика HUD ═══
    var hudSections = hud.querySelectorAll('.hud-section-title');
    var sectionNames = [];
    hudSections.forEach(function(s){ sectionNames.push(s.textContent.trim()); });
    console.log('[h5-to-html] ✅ HUD создан. Версия: ' + window.scriptVersion);
    console.log('[h5-to-html] HUD секции (' + sectionNames.length + '):', sectionNames);
    var btnLog = document.getElementById('btn-scale-log');
    var btnLinear = document.getElementById('btn-scale-linear');
    var csButtons = document.getElementById('colorscale-buttons');
    var gammaInput = document.getElementById('gamma-input');
    console.log('[h5-to-html] btn-scale-log:', btnLog ? 'OK' : '❌ ОТСУТСТВУЕТ');
    console.log('[h5-to-html] btn-scale-linear:', btnLinear ? 'OK' : '❌ ОТСУТСТВУЕТ');
    console.log('[h5-to-html] colorscale-buttons:', csButtons ? 'OK' : '❌ ОТСУТСТВУЕТ');
    console.log('[h5-to-html] gamma-input:', gammaInput ? 'OK' : '❌ ОТСУТСТВУЕТ');

    // Заполняем кнопки метрик (исходные и производные отдельно)
    var derivedMetrics = ['rms', 'dynamic_range', 'activity_index', 'snr', 'cv', 'spread'];
    var metricContainer = document.getElementById("metric-buttons");
    var derivedContainer = document.getElementById("derived-metric-buttons");
    if(metricContainer && window.masterData){
      Object.keys(window.masterData).forEach(function(mKey){
        var isDerived = derivedMetrics.indexOf(mKey) >= 0;
        var container = isDerived ? derivedContainer : metricContainer;
        if (!container) container = metricContainer;
        var btn = document.createElement("button");
        btn.className = "hud-btn" + (mKey === window.currentMetricKey ? " active" : "");
        btn.style.cssText = "font-weight:600;font-size:10px;padding:4px 6px;";
        btn.innerHTML = mKey.toUpperCase();
        btn.title = isDerived ? 'Производная метрика' : mKey;
        btn.onclick = function(){ window.switchMetric(mKey, btn); };
        container.appendChild(btn);
      });
    }

    // Заполняем кнопки цветовых палитр
    var csContainer = document.getElementById("colorscale-buttons");
    if(csContainer && typeof COLORSCALES !== "undefined"){
      Object.keys(COLORSCALES).forEach(function(name){
        var btn = document.createElement("button");
        btn.className = "hud-btn" + (name === window.currentColorscale ? " active" : "");
        btn.style.cssText = "min-width:55px;font-size:10px;padding:4px 5px;";
        btn.innerHTML = name;
        btn.onclick = function(){ window.setColorscale(name, btn); };
        csContainer.appendChild(btn);
      });
    }

    // === Переключение цветовой палитры ===
    window.setColorscale = function(name, btnEl) {
      if(!COLORSCALES[name]) return;
      window.currentColorscale = name;
      var btns = document.querySelectorAll("#colorscale-buttons .hud-btn");
      btns.forEach(function(b){ b.classList.remove("active"); });
      if(btnEl) btnEl.classList.add("active");
      renderCurrent();
    };

    // === Гамма-контраст ===
    window.setColorGamma = function(val) {
      val = parseFloat(val);
      if (isNaN(val) || val <= 0) { console.warn('[h5-to-html] setColorGamma: invalid value, fallback to 1.0, input=', val); val = 1.0; }
      window.colorGamma = val;
      console.log('[h5-to-html] gamma =', val);
      var inp = document.getElementById("gamma-input");
      if(inp) inp.value = val;
      renderCurrent();
    };

    // === Переключение цветового диапазона ===
    window.setColorMode = function(mode, btnEl) {
      window.colorMode = mode;
      var btnP = document.getElementById("btn-color-percentile");
      var btnF = document.getElementById("btn-color-full");
      if(btnP) btnP.classList.toggle("active", mode === "percentile");
      if(btnF) btnF.classList.toggle("active", mode === "full");
      renderCurrent();
    };

    // === Переключение шкалы цвета (линейная / log10) ===
    window.setColorScaleType = function(scaleType, btnEl) {
      window.colorScaleType = scaleType;
      console.log('[h5-to-html] colorScaleType =', scaleType);
      var btnL = document.getElementById("btn-scale-linear");
      var btnG = document.getElementById("btn-scale-log");
      if(btnL) btnL.classList.toggle("active", scaleType === "linear");
      if(btnG) btnG.classList.toggle("active", scaleType === "log");
      renderCurrent();
    };

    // === Переключение степени сжатия Z-оси ===
    window.setZAxisPower = function(p) {
      // Ограничиваем p в разумных пределах
      if (isNaN(p) || p <= 0.01) p = 0.01;
      if (p > 1.0) p = 1.0;
      window.zAxisPower = p;
      var inp = document.getElementById('zpower-input');
      if (inp) inp.value = p;
      console.log('[h5-to-html] zAxisPower =', p);
      renderCurrent();
    };

    // === Переключение сглаживания поверхности ===
    window.setZSmoothSigma = function(sigma) {
      if (isNaN(sigma) || sigma < 0) sigma = 0;
      if (sigma > 10) sigma = 10;
      window.zSmoothSigma = sigma;
      var inp = document.getElementById('zsmooth-input');
      if (inp) inp.value = sigma;
      console.log('[h5-to-html] zSmoothSigma =', sigma);
      renderCurrent();
    };

    // === Фильтр неположительных значений ===
    window.toggleFilterNonPositive = function(checked) {
      window.filterNonPositive = checked;
      window.currentResampled = null;
      renderCurrent();
    };

    // === Смена метрики ===
    window.switchMetric = function(mKey, btnEl){
      if(!window.masterData[mKey]) return;
      window.currentMetricKey = mKey;
      window.currentResampled = null;
      // Снимаем active со ВСЕХ кнопок метрик (исходных и производных)
      var btns1 = document.querySelectorAll("#metric-buttons .hud-btn");
      var btns2 = document.querySelectorAll("#derived-metric-buttons .hud-btn");
      btns1.forEach(function(b){ b.classList.remove("active"); });
      btns2.forEach(function(b){ b.classList.remove("active"); });
      if(btnEl) btnEl.classList.add("active");
      renderCurrent();
    };

    // === Переключение 2D/3D ===
    window.toggle3D2D = function(btn){
      var slider = document.getElementById("resolution-slider");
      var label = document.getElementById("resolution-label");

      if(window.currentViewMode === "3d"){
        window.currentViewMode = "flat";
        window.currentRes = window.defaultRes2D;
        window.currentResampled = null;
        if(slider){ slider.value = window.currentRes; }
        if(label){ label.textContent = window.currentRes; }
        renderCurrent();
        if(btn) btn.innerHTML = "Вернуться в 3D";
      } else {
        window.currentViewMode = "3d";
        window.currentRes = window.defaultRes3D;
        window.currentResampled = null;
        if(slider){ slider.value = window.currentRes; }
        if(label){ label.textContent = window.currentRes; }
        renderCurrent();
        if(btn) btn.innerHTML = "Переключить в 2D (вид сверху)";
      }
    };

    // === Изменение разрешения (слайдер с debounce) ===
    var _resDebounceTimer = null;
    window.changeResolution = function(val){
      window.currentRes = val;
      window.currentResampled = null;
      var label = document.getElementById("resolution-label");
      // Быстрый отклик: показываем размер сразу
      var rs = ensureResampled(val);
      if(label){
        label.textContent = rs.nWin + " x " + rs.nChan + " (шаг " + rs.stepW + " x " + rs.stepC + ")";
      }
      // Debounce: рендер не чаще чем раз в 80 мс
      if (_resDebounceTimer) clearTimeout(_resDebounceTimer);
      _resDebounceTimer = setTimeout(function(){
        _resDebounceTimer = null;
        renderCurrent();
      }, 80);
    };

    // === Ракурсы ===
    window.hudSetView = function(viewType){
      if(viewType === 'reset') applyCamera({x:1.6,y:-1.4,z:1.0},{x:0,y:0,z:0},{x:0,y:0,z:1});
      else if(viewType === 'top2d') applyCamera({x:0,y:0.001,z:2.2},{x:0,y:0,z:0},{x:0,y:1,z:0});
      else if(viewType === 'iso') applyCamera({x:1.6,y:-1.4,z:1.2},{x:0,y:0,z:0},{x:0,y:0,z:1});
      else if(viewType === 'side') applyCamera({x:2.2,y:0,z:0.2},{x:0,y:0,z:0},{x:0,y:0,z:1});
    };

    window.hudSetSpeed = function(mult, btnEl){
      speedMultiplier = mult;
      var btns = document.querySelectorAll("#speed-buttons .hud-btn");
      btns.forEach(function(b){ b.classList.remove("active"); });
      if(btnEl) btnEl.classList.add("active");
    };

    // === Экспорт PNG 1080p ===
    window.exportFlat2DPNG = function(btn){
      var origText = btn.innerHTML;
      btn.innerHTML = "PNG...";
      btn.disabled = true;

      var tempDiv = document.createElement("div");
      tempDiv.style.cssText = "position:fixed;left:-9999px;top:-9999px;width:1920px;height:1080px;background:#0a0a14;";
      document.body.appendChild(tempDiv);

      var rs = ensureResampled(window.currentRes);
      var titleText = getTitleText(window.currentMetricKey, "2D").replace(/<[^>]*>/g, " ");
      var ticks = computeTicks(rs.timeAbs, 8);
      var Zraw = rs.z[window.currentMetricKey];

      // Транспонируем и фильтруем для экспорта
      var Zt = transpose2D(Zraw);
      if (window.filterNonPositive) {
        for (var j = 0; j < Zt.length; j++) {
          for (var i = 0; i < Zt[j].length; i++) {
            if (Zt[j][i] < 0) Zt[j][i] = 0;
          }
        }
      }
      // Гауссово сглаживание для экспорта
      if (window.zSmoothSigma > 0) {
        Zt = gaussianBlur2D(Zt, window.zSmoothSigma);
      }
      // Глобальная нормализация (единая шкала на весь датасет)
      var cRange = getColorRange(window.currentMetricKey);
      var isLogColor = window.colorScaleType === 'log';
      var cMin, cMax;
      if (isLogColor) {
        cMin = Math.log10(Math.max(cRange.min, LOG_EPS));
        cMax = Math.log10(Math.max(cRange.max, LOG_EPS));
        if (cMax <= cMin) cMax = cMin + 1;
        var MAX_LOG_SPAN = 4;
        if ((cMax - cMin) > MAX_LOG_SPAN) {
          cMin = cMax - MAX_LOG_SPAN;
          console.warn('[h5-to-html] heatmap export: cMin clamped, log span capped at', MAX_LOG_SPAN);
        }
      } else {
        cMin = cRange.min;
        cMax = cRange.max;
      }
      var cSpan = cMax - cMin;
      if (cSpan <= 0) cSpan = 1;
      // Нормализуем
      var ZtCS = [];
      for (var j = 0; j < Zt.length; j++) {
        var row = [];
        for (var i = 0; i < Zt[j].length; i++) {
          var val = Zt[j][i];
          if (isLogColor) {
            val = Math.log10(Math.max(val, LOG_EPS));
          }
          var norm = (val - cMin) / cSpan;
          row.push(norm < 0 ? 0 : (norm > 1 ? 1 : norm));
        }
        ZtCS.push(row);
      }

      // Colorbar: реальные значения
      var nColorTicks = 10;
      var colorTickVals = [], colorTickText = [];
      for (var ct = 0; ct <= nColorTicks; ct++) {
        var frac = ct / nColorTicks;
        colorTickVals.push(frac);
        var realVal;
        if (isLogColor) {
          realVal = Math.pow(10, cMin + frac * cSpan);
        } else {
          realVal = cMin + frac * cSpan;
        }
        colorTickText.push(formatRealValue(realVal));
      }

      var data = [{
        type: "heatmap",
        x: rs.timeAbs,
        y: rs.channels,
        z: ZtCS,
        colorscale: getActiveColorscale(),
        zauto: false,
        zmin: 0,
        zmax: 1,
        colorbar: {
          title: { text: "Уровень", font: { color: "#fff", size: 16 } },
          thickness: 25, len: 0.85,
          tickfont: {color: "#ddd", size: 14}, titlefont: {color: "#fff", size: 16},
          tickmode: "array",
          tickvals: colorTickVals,
          ticktext: colorTickText
        }
      }];
      var layout = {
        title: { text: titleText, font: { color: "#ffffff", size: 22 }, x: 0.5 },
        xaxis: {
          title: "Абсолютное время", tickvals: ticks.vals, ticktext: ticks.text,
          gridcolor: "#333", linecolor: "#666",
          tickfont: { color: "#ddd", size: 14 }, titlefont: {color: "#fff", size: 16}
        },
        yaxis: {
          title: "Каналы / расстояние (м)",
          gridcolor: "#333", linecolor: "#666",
          tickfont: { color: "#ddd", size: 14 }, titlefont: {color: "#fff", size: 16}
        },
        paper_bgcolor: "#0a0a14", plot_bgcolor: "#0a0a14",
        margin: { l: 90, r: 60, b: 80, t: 70 }
      };

      Plotly.newPlot(tempDiv, data, layout).then(function(){
        return Plotly.toImage(tempDiv, {format: "png", width: 1920, height: 1080});
      }).then(function(imgUrl){
        var a = document.createElement("a");
        a.href = imgUrl;
        a.download = "activity_flat_2d_heatmap_1080p.png";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        Plotly.purge(tempDiv);  // очистить WebGL-контекст перед удалением
        document.body.removeChild(tempDiv);
        btn.innerHTML = origText;
        btn.disabled = false;
      }).catch(function(){
        try { Plotly.purge(tempDiv); } catch(e) {}  // очистить даже при ошибке
        try { document.body.removeChild(tempDiv); }catch(e){}
        btn.innerHTML = origText;
        btn.disabled = false;
      });
    };

    // === WASD-навигация ===
    // (getCam и applyCamera уже определены во внешней области)
    var _animFrameId = null;
    function updateLoop(){
      var cam = getCam();
      var eye = cam.eye || {x:1.6,y:-1.4,z:1.0};
      var center = cam.center || {x:0,y:0,z:0};
      var up = cam.up || {x:0,y:0,z:1};
      var fx = center.x - eye.x, fy = center.y - eye.y, fz = center.z - eye.z;
      var flen = Math.sqrt(fx*fx + fy*fy + fz*fz) || 1;
      fx /= flen; fy /= flen; fz /= flen;
      var ux = up.x||0, uy = up.y||0, uz = up.z||1;
      var rx = fy*uz - fz*uy, ry = fz*ux - fx*uz, rz = fx*uy - fy*ux;
      var rlen = Math.sqrt(rx*rx + ry*ry + rz*rz) || 1;
      rx /= rlen; ry /= rlen; rz /= rlen;
      var curSpeed = baseSpeed * speedMultiplier * (keysPressed["shift"] ? 3.0 : 1.0);
      var moved = false;
      if(keysPressed["w"] || keysPressed["arrowup"]){
        center.x += fx*curSpeed; center.y += fy*curSpeed; center.z += fz*curSpeed;
        eye.x += fx*curSpeed; eye.y += fy*curSpeed; eye.z += fz*curSpeed; moved = true;
      }
      if(keysPressed["s"] || keysPressed["arrowdown"]){
        center.x -= fx*curSpeed; center.y -= fy*curSpeed; center.z -= fz*curSpeed;
        eye.x -= fx*curSpeed; eye.y -= fy*curSpeed; eye.z -= fz*curSpeed; moved = true;
      }
      if(keysPressed["d"] || keysPressed["arrowright"]){
        center.x += rx*curSpeed; center.y += ry*curSpeed; center.z += rz*curSpeed;
        eye.x += rx*curSpeed; eye.y += ry*curSpeed; eye.z += rz*curSpeed; moved = true;
      }
      if(keysPressed["a"] || keysPressed["arrowleft"]){
        center.x -= rx*curSpeed; center.y -= ry*curSpeed; center.z -= rz*curSpeed;
        eye.x -= rx*curSpeed; eye.y -= ry*curSpeed; eye.z -= rz*curSpeed; moved = true;
      }
      if(keysPressed["e"] || keysPressed[" "]){
        center.z += curSpeed; eye.z += curSpeed; moved = true;
      }
      if(keysPressed["q"] || keysPressed["c"]){
        center.z -= curSpeed; eye.z -= curSpeed; moved = true;
      }
      if(moved) {
        applyCamera(eye, center, up);
        _animFrameId = requestAnimationFrame(updateLoop);
      } else {
        _animFrameId = null;  // останавливаем loop когда нет движения
      }
    }
    function _startUpdateLoop() {
      if (!_animFrameId) _animFrameId = requestAnimationFrame(updateLoop);
    }

    document.addEventListener("keydown", function(e){
      if(["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
      var key = e.key.toLowerCase();
      keysPressed[key] = true;
      _startUpdateLoop();  // запуск loop при нажатии клавиши
      if(e.key === "+" || e.key === "="){
        speedMultiplier = Math.min(8.0, speedMultiplier * 1.5);
        var btns = document.querySelectorAll("#speed-buttons .hud-btn");
        btns.forEach(function(b){ b.classList.remove("active"); });
      } else if(e.key === "-" || e.key === "_"){
        speedMultiplier = Math.max(0.2, speedMultiplier / 1.5);
        var btns = document.querySelectorAll("#speed-buttons .hud-btn");
        btns.forEach(function(b){ b.classList.remove("active"); });
      }
      if(key === "arrowup" || key === "arrowdown" || key === "arrowleft" || key === "arrowright" || key === " " || ["w","a","s","d","q","e","c"].includes(key)){
        e.preventDefault();
      }
    }, false);

    document.addEventListener("keyup", function(e){
      keysPressed[e.key.toLowerCase()] = false;
    }, false);

    window.addEventListener("blur", function(){ keysPressed = {}; });

    // Зум колёсиком: capture-фаза перехватывает событие до Plotly,
    // stopImmediatePropagation не даёт встроенному обработчику Plotly сработать.
    gd.addEventListener("wheel", function(e){
      e.preventDefault();
      e.stopImmediatePropagation();
      try{
        var cam = getCam();
        var eye = cam.eye || {x:1.6,y:-1.4,z:1.0};
        var center = cam.center || {x:0,y:0,z:0};
        var dx = eye.x - center.x, dy = eye.y - center.y, dz = eye.z - center.z;
        var dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
        // Ограничение: не ближе 0.15 и не дальше 20
        if (e.deltaY < 0 && dist < 0.15) return;
        if (e.deltaY > 0 && dist > 20) return;
        // Замедление зума при приближении к поверхности
        // Чем ближе камера, тем медленнее зум (плавное приближение)
        var baseDelta = (e.deltaY < 0) ? -0.10 : 0.10;
        var speedScale = Math.min(1.0, dist / 2.0);  // На расстоянии < 2 зум замедляется
        var factor = 1.0 + baseDelta * Math.max(0.12, speedScale);
        eye.x = center.x + dx*factor;
        eye.y = center.y + dy*factor;
        eye.z = center.z + dz*factor;
        applyCamera(eye, center, cam.up);
      }catch(err){}
    }, {passive:false, capture:true});

    // Начальный рендер через JS (единообразный цветовой диапазон с первого кадра)
    renderCurrent();

    // Убедиться, что HUD поверх Plotly — перемещаем в body последним элементом
    var hudEl = document.getElementById('editor-hud');
    if (hudEl && hudEl.parentElement) {
      hudEl.parentElement.removeChild(hudEl);
      document.body.appendChild(hudEl);
    }

    // WASD loop запускается только при нажатии клавиш
  }

  if(document.readyState === "loading"){
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
</script>"""

    # Собираем полный скрипт: данные → утилиты → приложение
    wasd_script = js_utils + js_vars + js_app

    hud_css = """
<style id="h5viz-hud-css">
html{height:100vh;width:100vw;margin:0;padding:0;background:#0a0a14;overflow:hidden;}
body{margin:0;padding:0;background:#0a0a14;overflow:hidden;height:100vh;width:100vw;position:relative;}
.plotly-graph-div{position:fixed!important;top:0!important;left:0!important;
width:100vw!important;height:100vh!important;background:#0a0a14!important;z-index:1!important;}
.js-plotly-plot,.plot-container,.svg-container{z-index:1!important;}
#editor-hud {
  position: fixed !important; top: 15px !important; left: 15px !important; z-index: 99999 !important;
  isolation: isolate;
  width: 360px; background: rgba(18, 18, 30, 0.95);
  backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.18); border-radius: 10px;
  padding: 14px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: #e0e0e0; box-shadow: 0 10px 40px rgba(0,0,0,0.7); user-select: none;
  transition: width 0.2s ease, padding 0.2s ease; max-height: 95vh; overflow-y: auto;
  will-change: transform;
}
#editor-hud::-webkit-scrollbar { width: 4px; }
#editor-hud::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 2px; }
#editor-hud.minimized .hud-content { display: none; }
#editor-hud.minimized { width: auto; padding: 10px 14px; }
.hud-header {
  display: flex; justify-content: space-between; align-items: center;
  font-weight: 600; font-size: 14px; color: #fff; margin-bottom: 10px; cursor: pointer;
}
.hud-header .toggle-btn {
  background: rgba(255,255,255,0.1); border: none; color: #fff;
  border-radius: 4px; padding: 2px 6px; cursor: pointer; font-size: 12px;
}
.hud-section-title {
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;
  color: #9aa0af; margin: 10px 0 6px 0; font-weight: 600;
}
.hud-btn-group { display: flex; flex-wrap: wrap; gap: 6px; }
.hud-btn {
  flex: 1 1 auto; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.14);
  color: #eee; border-radius: 6px; padding: 6px 8px; font-size: 12px; font-weight: 500;
  cursor: pointer; text-align: center; transition: all 0.15s ease;
}
.hud-btn:hover { background: rgba(255,255,255,0.18); border-color: rgba(255,255,255,0.3); color: #fff; }
.hud-btn.active { background: #3b82f6; border-color: #60a5fa; color: #fff; }
.hud-btn.toggle-mode-btn { background: rgba(59, 130, 246, 0.22); border-color: rgba(59, 130, 246, 0.55); color: #60a5fa; font-weight: 600; }
.hud-btn.toggle-mode-btn:hover { background: rgba(59, 130, 246, 0.4); border-color: #60a5fa; color: #fff; }
.hud-btn.download-btn { background: rgba(16, 185, 129, 0.18); border-color: rgba(16, 185, 129, 0.45); color: #34d399; font-weight: 600; }
.hud-btn.download-btn:hover { background: rgba(16, 185, 129, 0.35); border-color: #34d399; color: #fff; }

.hud-help-box {
  background: rgba(0,0,0,0.35); border-radius: 6px; padding: 8px 10px;
  font-size: 11px; line-height: 1.6; color: #ccc; margin-top: 8px; display: none;
}
.hud-help-box.show { display: block; }
.hud-help-box b { color: #fff; }
.hud-slider-row {
  display: flex; align-items: center; gap: 8px; margin-top: 4px;
}
.hud-slider-row input[type="range"] {
  flex: 1; accent-color: #3b82f6; height: 6px; cursor: pointer;
}
.hud-slider-label {
  font-size: 11px; color: #60a5fa; text-align: center; margin-top: 2px;
  font-variant-numeric: tabular-nums;
}
.hud-perf-bar {
  font-size: 10px; color: #6b7280; margin-top: 6px; padding-top: 6px;
  border-top: 1px solid rgba(255,255,255,0.06);
  font-variant-numeric: tabular-nums;
}
.hud-perf-bar span { color: #60a5fa; }
</style>
"""
    css_insert = (
        '<meta name="viewport" content="width=device-width, initial-scale=1.0, '
        'maximum-scale=1.0, user-scalable=no">'
        '<!-- h5-to-html ' + SCRIPT_VERSION + ' -->'
        + hud_css
    )

    html_content = inject_to_html(html_content, css_insert, wasd_script)

    out_path.write_text(html_content, encoding="utf-8")

    print(f"[OK] HTML сохранён: {out_path.resolve()}")
    print(f"     Размер: {out_path.stat().st_size / 1024:.1f} KB")
    print(f"     Данные: {n_windows} x {n_channels} | Мастер: {master_n_win} x {master_n_chan} | Метрика: {metric_key.upper()}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════

def demo_synthetic(output_html: str = "demo_activity_3d.html") -> Path:
    """Создаёт демонстрационный .h5-файл (demo_activity.h5) со сложной интерференционной структурой."""
    np.random.seed(42)
    n_windows, n_channels = 120, 150

    t_abs = np.linspace(1719000000.0, 1719000045.0, n_windows)
    t_rel = t_abs - t_abs[0]
    c = np.linspace(0, 40, n_channels)[None, :]

    tr = t_rel[:, None]
    base = (
        np.sin(0.8 * tr) * np.cos(0.4 * c) * 2.0
        + np.exp(-((tr - 15.0) ** 2) / 16.0 - (c - 20.0) ** 2 / 25.0) * 12.0
        + np.exp(-((tr - 8.0) ** 2) / 4.0 - (c - 10.0) ** 2 / 16.0) * 7.0
        + np.exp(-((tr - 28.0) ** 2) / 9.0 - (c - 30.0) ** 2 / 10.0) * 8.0
        + 3.0
    )
    base += np.random.normal(scale=0.3, size=(n_windows, n_channels))
    base = np.clip(base, 0.0, None).astype(np.float32)

    demo_path = Path("demo_activity.h5")
    with h5py.File(demo_path, "w") as f:
        f.attrs["format"] = "activity"
        f.attrs["window_seconds"] = 0.375
        f.attrs["window_samples"] = 1000
        f.attrs["sample_rate_hz"] = 1000.0
        f.attrs["channels_count"] = n_channels
        f.attrs["samples_count"] = n_windows * 1000
        f.attrs["metric_units"] = "arb"
        f.attrs["start_time"] = t_abs[0]  # Абсолютное время начала записи
        f.create_dataset("valid_count", data=np.ones((n_windows, n_channels), dtype=np.uint32) * 800)
        f.create_dataset("missing_count", data=np.zeros((n_windows, n_channels), dtype=np.uint32))
        f.create_dataset("mean", data=base, dtype="f4", compression="lzf")
        f.create_dataset("m2", data=base * 0.5, dtype="f8", compression="lzf")
        f.create_dataset("min", data=base * 0.3, dtype="f4", compression="lzf")
        f.create_dataset("max", data=base * 1.5, dtype="f4", compression="lzf")
        f.create_dataset("min_offset", data=np.zeros((n_windows, n_channels), dtype=np.int32))
        f.create_dataset("max_offset", data=np.zeros((n_windows, n_channels), dtype=np.int32))
        # Относительные сэмплы (sample counts, конвертируются в секунды при чтении)
        f.create_dataset("time_start_sample", data=(t_rel * 1000).astype(np.int64))
        f.create_dataset("time_stop_sample", data=((t_rel + 0.375) * 1000).astype(np.int64))
        distances = np.arange(n_channels, dtype=np.float64) * 0.5
        f.create_dataset("distance_m", data=distances, dtype="f8")

    print(f"[DEMO] Создан demo_activity.h5 с абсолютным временем (start_time={t_abs[0]})")
    return build_interactive_3d(demo_path, output_html=output_html, metric_key="mean")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Интерактивный общий 3D/2D график для последовательной коллекции .h5."
    )
    parser.add_argument(
        "inputs", nargs="*",
        help="Файлы или папки. Без аргументов берутся все .h5/.hdf5 рядом со скриптом.",
    )
    parser.add_argument("-o", "--output", default="combined_activity_uint16.html")
    parser.add_argument("--metric", default="mean")
    parser.add_argument("--time-stride", type=int, default=4)
    parser.add_argument("--channel-stride", type=int, default=5)
    parser.add_argument("--target-time-rows", type=int, default=6000)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        demo_synthetic(output_html=args.output)
        return

    paths: list[Path] = []
    if args.inputs:
        for raw in args.inputs:
            p = Path(raw).resolve()
            if p.is_dir():
                paths.extend(p.glob("*.h5"))
                paths.extend(p.glob("*.hdf5"))
            else:
                paths.append(p)
    else:
        folder = Path(__file__).resolve().parent
        paths.extend(folder.glob("*.h5"))
        paths.extend(folder.glob("*.hdf5"))

    paths = sorted({p.resolve() for p in paths})
    if not paths:
        print("[ERROR] H5-файлы не найдены.")
        sys.exit(1)
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"[ERROR] Файл не найден: {missing[0]}")
        sys.exit(1)

    build_interactive_3d(
        paths, output_html=args.output, metric_key=args.metric,
        time_stride=args.time_stride, channel_stride=args.channel_stride,
        target_time_rows=args.target_time_rows, percentile=args.percentile,
    )


if __name__ == "__main__":
    main()
