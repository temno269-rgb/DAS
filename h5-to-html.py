#!/usr/bin/env python3
"""Собирает все raw H5 DAS рядом со скриптом в один интерактивный HTML.

Ожидаемая структура исходников:
  DataStreams/PassThrough/Phase#...  shape=(time, channels), dtype=uint16,
  NumericType=fp16, DataStartTime, DataFrequency, MetersPerChannel.

Обработка по умолчанию:
  - сортировка файлов по DataStartTime;
  - каждый 4-й отсчёт по времени;
  - каждый 5-й канал;
  - P95 по временным блокам до 6000 итоговых строк;
  - упаковка итоговой матрицы в uint16 внутри одного HTML.
"""

from __future__ import annotations

import argparse
import base64
import bisect
import json
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

TIME_STRIDE = 4
CHANNEL_STRIDE = 5
TARGET_TIME_ROWS = 6000
PERCENTILE = 95.0


def parse_compact_time(value: object) -> float:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        pass
    try:
        date_part, time_part = text.split("T", 1)
        hhmmss = time_part[:6]
        fraction = time_part[6:].ljust(6, "0")[:6]
        dt = datetime.strptime(
            f"{date_part}T{hhmmss}{fraction}", "%Y%m%dT%H%M%S%f"
        ).replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0


def find_phase_dataset(h5_file: h5py.File) -> str:
    candidates: list[tuple[int, str]] = []

    def visitor(name: str, obj: object) -> None:
        if not isinstance(obj, h5py.Dataset) or obj.ndim != 2:
            return
        data_type = obj.attrs.get("DataType", "")
        if isinstance(data_type, bytes):
            data_type = data_type.decode("utf-8", errors="replace")
        if str(data_type).lower() == "phase" or "/phase#" in f"/{name}".lower():
            candidates.append((int(obj.shape[0]) * int(obj.shape[1]), name))

    h5_file.visititems(visitor)
    if not candidates:
        raise ValueError("В H5 не найден двумерный dataset Phase")
    candidates.sort(reverse=True)
    return candidates[0][1]


def inspect_file(path: Path) -> dict:
    with h5py.File(path, "r") as h5_file:
        dataset_path = find_phase_dataset(h5_file)
        dataset = h5_file[dataset_path]
        attrs = {key: dataset.attrs[key] for key in dataset.attrs.keys()}
        start = parse_compact_time(attrs.get("DataStartTime", 0.0))
        frequency = float(attrs.get("DataFrequency", 500.0) or 500.0)
        meters_per_channel = float(attrs.get("MetersPerChannel", 1.0) or 1.0)
        numeric_type = attrs.get("NumericType", "")
        if isinstance(numeric_type, bytes):
            numeric_type = numeric_type.decode("utf-8", errors="replace")
        return {
            "path": path,
            "dataset_path": dataset_path,
            "shape": tuple(int(x) for x in dataset.shape),
            "start": start,
            "frequency": frequency if frequency > 0 else 500.0,
            "meters_per_channel": meters_per_channel,
            "numeric_type": str(numeric_type).lower(),
        }


def decode_phase(raw: np.ndarray, numeric_type: str) -> np.ndarray:
    raw = np.asarray(raw)
    if "fp16" in numeric_type or "float16" in numeric_type:
        words = np.asarray(raw, dtype="<u2")
        values = words.view("<f2").astype(np.float32)
    else:
        values = raw.astype(np.float32, copy=False)
    return np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def collect_h5_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    if inputs:
        for value in inputs:
            path = Path(value).expanduser().resolve()
            if path.is_dir():
                paths.extend(path.glob("*.h5"))
                paths.extend(path.glob("*.hdf5"))
            else:
                paths.append(path)
    else:
        folder = Path(__file__).resolve().parent
        paths.extend(folder.glob("*.h5"))
        paths.extend(folder.glob("*.hdf5"))

    unique = sorted({path.resolve() for path in paths if path.is_file()})
    if not unique:
        raise FileNotFoundError("Рядом со скриптом не найдены файлы .h5 или .hdf5")
    return unique


def aggregate_files(
    paths: list[Path],
    time_stride: int,
    channel_stride: int,
    target_rows: int,
    percentile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    infos = [inspect_file(path) for path in paths]
    infos.sort(key=lambda item: (
        item["start"] <= 0.0,
        item["start"] if item["start"] > 0.0 else float("inf"),
        item["path"].name,
    ))

    min_channels = min(info["shape"][1] for info in infos)
    sampled_rows = [
        (info["shape"][0] + time_stride - 1) // time_stride for info in infos
    ]
    cumulative = np.zeros(len(infos) + 1, dtype=np.int64)
    cumulative[1:] = np.cumsum(sampled_rows, dtype=np.int64)
    total_sampled = int(cumulative[-1])
    output_rows = min(target_rows, total_sampled)
    if output_rows <= 0:
        raise ValueError("В Phase dataset нет временных отсчётов")

    edges = np.linspace(0, total_sampled, output_rows + 1, dtype=np.int64)
    channels = np.arange(0, min_channels, channel_stride, dtype=np.int64)
    meters = float(infos[0]["meters_per_channel"])
    distance = channels.astype(np.float64) * meters

    sampled_time_parts: list[np.ndarray] = []
    fallback_start = 0.0
    for info, count in zip(infos, sampled_rows):
        start = info["start"] if info["start"] > 0.0 else fallback_start
        times = start + np.arange(count, dtype=np.float64) * (
            time_stride / info["frequency"]
        )
        sampled_time_parts.append(times)
        fallback_start = float(times[-1] + time_stride / info["frequency"]) if count else start
    sampled_times = np.concatenate(sampled_time_parts)

    result = np.empty((output_rows, len(channels)), dtype=np.float32)
    output_times = np.empty(output_rows, dtype=np.float64)

    print(f"[INFO] Найдено файлов: {len(infos)}")
    print(f"[INFO] Phase после шага {time_stride}x{channel_stride}: "
          f"{total_sampled} x {len(channels)}")
    print(f"[INFO] Итоговая матрица: {output_rows} x {len(channels)}, P{percentile:g}")
    for index, info in enumerate(infos, 1):
        stamp = datetime.fromtimestamp(info["start"], tz=timezone.utc).isoformat() \
            if info["start"] > 0 else "без времени"
        print(f"  {index:02d}. {stamp} | {info['path'].name}")

    with ExitStack() as stack:
        opened = [stack.enter_context(h5py.File(info["path"], "r")) for info in infos]
        datasets = [h5_file[info["dataset_path"]] for h5_file, info in zip(opened, infos)]

        for out_index in range(output_rows):
            global_start = int(edges[out_index])
            global_stop = int(edges[out_index + 1])
            output_times[out_index] = float(np.mean(sampled_times[global_start:global_stop]))

            pieces: list[np.ndarray] = []
            position = global_start
            file_index = max(0, bisect.bisect_right(cumulative, position) - 1)

            while position < global_stop and file_index < len(infos):
                file_global_start = int(cumulative[file_index])
                file_global_stop = int(cumulative[file_index + 1])
                overlap_stop = min(global_stop, file_global_stop)
                local_start = position - file_global_start
                local_stop = overlap_stop - file_global_start

                raw_start = local_start * time_stride
                raw_stop = min(infos[file_index]["shape"][0], local_stop * time_stride)
                raw = datasets[file_index][
                    raw_start:raw_stop:time_stride,
                    0:min_channels:channel_stride,
                ]
                pieces.append(decode_phase(raw, infos[file_index]["numeric_type"]))
                position = overlap_stop
                file_index += 1

            block = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)
            result[out_index] = np.percentile(block, percentile, axis=0)

            if (out_index + 1) % 250 == 0 or out_index + 1 == output_rows:
                print(f"[INFO] P95: {out_index + 1}/{output_rows}")

    return result, output_times, distance, infos


def build_html(
    data: np.ndarray,
    times: np.ndarray,
    distance: np.ndarray,
    infos: list[dict],
    output: Path,
    time_stride: int,
    channel_stride: int,
    percentile: float,
) -> None:
    finite = data[np.isfinite(data)]
    if finite.size:
        value_min = float(finite.min())
        value_max = float(finite.max())
        color_min = float(np.percentile(finite, 2))
        color_max = float(np.percentile(finite, 98))
    else:
        value_min, value_max, color_min, color_max = 0.0, 1.0, 0.0, 1.0
    if value_max <= value_min:
        value_max = value_min + 1.0
    if color_max <= color_min:
        color_max = color_min + 1.0

    quantized = np.rint(
        (data - value_min) * (65535.0 / (value_max - value_min))
    ).clip(0, 65535).astype("<u2")

    metadata = {
        "rows": int(data.shape[0]),
        "cols": int(data.shape[1]),
        "valueMin": value_min,
        "valueMax": value_max,
        "colorMin": color_min,
        "colorMax": color_max,
        "times": times.tolist(),
        "distance": distance.tolist(),
        "payload": base64.b64encode(quantized.tobytes()).decode("ascii"),
        "fileCount": len(infos),
        "timeStride": time_stride,
        "channelStride": channel_stride,
        "percentile": percentile,
    }

    page = r'''<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DAS Phase P95</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
html,body,#graph{width:100%;height:100%;margin:0;background:#080b12;overflow:hidden}
#panel{position:fixed;z-index:10;left:14px;top:14px;width:340px;padding:13px;background:#121725ee;color:#e8ecf4;border:1px solid #ffffff2b;border-radius:10px;font:12px "Segoe UI",sans-serif;box-shadow:0 10px 35px #0009}
#panel b{font-size:14px}button,select{margin:6px 4px 2px 0;padding:6px 9px;color:#eee;background:#ffffff15;border:1px solid #ffffff30;border-radius:6px;cursor:pointer}input{width:175px;vertical-align:middle}.muted{color:#9aa5b5;line-height:1.45;margin-top:7px}
</style>
</head>
<body>
<div id="graph"></div>
<div id="panel"><b>DAS Phase · объединённый P95</b><div id="info" class="muted"></div><button id="mode">Переключить в 2D</button><button id="reset">Сброс камеры</button><select id="palette"><option>Turbo</option><option>Viridis</option><option>Inferno</option><option>Plasma</option><option>Cividis</option></select><div>Разрешение: <input id="resolution" type="range" min="40" max="1400" value="240"><span id="resolutionValue"></span></div><div class="muted">В HTML хранится полная агрегированная матрица. Ползунок меняет только плотность отображения.</div></div>
<script>
const M=__METADATA__;
const binary=atob(M.payload), bytes=new Uint8Array(binary.length);
for(let i=0;i<binary.length;i++) bytes[i]=binary.charCodeAt(i);
const packed=new Uint16Array(bytes.buffer), values=new Float32Array(packed.length);
const scale=(M.valueMax-M.valueMin)/65535;
for(let i=0;i<packed.length;i++) values[i]=M.valueMin+packed[i]*scale;
const graph=document.getElementById('graph'), slider=document.getElementById('resolution'), label=document.getElementById('resolutionValue');
let mode='3d', palette='Turbo', camera=null, timer=null;
document.getElementById('info').textContent=M.fileCount+' файлов · '+M.rows+'×'+M.cols+' · P'+M.percentile+' · шаг '+M.timeStride+'×'+M.channelStride;
function sampled(limit){
  const rowStep=Math.max(1,Math.ceil(M.rows/limit));
  const colStep=Math.max(1,Math.ceil(M.cols/limit));
  const x=[], y=[], z=[];
  const t0=M.times[0];
  for(let c=0;c<M.cols;c+=colStep) y.push(M.distance[c]);
  for(let r=0;r<M.rows;r+=rowStep){
    x.push(M.times[r]-t0);
    const row=[];
    for(let c=0;c<M.cols;c+=colStep) row.push(values[r*M.cols+c]);
    z.push(row);
  }
  return {x,y,z};
}
function render(){
  const A=sampled(+slider.value); label.textContent=' '+A.z.length+'×'+A.y.length;
  if(graph._fullLayout&&graph._fullLayout.scene) camera=graph._fullLayout.scene.camera;
  let traces,layout;
  if(mode==='3d'){
    traces=[{type:'surface',x:A.x,y:A.y,z:A.z,colorscale:palette,cmin:M.colorMin,cmax:M.colorMax,hovertemplate:'Время: %{x:.3f} с<br>Расстояние: %{y:.2f} м<br>Phase: %{z:.5g}<extra></extra>'}];
    layout={template:'plotly_dark',title:'DAS Phase · P95',margin:{l:0,r:0,b:0,t:42},scene:{xaxis:{title:'Время от начала, с'},yaxis:{title:'Расстояние, м'},zaxis:{title:'Phase'},camera:camera||{eye:{x:1.55,y:-1.45,z:1.0}},aspectratio:{x:1.7,y:1.3,z:1.0}}};
  }else{
    const heat=[];
    for(let c=0;c<A.y.length;c++){const row=[];for(let r=0;r<A.z.length;r++)row.push(A.z[r][c]);heat.push(row)}
    traces=[{type:'heatmap',x:A.x,y:A.y,z:heat,colorscale:palette,zmin:M.colorMin,zmax:M.colorMax,hovertemplate:'Время: %{x:.3f} с<br>Расстояние: %{y:.2f} м<br>Phase: %{z:.5g}<extra></extra>'}];
    layout={template:'plotly_dark',title:'DAS Phase · P95 · 2D',margin:{l:65,r:15,b:55,t:42},xaxis:{title:'Время от начала, с'},yaxis:{title:'Расстояние, м'}};
  }
  Plotly.react(graph,traces,layout,{responsive:true,scrollZoom:true});
}
document.getElementById('mode').onclick=()=>{mode=mode==='3d'?'2d':'3d';document.getElementById('mode').textContent=mode==='3d'?'Переключить в 2D':'Переключить в 3D';slider.value=mode==='3d'?240:900;render()};
document.getElementById('reset').onclick=()=>{camera={eye:{x:1.55,y:-1.45,z:1.0}};render()};
document.getElementById('palette').onchange=e=>{palette=e.target.value;render()};
slider.oninput=()=>{clearTimeout(timer);timer=setTimeout(render,120)};
render();
</script>
</body>
</html>'''.replace("__METADATA__", json.dumps(metadata, separators=(",", ":")))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    print(f"[OK] HTML создан: {output}")
    print(f"[OK] Размер HTML: {output.stat().st_size / 1048576:.1f} MiB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Объединяет raw DAS Phase H5 в один P95 HTML")
    parser.add_argument("inputs", nargs="*", help="Файлы или папки; без аргументов ищет H5 рядом со скриптом")
    parser.add_argument("-o", "--output", default="combined_phase_p95_uint16.html")
    parser.add_argument("--time-stride", type=int, default=TIME_STRIDE)
    parser.add_argument("--channel-stride", type=int, default=CHANNEL_STRIDE)
    parser.add_argument("--target-time-rows", type=int, default=TARGET_TIME_ROWS)
    parser.add_argument("--percentile", type=float, default=PERCENTILE)
    args = parser.parse_args()

    try:
        if args.time_stride < 1 or args.channel_stride < 1 or args.target_time_rows < 1:
            raise ValueError("Шаги и target-time-rows должны быть не меньше 1")
        if not 0.0 <= args.percentile <= 100.0:
            raise ValueError("percentile должен находиться в диапазоне 0..100")

        paths = collect_h5_paths(args.inputs)
        data, times, distance, infos = aggregate_files(
            paths,
            time_stride=args.time_stride,
            channel_stride=args.channel_stride,
            target_rows=args.target_time_rows,
            percentile=args.percentile,
        )
        build_html(
            data,
            times,
            distance,
            infos,
            Path(args.output).expanduser().resolve(),
            args.time_stride,
            args.channel_stride,
            args.percentile,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
