#!/usr/bin/env python3
"""Все H5 рядом со скриптом -> один P95/uint16 3D/2D HTML."""
from __future__ import annotations
import argparse, base64, json, sys
from datetime import datetime
from pathlib import Path
import h5py, numpy as np

T_STEP, Y_STEP, TARGET, PCTL = 4, 5, 6000, 95.0


def start_time(v):
    if isinstance(v, bytes): v = v.decode()
    try: return float(v)
    except (TypeError, ValueError):
        try:
            d, t = str(v).split("T", 1); frac = t[6:].ljust(6, "0")[:6]
            return datetime.strptime(d+"T"+t[:6]+frac, "%Y%m%dT%H%M%S%f").timestamp()
        except Exception: return 0.0


def inspect(path):
    with h5py.File(path, "r") as f:
        keys=[k for k in f if isinstance(f[k],h5py.Dataset) and f[k].ndim==2 and "count" not in k and "offset" not in k]
        if not keys: raise ValueError(f"{path.name}: нет 2D-метрик")
        return {"p":path,"t":start_time(f.attrs.get("start_time",0)),"k":keys,"s":{k:f[k].shape for k in keys},"a":dict(f.attrs)}


def times(info, rows, step):
    sr=float(info["a"].get("sample_rate_hz",1) or 1)
    with h5py.File(info["p"],"r") as f:
        if "time_start_sample" in f:
            a=np.asarray(f["time_start_sample"][::step],float)/sr
            if "time_stop_sample" in f: a=(a+np.asarray(f["time_stop_sample"][::step],float)/sr)/2
        else: a=np.arange(0,rows,step,dtype=float)*float(info["a"].get("window_seconds",1))
    if info["t"]: a+=info["t"]
    return a


def load(paths, metric, ts, ys, target, pct):
    info=[inspect(p) for p in paths]; common=set(info[0]["k"])
    for x in info[1:]: common&=set(x["k"])
    if not common: raise ValueError("Нет общей 2D-метрики")
    if metric not in common: metric="mean" if "mean" in common else sorted(common)[0]
    info.sort(key=lambda x:(not x["t"],x["t"] or float("inf"),x["p"].name))
    print(f"[INFO] Файлов: {len(info)}; метрика: {metric}")
    for i,x in enumerate(info,1): print(f" {i:02}. {x['t']:.6f} | {x['p'].name}")
    channels=info[0]["s"][metric][1]
    if any(x["s"][metric][1]!=channels for x in info): raise ValueError("Разное число каналов")
    with h5py.File(info[0]["p"],"r") as f: dist=np.asarray(f["distance_m"][::ys],float) if "distance_m" in f else np.arange(0,channels,ys,dtype=float)
    parts=[]; taxis=[]
    for i,x in enumerate(info,1):
        print(f"[INFO] Чтение {i}/{len(info)}: {x['p'].name}")
        with h5py.File(x["p"],"r") as f: parts.append(np.asarray(f[metric][::ts,::ys],np.float32))
        taxis.append(times(x,x["s"][metric][0],ts))
    data=np.concatenate(parts); tx=np.concatenate(taxis); del parts,taxis
    np.nan_to_num(data,copy=False,nan=0,posinf=0,neginf=0)
    n=min(target,len(data)); edges=np.linspace(0,len(data),n+1,dtype=int)
    out=np.empty((n,data.shape[1]),np.float32); tout=np.empty(n,float)
    for i in range(n): out[i]=np.percentile(data[edges[i]:edges[i+1]],pct,axis=0); tout[i]=tx[edges[i]:edges[i+1]].mean()
    return out,tout,dist,metric,len(info)


def html(data,t,dist,metric,count,ts,ys,pct,out):
    lo,hi=float(data.min()),float(data.max())
    q=np.zeros(data.shape,"<u2") if hi<=lo else np.rint((data-lo)*65535/(hi-lo)).clip(0,65535).astype("<u2")
    vals=data[np.isfinite(data)]; c0,c1=(float(np.percentile(vals,2)),float(np.percentile(vals,98))) if vals.size else (0,1)
    if c1<=c0: c1=c0+1
    meta={"r":data.shape[0],"c":data.shape[1],"lo":lo,"hi":hi,"c0":c0,"c1":c1,"t":t.tolist(),"d":dist.tolist(),"z":base64.b64encode(q.tobytes()).decode(),"m":metric,"n":count,"ts":ts,"ys":ys,"p":pct}
    page='''<!doctype html><meta charset="utf-8"><title>H5 P95 uint16</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>html,body,#g{width:100%;height:100%;margin:0;background:#090b12;overflow:hidden}#u{position:fixed;z-index:9;top:12px;left:12px;width:315px;padding:12px;background:#121622ee;color:#eee;border:1px solid #ffffff30;border-radius:9px;font:12px Segoe UI}button,select{margin:4px;padding:5px;background:#ffffff18;color:#eee;border:1px solid #ffffff30;border-radius:5px}input{width:150px}</style>
<div id="g"></div><div id="u"><b>Объединённый H5 · P95 · uint16</b><div id="s"></div><button id="mode">2D</button><button id="reset">Сброс</button><select id="pal"><option>Turbo</option><option>Viridis</option><option>Inferno</option><option>Plasma</option></select><br>Разрешение <input id="res" type="range" min="30" max="1600" value="200"><span id="rv"></span><div>Полные данные сохранены до 6000×N; рендер ограничен 1600.</div></div>
<script>const M=__M__,b=atob(M.z),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);const q=new Uint16Array(u.buffer),z=new Float32Array(q.length),sc=M.hi>M.lo?(M.hi-M.lo)/65535:0;for(let i=0;i<q.length;i++)z[i]=M.lo+q[i]*sc;let md=3,pal='Turbo',cam=null,timer;const G=document.getElementById('g'),R=document.getElementById('res'),V=document.getElementById('rv');document.getElementById('s').textContent=M.n+' файлов · '+M.r+'×'+M.c+' · P'+M.p+' · '+M.ts+'×'+M.ys;
function sample(n){let a=Math.max(1,Math.ceil(M.r/n)),c=Math.max(1,Math.ceil(M.c/n)),x=[],y=[],v=[];for(let j=0;j<M.c;j+=c)y.push(M.d[j]);for(let i=0;i<M.r;i+=a){x.push(M.t[i]-M.t[0]);let r=[];for(let j=0;j<M.c;j+=c)r.push(z[i*M.c+j]);v.push(r)}return{x:x,y:y,z:v}}
function draw(){let A=sample(+R.value);V.textContent=' '+A.z.length+'×'+A.y.length;if(G._fullLayout&&G._fullLayout.scene)cam=G._fullLayout.scene.camera;let d,l;if(md==3){d=[{type:'surface',x:A.x,y:A.y,z:A.z,colorscale:pal,cmin:M.c0,cmax:M.c1}];l={template:'plotly_dark',margin:{l:0,r:0,b:0,t:45},title:'H5 P95 uint16',scene:{xaxis:{title:'Время от старта, с'},yaxis:{title:'Расстояние, м'},zaxis:{title:M.m},camera:cam||{eye:{x:1.6,y:-1.4,z:1}}}}}else{let w=[];for(let j=0;j<A.y.length;j++){let r=[];for(let i=0;i<A.z.length;i++)r.push(A.z[i][j]);w.push(r)}d=[{type:'heatmap',x:A.x,y:A.y,z:w,colorscale:pal,zmin:M.c0,zmax:M.c1}];l={template:'plotly_dark',margin:{l:60,r:10,b:55,t:45},title:'H5 P95 uint16 · 2D',xaxis:{title:'Время от старта, с'},yaxis:{title:'Расстояние, м'}}}Plotly.react(G,d,l,{responsive:true})}
document.getElementById('mode').onclick=()=>{md=md==3?2:3;R.value=md==3?200:Math.min(1200,M.r);draw()};document.getElementById('reset').onclick=()=>{cam={eye:{x:1.6,y:-1.4,z:1}};draw()};document.getElementById('pal').onchange=e=>{pal=e.target.value;draw()};R.oninput=()=>{clearTimeout(timer);timer=setTimeout(draw,100)};draw();</script>'''.replace('__M__',json.dumps(meta,separators=(',',':')))
    out.write_text(page,encoding='utf-8'); print(f"[OK] {out} | {out.stat().st_size/1048576:.1f} MiB")


def main():
    p=argparse.ArgumentParser(); p.add_argument('inputs',nargs='*'); p.add_argument('-o','--output',default='combined_activity_uint16.html'); p.add_argument('--metric',default='mean'); p.add_argument('--time-stride',type=int,default=T_STEP); p.add_argument('--channel-stride',type=int,default=Y_STEP); p.add_argument('--target-time-rows',type=int,default=TARGET); p.add_argument('--percentile',type=float,default=PCTL); a=p.parse_args()
    try:
        folder=Path(__file__).resolve().parent; paths=[]
        for v in a.inputs:
            x=Path(v).resolve(); paths+=list(x.glob('*.h5'))+list(x.glob('*.hdf5')) if x.is_dir() else [x]
        if not paths: paths=list(folder.glob('*.h5'))+list(folder.glob('*.hdf5'))
        if not paths: raise FileNotFoundError('H5 рядом со скриптом не найдены')
        d,t,y,m,n=load(sorted(set(paths)),a.metric,a.time_stride,a.channel_stride,a.target_time_rows,a.percentile)
        html(d,t,y,m,n,a.time_stride,a.channel_stride,a.percentile,Path(a.output).resolve())
    except Exception as e: print('[ERROR]',e); sys.exit(1)
if __name__=='__main__': main()
