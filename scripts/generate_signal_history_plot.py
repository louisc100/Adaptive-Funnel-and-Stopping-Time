from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def _num(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def generate_signal_history_plot(
    source_path="data/ibkr_signals/spcx_signals.csv",
    output_path="data/ibkr_signals/spcx_signal_history.html",
    symbol="SPCX",
    k=0.8,
    z_trend=0.35,
):
    source = Path(source_path)
    output = Path(output_path)
    points = []
    markers = []
    with source.open(newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for i, row in enumerate(reader):
            signal = (row.get("signal") or "").upper()
            mid = _num(row.get("close_mid"))
            point = {
                "x": i,
                "timestamp": row.get("timestamp", ""),
                "bars": _num(row.get("bars")),
                "mid": mid,
                "bid": _num(row.get("close_bid")),
                "ask": _num(row.get("close_ask")),
                "z": _num(row.get("z")),
                "signal": signal,
                "position": row.get("position", ""),
                "note": row.get("note", ""),
            }
            points.append(point)
            if signal in ("FILLED BUY", "FILLED SELL") and mid is not None:
                markers.append({
                    "x": i,
                    "action": "BUY" if signal == "FILLED BUY" else "SELL",
                    "price": mid,
                    "timestamp": row.get("timestamp", ""),
                    "note": row.get("note", ""),
                })

    payload = {
        "symbol": symbol,
        "updatedAt": points[-1]["timestamp"] if points else "",
        "points": points,
        "markers": markers,
        "k": k,
        "zTrend": z_trend,
    }
    payload_json = json.dumps(payload, allow_nan=False)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{symbol} Signal History</title>
<style>
  :root {{ --bg:#0b1018; --panel:#111827; --grid:#283244; --text:#e5eefc; --muted:#94a3b8; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:radial-gradient(circle at top left,#172033,var(--bg) 48%); color:var(--text); font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
  main {{ width:min(1500px,100vw); margin:0 auto; padding:18px; }}
  .topbar {{ display:flex; justify-content:space-between; flex-wrap:wrap; gap:12px; margin-bottom:12px; }}
  h1 {{ margin:0; font-size:22px; }}
  .meta {{ color:var(--muted); font-size:13px; }}
  .card {{ background:rgba(17,24,39,.92); border:1px solid #263244; border-radius:18px; box-shadow:0 18px 55px rgba(0,0,0,.35); padding:14px; }}
  canvas {{ width:100%; height:700px; display:block; background:#0f1724; border-radius:12px; cursor:grab; }}
  canvas:active {{ cursor:grabbing; }}
  .controls {{ display:grid; grid-template-columns:1fr auto auto auto; gap:10px; align-items:center; margin-top:12px; }}
  button {{ color:var(--text); background:#1f2937; border:1px solid #334155; border-radius:10px; padding:8px 11px; cursor:pointer; }}
  button:hover {{ background:#273449; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:13px; margin-top:10px; color:var(--muted); font-size:13px; }}
  .swatch {{ display:inline-block; width:12px; height:3px; margin-right:5px; vertical-align:middle; }}
</style>
</head>
<body>
<main>
  <div class="topbar">
    <div><h1>{symbol} signal history</h1><div class="meta" id="meta"></div></div>
    <div class="meta">Drag to pan. Wheel/trackpad to zoom. Slider jumps through the full CSV history.</div>
  </div>
  <section class="card">
    <canvas id="chart"></canvas>
    <div class="controls">
      <input id="range" type="range" min="0" value="0" step="1">
      <button id="zoomIn">Zoom in</button>
      <button id="zoomOut">Zoom out</button>
      <button id="latest">Latest</button>
    </div>
    <div class="legend">
      <span><i class="swatch" style="background:#60a5fa"></i>Mid price</span>
      <span><i class="swatch" style="background:#14b8a6"></i>Z statistic</span>
      <span style="color:#22c55e">▲ Filled buy</span>
      <span style="color:#ef4444">▼ Filled sell</span>
      <span><i class="swatch" style="background:#ef4444"></i>+k</span>
      <span><i class="swatch" style="background:#22c55e"></i>-k</span>
      <span><i class="swatch" style="background:#f59e0b"></i>z_trend</span>
    </div>
  </section>
</main>
<script>
const payload = {payload_json};
const points = payload.points || [];
const markers = payload.markers || [];
const canvas = document.getElementById("chart");
const ctx = canvas.getContext("2d");
const range = document.getElementById("range");
let windowSize = Math.min(Math.max(150, Math.floor(points.length * 0.25)), Math.max(points.length, 1));
let start = Math.max(0, points.length - windowSize);
let dragging = false, dragStartX = 0, dragStartStart = 0;
document.getElementById("meta").textContent = `Rows ${{points.length}} | fills ${{markers.length}} | latest ${{payload.updatedAt || "n/a"}}`;
function fmt(v) {{ return v == null ? "n/a" : Number(v).toFixed(4).replace(/0+$/,"").replace(/\\.$/,""); }}
function fmtAxis(v) {{ const a=Math.abs(v); if(a>=100)return v.toFixed(2); if(a>=10)return v.toFixed(3).replace(/0+$/,"").replace(/\\.$/,""); return v.toFixed(4).replace(/0+$/,"").replace(/\\.$/,""); }}
function resize() {{ const r=window.devicePixelRatio||1,b=canvas.getBoundingClientRect(); canvas.width=Math.floor(b.width*r); canvas.height=Math.floor(b.height*r); ctx.setTransform(r,0,0,r,0,0); }}
function finite(a) {{ return a.filter(v => v != null && Number.isFinite(v)); }}
function yScale(vals, top, bottom) {{ let fs=finite(vals), mn=fs.length?Math.min(...fs):0, mx=fs.length?Math.max(...fs):1; if(Math.abs(mx-mn)<1e-9){{mn-=1;mx+=1;}} const p=(mx-mn)*.08; mn-=p; mx+=p; const s=v=>bottom-((v-mn)/(mx-mn))*(bottom-top); s.min=mn; s.max=mx; return s; }}
function grid(l,t,r,b,yFor=null,xStart=null,xEnd=null) {{ ctx.strokeStyle="#283244"; ctx.lineWidth=1; ctx.fillStyle="#94a3b8"; ctx.font="12px system-ui"; ctx.textAlign="right"; ctx.textBaseline="middle"; for(let i=0;i<=5;i++){{const y=t+(b-t)*i/5;ctx.beginPath();ctx.moveTo(l,y);ctx.lineTo(r,y);ctx.stroke(); if(yFor)ctx.fillText(fmtAxis(yFor.max-(yFor.max-yFor.min)*i/5),l-8,y);}} ctx.textAlign="center"; ctx.textBaseline="top"; for(let i=0;i<=8;i++){{const x=l+(r-l)*i/8;ctx.beginPath();ctx.moveTo(x,t);ctx.lineTo(x,b);ctx.stroke(); if(xStart!==null&&xEnd!==null)ctx.fillText(String(Math.round(xStart+(xEnd-xStart)*i/8)),x,b+7);}} ctx.textAlign="left"; ctx.textBaseline="alphabetic"; }}
function line(series,xFor,yFor,color,dash=[]) {{ ctx.strokeStyle=color; ctx.lineWidth=2; ctx.setLineDash(dash); ctx.beginPath(); let on=false; for(const p of series){{ if(p.value==null||!Number.isFinite(p.value)){{on=false;continue;}} const x=xFor(p.x), y=yFor(p.value); if(!on){{ctx.moveTo(x,y);on=true;}}else ctx.lineTo(x,y); }} ctx.stroke(); ctx.setLineDash([]); }}
function tri(x,y,up,color) {{ ctx.fillStyle=color; ctx.strokeStyle="#f8fafc"; ctx.lineWidth=1; ctx.beginPath(); if(up){{ctx.moveTo(x,y-9);ctx.lineTo(x-8,y+7);ctx.lineTo(x+8,y+7);}}else{{ctx.moveTo(x,y+9);ctx.lineTo(x-8,y-7);ctx.lineTo(x+8,y-7);}} ctx.closePath(); ctx.fill(); ctx.stroke(); }}
function draw() {{ resize(); const rect=canvas.getBoundingClientRect(); ctx.clearRect(0,0,rect.width,rect.height); if(!points.length)return; range.max=Math.max(0,points.length-windowSize); range.value=start; const l=76,r=rect.width-24,pt=34,pb=Math.floor(rect.height*.63),zt=pb+62,zb=rect.height-50; const end=Math.min(points.length,start+windowSize), vis=points.slice(start,end), denom=Math.max(vis.length-1,1); const xEnd=Math.min(points.length-1,start+windowSize-1); const xFor=x=>l+((x-start)/denom)*(r-l); const yP=yScale(vis.flatMap(p=>[p.mid,p.bid,p.ask]),pt,pb); const zVals=vis.map(p=>p.z).concat([payload.k,-payload.k,payload.zTrend]); const yZ=yScale(zVals,zt,zb); grid(l,pt,r,pb,yP,start,xEnd); grid(l,zt,r,zb,yZ,start,xEnd); line(vis.map(p=>({{x:p.x,value:p.mid}})),xFor,yP,"#60a5fa"); line(vis.map(p=>({{x:p.x,value:p.z}})),xFor,yZ,"#14b8a6"); const h=(v,c,label,d=[])=>{{ if(v==null)return; const y=yZ(v); ctx.strokeStyle=c; ctx.lineWidth=1.3; ctx.setLineDash(d); ctx.beginPath(); ctx.moveTo(l,y); ctx.lineTo(r,y); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle=c; ctx.font="12px system-ui"; ctx.fillText(label,r-56,y-5); }}; h(payload.k,"#ef4444","+k",[7,5]); h(-payload.k,"#22c55e","-k",[7,5]); h(payload.zTrend,"#f59e0b","z_trend",[2,5]); markers.filter(m=>m.price!=null&&m.x>=start&&m.x<end).forEach(m=>tri(xFor(m.x),yP(m.price),m.action==="BUY",m.action==="BUY"?"#22c55e":"#ef4444")); ctx.fillStyle="#e5eefc"; ctx.font="13px system-ui"; ctx.fillText(`Rows ${{start}}-${{Math.min(points.length-1,end-1)}} of ${{points.length-1}}`,l,22); const last=points[points.length-1]; ctx.fillStyle="#94a3b8"; ctx.fillText(`Latest mid ${{fmt(last.mid)}} | bid ${{fmt(last.bid)}} | ask ${{fmt(last.ask)}}`,r-300,22); ctx.textAlign="center"; ctx.fillText("CSV row index", (l+r)/2, rect.height-14); ctx.textAlign="left"; ctx.fillText("Price",12,pt+18); ctx.fillText("Z",26,zt+18); }}
function setStart(v) {{ start=Math.max(0,Math.min(Number(v),Math.max(0,points.length-windowSize))); draw(); }}
function zoom(f) {{ const c=start+windowSize/2; windowSize=Math.max(20,Math.min(points.length||1,Math.round(windowSize*f))); setStart(Math.round(c-windowSize/2)); }}
range.addEventListener("input",e=>setStart(e.target.value));
document.getElementById("zoomIn").onclick=()=>zoom(.7);
document.getElementById("zoomOut").onclick=()=>zoom(1.35);
document.getElementById("latest").onclick=()=>setStart(Math.max(0,points.length-windowSize));
canvas.addEventListener("wheel",e=>{{e.preventDefault();zoom(e.deltaY<0?.82:1.18);}},{{passive:false}});
canvas.addEventListener("mousedown",e=>{{dragging=true;dragStartX=e.clientX;dragStartStart=start;}});
window.addEventListener("mouseup",()=>dragging=false);
window.addEventListener("mousemove",e=>{{if(!dragging)return; const rect=canvas.getBoundingClientRect(); const moved=Math.round(-(e.clientX-dragStartX)/Math.max(rect.width-82,1)*windowSize); setStart(dragStartStart+moved);}});
window.addEventListener("resize",draw);
draw();
</script>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output, len(points), len(markers)


if __name__ == "__main__":
    output, rows, markers = generate_signal_history_plot()
    print(output)
    print(f"rows={rows} markers={markers}")
