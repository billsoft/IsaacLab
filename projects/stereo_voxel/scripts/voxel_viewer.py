"""语义体素 3D 可视化工具
=========================
Python HTTP 后端 + Three.js 前端（本地离线，无需 CDN）。

功能：
  - 3D 语义体素（按类别着色，鼠标旋转/缩放/平移）
  - 同步左右眼立体图像
  - 类别图例（点击可切换显示/隐藏）
  - 帧导航（滑块 + 键盘 ← →）

运行（仅需 numpy，不需要 Isaac Sim）：
    isaaclab.bat -p projects/stereo_voxel/scripts/voxel_viewer.py
    isaaclab.bat -p projects/stereo_voxel/scripts/voxel_viewer.py --data_dir X --port 8080
"""

import argparse
import http.server
import json
import os
import re
import socket
import sys
import threading
import webbrowser

import numpy as np

# ---------------------------------------------------------------------------
# semantic_classes 导入
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from semantic_classes import CLASS_COLORS, CLASS_NAMES, CLASS_NAMES_ZH, FREE, NUM_CLASSES, UNOBSERVED

# vendor 目录（three.min.js + OrbitControls.js）
VENDOR_DIR = os.path.join(SCRIPT_DIR, "vendor")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DEFAULT_DATA = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "output"))

parser = argparse.ArgumentParser(description="Voxel 3D Viewer")
parser.add_argument("--data_dir", default=DEFAULT_DATA, help="Dataset output directory")
parser.add_argument("--port", type=int, default=7860, help="HTTP port")
parser.add_argument("--no_browser", action="store_true")
args = parser.parse_args()

DATA_DIR = os.path.abspath(args.data_dir)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def scan_frames(data_dir):
    left = os.path.join(data_dir, "left")
    if not os.path.isdir(left):
        return []
    ids = []
    for f in os.listdir(left):
        m = re.match(r"frame_(\d+)\.png$", f)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def load_voxels_json(data_dir, fid):
    path = os.path.join(data_dir, "voxel", f"frame_{fid:06d}_semantic.npz")
    if not os.path.isfile(path):
        return {"voxels": [], "stats": {}}
    sem = np.load(path)["data"]
    mask = (sem != FREE) & (sem != UNOBSERVED)
    idx = np.argwhere(mask)
    voxels = [[int(r[0]), int(r[1]), int(r[2]), int(sem[r[0], r[1], r[2]])] for r in idx]
    stats = {}
    for cid in range(NUM_CLASSES):
        c = int(np.sum(sem == cid))
        if c > 0:
            stats[str(cid)] = c
    stats["255"] = int(np.sum(sem == UNOBSERVED))
    stats["total"] = int(sem.size)
    return {"voxels": voxels, "stats": stats}


def load_meta(data_dir, fid):
    p = os.path.join(data_dir, "meta", f"frame_{fid:06d}.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def class_info_js():
    d = {}
    for cid in range(NUM_CLASSES):
        r, g, b = CLASS_COLORS[cid]
        d[cid] = {"name": CLASS_NAMES[cid], "zh": CLASS_NAMES_ZH[cid], "rgb": [r, g, b]}
    return json.dumps(d)


# ---------------------------------------------------------------------------
# 自动找可用端口
# ---------------------------------------------------------------------------
def find_free_port(start, tries=20):
    for p in range(start, start + tries):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", p))
            s.close()
            return p
        except OSError:
            continue
    return None


# ---------------------------------------------------------------------------
# HTML 前端
# ---------------------------------------------------------------------------
FRONTEND = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>Voxel Viewer</title>
<style>
:root{--bg:#111827;--panel:#1f2937;--border:#374151;--accent:#f59e0b;--text:#e5e7eb;--dim:#9ca3af}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);
     overflow:hidden;height:100vh;display:flex;flex-direction:column}

.bar{display:flex;align-items:center;gap:10px;padding:6px 14px;background:var(--panel);
     border-bottom:1px solid var(--border);height:44px;flex-shrink:0;flex-wrap:wrap}
.bar label{font-size:12px;color:var(--dim)}
.bar input[type=text]{background:#111827;border:1px solid var(--border);color:var(--text);
     padding:3px 8px;border-radius:4px;width:320px;font-size:12px}
.bar button{background:#374151;border:1px solid var(--border);color:var(--text);
     padding:4px 12px;border-radius:4px;cursor:pointer;font-size:13px}
.bar button:hover{background:#4b5563}
.bar .sep{width:1px;height:22px;background:var(--border)}
#slider{width:200px;accent-color:var(--accent)}
#flab{font-weight:700;color:var(--accent);min-width:160px;font-size:13px}

.main{display:flex;flex:1;overflow:hidden}

.side{width:320px;min-width:260px;background:var(--panel);border-right:1px solid var(--border);
      overflow-y:auto;padding:8px;flex-shrink:0}
.side h3{font-size:11px;color:var(--dim);margin:8px 0 3px;text-transform:uppercase;letter-spacing:1px}
.side img{width:100%;border-radius:4px;border:1px solid var(--border);display:block}

.stats{margin-top:8px;padding:6px;background:#111827;border-radius:4px;font-size:11px}
.srow{display:flex;align-items:center;gap:5px;padding:1px 0}
.sw{width:10px;height:10px;border-radius:2px;flex-shrink:0}
.sn{flex:1;color:var(--dim)}
.sc{font-family:monospace;color:var(--accent);font-size:11px}

.view{flex:1;position:relative}
canvas{display:block;width:100%;height:100%}

.legend{position:absolute;bottom:10px;left:10px;background:rgba(31,41,55,.93);
        border:1px solid var(--border);border-radius:6px;padding:8px 12px;
        font-size:11px;max-height:55vh;overflow-y:auto;user-select:none}
.legend h4{color:var(--accent);margin-bottom:4px;font-size:12px}
.lg{display:flex;align-items:center;gap:5px;padding:2px 0;cursor:pointer}
.lg:hover{opacity:.75}
.lg.off .ls{opacity:.25}
.lg.off .lt{text-decoration:line-through;opacity:.35}
.ls{width:12px;height:12px;border-radius:2px;border:1px solid rgba(255,255,255,.15);flex-shrink:0}
.lt{white-space:nowrap}

.hud{position:absolute;top:10px;left:10px;background:rgba(31,41,55,.8);border-radius:4px;
     padding:5px 10px;font-size:11px;color:var(--dim);pointer-events:none}
#ld{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
    font-size:16px;color:var(--accent)}
</style></head>
<body>

<div class="bar">
  <label>Dir:</label>
  <input type="text" id="dirIn" value="">
  <button onclick="chDir()">Load</button>
  <div class="sep"></div>
  <button onclick="go(-1)">&#9664;</button>
  <input type="range" id="slider" min="0" max="0" value="0" oninput="go(0,+this.value)">
  <button onclick="go(+1)">&#9654;</button>
  <span id="flab">&mdash;</span>
</div>

<div class="main">
  <div class="side">
    <h3>Left Eye</h3>
    <img id="imgL" alt="left">
    <h3>Right Eye</h3>
    <img id="imgR" alt="right">
    <h3>Statistics</h3>
    <div class="stats" id="statsBox"></div>
  </div>
  <div class="view" id="vp">
    <canvas id="c"></canvas>
    <div class="hud" id="hud">&mdash;</div>
    <div class="legend" id="legend"></div>
    <div id="ld">Loading&hellip;</div>
  </div>
</div>

<!-- Three.js 从本地 /vendor/ 加载 -->
<script src="/vendor/three.min.js"></script>
<script src="/vendor/OrbitControls.js"></script>

<script>
const CI=__CLASS_JSON__;
const VS=0.1,NX=72,NY=60,NZ=32,CX=36,CY=30,ZG=7;
let frames=[],ci=0,hidden=new Set(),curVox=[],mesh=null;

/* === Three.js === */
const cvs=document.getElementById('c');
const ren=new THREE.WebGLRenderer({canvas:cvs,antialias:true});
ren.setPixelRatio(Math.min(devicePixelRatio,2));
ren.setClearColor(0x111827);
const sc=new THREE.Scene();
const cam=new THREE.PerspectiveCamera(45,1,0.05,200);
cam.position.set(6,5,5);
const ctrl=new THREE.OrbitControls(cam,cvs);
ctrl.enableDamping=true; ctrl.dampingFactor=0.1; ctrl.target.set(0,1,0);

sc.add(new THREE.AmbientLight(0x8899bb,0.5));
const d1=new THREE.DirectionalLight(0xffffff,0.9); d1.position.set(8,12,6); sc.add(d1);
const d2=new THREE.DirectionalLight(0x4466aa,0.3); d2.position.set(-4,6,-8); sc.add(d2);
sc.add(new THREE.GridHelper(8,40,0x334155,0x1f2937));

// bounding wireframe
(function(){
  const g=new THREE.BoxGeometry(NX*VS,NZ*VS,NY*VS);
  const e=new THREE.EdgesGeometry(g);
  const l=new THREE.LineSegments(e,new THREE.LineBasicMaterial({color:0x374151}));
  l.position.set(0,-ZG*VS+NZ*VS/2,0); sc.add(l);
})();

// axes
const ax=new THREE.AxesHelper(0.8);
ax.position.set(-NX*VS/2,-ZG*VS,-NY*VS/2); sc.add(ax);

/* === Voxel rendering === */
const boxG=new THREE.BoxGeometry(VS*0.9,VS*0.9,VS*0.9);

function v2p(i,j,k){
  return new THREE.Vector3((i-CX+.5)*VS,(k-ZG+.5)*VS,(j-CY+.5)*VS);
}

function rebuild(){
  if(mesh){sc.remove(mesh);mesh.geometry.dispose();mesh.material.dispose();mesh=null;}
  const vis=curVox.filter(v=>!hidden.has(v[3]));
  if(!vis.length){hud(0);return;}
  const mat=new THREE.MeshLambertMaterial({vertexColors:true});
  const im=new THREE.InstancedMesh(boxG,mat,vis.length);
  const dm=new THREE.Object3D(), cl=new THREE.Color();
  for(let n=0;n<vis.length;n++){
    const[i,j,k,cid]=vis[n];
    dm.position.copy(v2p(i,j,k)); dm.updateMatrix();
    im.setMatrixAt(n,dm.matrix);
    const rgb=CI[cid]?.rgb||[150,150,150];
    cl.setRGB(rgb[0]/255,rgb[1]/255,rgb[2]/255);
    im.setColorAt(n,cl);
  }
  im.instanceMatrix.needsUpdate=true;
  if(im.instanceColor)im.instanceColor.needsUpdate=true;
  sc.add(im); mesh=im; hud(vis.length);
}

function hud(n){
  document.getElementById('hud').textContent=
    'Occupied: '+n.toLocaleString()+' | Grid '+NX+'x'+NY+'x'+NZ+' = '+(NX*NY*NZ).toLocaleString()+' | Drag=rotate  Scroll=zoom  RightDrag=pan';
}

/* === Legend === */
function mkLeg(){
  let h='<h4>Semantic Classes</h4>';
  for(let c=0;c<18;c++){
    const i=CI[c]; if(!i)continue;
    const[r,g,b]=i.rgb;
    h+='<div class="lg'+(hidden.has(c)?' off':'')+'" onclick="tog('+c+')"><div class="ls" style="background:rgb('+r+','+g+','+b+')"></div><span class="lt">'+c+' '+i.zh+' ('+i.name+')</span></div>';
  }
  document.getElementById('legend').innerHTML=h;
}
function tog(c){hidden.has(c)?hidden.delete(c):hidden.add(c);mkLeg();rebuild();}

/* === Data fetch === */
async function init(){
  try{
    const d=await(await fetch('/api/frames')).json();
    frames=d.frames;
    document.getElementById('dirIn').value=d.data_dir;
    const sl=document.getElementById('slider');
    sl.max=Math.max(0,frames.length-1); sl.value=0; ci=0;
    if(frames.length) loadFrame(0);
    else document.getElementById('ld').textContent='No frames in '+d.data_dir;
  }catch(e){document.getElementById('ld').textContent='API error: '+e.message;}
}

async function loadFrame(idx){
  if(idx<0||idx>=frames.length)return;
  ci=idx;
  const fid=frames[idx],fs=String(fid).padStart(6,'0');
  document.getElementById('flab').textContent='frame_'+fs+'  ('+(idx+1)+' / '+frames.length+')';
  document.getElementById('slider').value=idx;
  document.getElementById('ld').style.display='block';
  document.getElementById('ld').textContent='Loading frame '+fs+'...';

  // images
  document.getElementById('imgL').src='/img/left/frame_'+fs+'.png?'+Date.now();
  document.getElementById('imgR').src='/img/right/frame_'+fs+'.png?'+Date.now();

  // voxels
  try{
    const d=await(await fetch('/api/voxel/'+fid)).json();
    curVox=d.voxels||[];
    rebuild();
    fillStats(d.stats||{});
    document.getElementById('ld').style.display='none';
  }catch(e){
    document.getElementById('ld').textContent='Error: '+e.message;
    console.error(e);
  }
}

function fillStats(st){
  const el=document.getElementById('statsBox');
  if(!st||!Object.keys(st).length){el.innerHTML='';return;}
  let h='';
  for(let c=0;c<18;c++){
    const n=st[String(c)]||0; if(!n&&c)continue;
    const i=CI[c],pct=(n/(st.total||1)*100).toFixed(1);
    h+='<div class="srow"><div class="sw" style="background:rgb('+i.rgb+')"></div><span class="sn">'+i.zh+'</span><span class="sc">'+n.toLocaleString()+' ('+pct+'%)</span></div>';
  }
  const un=st['255']||0;
  if(un){const pct=(un/(st.total||1)*100).toFixed(1);
    h+='<div class="srow"><div class="sw" style="background:#333"></div><span class="sn">unobserved</span><span class="sc">'+un.toLocaleString()+' ('+pct+'%)</span></div>';}
  el.innerHTML=h;
}

/* === Nav === */
function go(delta,abs){
  if(abs!==undefined)loadFrame(parseInt(abs));
  else loadFrame(ci+delta);
}
function chDir(){
  const v=document.getElementById('dirIn').value.trim();
  if(!v)return;
  fetch('/api/set_dir',{method:'POST',body:v}).then(()=>init());
}
document.addEventListener('keydown',function(e){
  if(e.key==='ArrowLeft'||e.key==='a')go(-1);
  if(e.key==='ArrowRight'||e.key==='d')go(+1);
  if(e.key==='Home')go(0,0);
  if(e.key==='End')go(0,frames.length-1);
});

/* === Resize + loop === */
function onRz(){
  const vp=document.getElementById('vp');
  ren.setSize(vp.clientWidth,vp.clientHeight);
  cam.aspect=vp.clientWidth/vp.clientHeight;
  cam.updateProjectionMatrix();
}
window.addEventListener('resize',onRz);
if(typeof ResizeObserver!=='undefined')new ResizeObserver(onRz).observe(document.getElementById('vp'));
(function lp(){requestAnimationFrame(lp);ctrl.update();ren.render(sc,cam);})();

mkLeg(); onRz(); init();
</script></body></html>"""


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------
class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "VoxelViewer/2.0"

    def log_message(self, fmt, *a):
        msg = str(a[0] if a else "")
        if "/api/" in msg or "vendor" in msg:
            super().log_message(fmt, *a)

    def do_GET(self):
        path = self.path.split("?")[0]

        # 首页
        if path in ("/", ""):
            return self._html()

        # vendor JS 文件（Three.js + OrbitControls）
        m = re.match(r"/vendor/(.+\.js)$", path)
        if m:
            fp = os.path.join(VENDOR_DIR, m.group(1))
            return self._file(fp, "application/javascript")

        # API: 帧列表
        if path == "/api/frames":
            return self._json({"frames": scan_frames(DATA_DIR), "data_dir": DATA_DIR})

        # API: 体素数据
        m = re.match(r"/api/voxel/(\d+)$", path)
        if m:
            fid = int(m.group(1))
            d = load_voxels_json(DATA_DIR, fid)
            d["meta"] = load_meta(DATA_DIR, fid)
            return self._json(d)

        # 图片: /img/left/frame_000042.png
        m = re.match(r"/img/(left|right)/(.+\.png)$", path)
        if m:
            fp = os.path.join(DATA_DIR, m.group(1), m.group(2))
            return self._file(fp, "image/png")

        self.send_error(404)

    def do_POST(self):
        global DATA_DIR
        if self.path.split("?")[0] == "/api/set_dir":
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n).decode().strip()
            if os.path.isdir(body):
                DATA_DIR = os.path.abspath(body)
                print(f"[Viewer] Dir -> {DATA_DIR}")
            return self._json({"ok": True, "data_dir": DATA_DIR})
        self.send_error(404)

    def do_HEAD(self):
        """支持 HEAD 请求（浏览器预检）。"""
        path = self.path.split("?")[0]
        m = re.match(r"/img/(left|right)/(.+\.png)$", path)
        if m:
            fp = os.path.join(DATA_DIR, m.group(1), m.group(2))
            if os.path.isfile(fp):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", os.path.getsize(fp))
                self.end_headers()
                return
        self.send_error(404)

    def _html(self):
        page = FRONTEND.replace("__CLASS_JSON__", class_info_js())
        raw = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html;charset=utf-8")
        self.send_header("Content-Length", len(raw))
        self.end_headers()
        self.wfile.write(raw)

    def _json(self, obj):
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json;charset=utf-8")
        self.send_header("Content-Length", len(raw))
        self.end_headers()
        self.wfile.write(raw)

    def _file(self, fp, mime):
        if not os.path.isfile(fp):
            return self.send_error(404, f"Not found: {fp}")
        with open(fp, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", len(data))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 检查 vendor 文件
    for fname in ["three.min.js", "OrbitControls.js"]:
        fp = os.path.join(VENDOR_DIR, fname)
        if not os.path.isfile(fp):
            print(f"[Viewer] ERROR: Missing {fp}")
            print(f"         Please download Three.js r137 files to {VENDOR_DIR}/")
            sys.exit(1)

    port = find_free_port(args.port)
    if port is None:
        print(f"[Viewer] ERROR: No free port found near {args.port}")
        sys.exit(1)
    if port != args.port:
        print(f"[Viewer] Port {args.port} busy, using {port}")

    # 绑定 127.0.0.1（避免 Windows 0.0.0.0 权限问题）
    srv = http.server.HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"

    nf = len(scan_frames(DATA_DIR))
    print(f"[Viewer] {nf} frames in {DATA_DIR}")
    print(f"[Viewer] Open {url}")
    print(f"[Viewer] Ctrl+C to quit")

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[Viewer] Bye.")
        srv.shutdown()


if __name__ == "__main__":
    main()
