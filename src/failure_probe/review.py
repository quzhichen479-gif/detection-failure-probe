"""Loopback-only review server with a zero-build canvas UI."""

from __future__ import annotations

import json
import mimetypes
import secrets
import threading
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from failure_probe.dataset import IMAGE_EXTENSIONS
from failure_probe.errors import RunFormatError, UnsafePathError
from failure_probe.paths import atomic_write_json, resolve_within, validate_run_dir

MAX_NOTE_REQUEST = 64_000
MAX_REVIEW_ASSET_BYTES = 100_000_000
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
ALLOWED_FLAGS = {"reviewed", "confirmed_suspicious", "confirmed_duplicate"}


def serve_review(
    run_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    """Serve the local review UI until interrupted."""
    if host not in LOOPBACK_HOSTS:
        raise ValueError("Review UI may only bind to a loopback host")
    root = validate_run_dir(run_dir)
    manifest = _read_object(root / "manifest.json")
    dataset_root_value = manifest.get("dataset_root")
    if not isinstance(dataset_root_value, str):
        raise RunFormatError("Run manifest has no valid dataset_root")
    dataset_root = Path(dataset_root_value).resolve(strict=True)
    audit = _read_object(root / "audit.json")
    allowed_images = {
        item["image"]
        for item in audit.get("images", [])
        if isinstance(item, dict) and isinstance(item.get("image"), str)
    }
    token = secrets.token_urlsafe(24)
    notes_lock = threading.Lock()

    class ReviewHandler(BaseHTTPRequestHandler):
        server_version = "FailureProbeReview/1"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(_review_html(token))
                return
            if not self._valid_token(parsed.query):
                self.send_error(HTTPStatus.FORBIDDEN, "Invalid review token")
                return
            if parsed.path == "/api/data":
                payload: dict[str, Any] = {
                    "audit": audit,
                    "analysis": _read_optional_object(root / "analysis.json"),
                    "notes": _read_object(root / "reviewer_notes.json"),
                }
                self._send_json(payload)
                return
            if parsed.path.startswith("/asset/"):
                relative = unquote(parsed.path.removeprefix("/asset/"))
                self._send_asset(relative)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/notes":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not self._valid_token(parsed.query) or not self._valid_origin():
                self.send_error(HTTPStatus.FORBIDDEN, "Invalid review request")
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid content length")
                return
            if not 0 < content_length <= MAX_NOTE_REQUEST:
                self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return
            try:
                request = json.loads(self.rfile.read(content_length).decode("utf-8"))
                image = request["image"]
                note = request.get("note", "")
                flags = request.get("flags", [])
            except (UnicodeError, json.JSONDecodeError, KeyError, TypeError):
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON note payload")
                return
            if (
                image not in allowed_images
                or not isinstance(note, str)
                or len(note) > 2000
                or not isinstance(flags, list)
                or any(flag not in ALLOWED_FLAGS for flag in flags)
            ):
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid note fields")
                return
            with notes_lock:
                notes_payload = _read_object(root / "reviewer_notes.json")
                notes = notes_payload.setdefault("notes", {})
                notes[image] = {
                    "note": note,
                    "flags": sorted(set(flags)),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                atomic_write_json(root / "reviewer_notes.json", notes_payload)
            self._send_json({"saved": True})

        def _valid_token(self, query: str) -> bool:
            values = parse_qs(query).get("token", [])
            return len(values) == 1 and secrets.compare_digest(values[0], token)

        def _valid_origin(self) -> bool:
            origin = self.headers.get("Origin")
            if origin is None:
                return True
            parsed = urlparse(origin)
            return parsed.scheme == "http" and parsed.hostname in LOOPBACK_HOSTS

        def _send_asset(self, relative: str) -> None:
            if (
                relative not in allowed_images
                or Path(relative).suffix.lower() not in IMAGE_EXTENSIONS
            ):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                asset = resolve_within(dataset_root, relative, must_exist=True)
            except (OSError, ValueError, UnsafePathError):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not asset.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if asset.stat().st_size > MAX_REVIEW_ASSET_BYTES:
                self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return
            content_type = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
            try:
                payload = asset.read_bytes()
            except OSError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self._security_headers(content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_html(self, markup: str) -> None:
            payload = markup.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self._security_headers("text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, value: Any) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self._security_headers("application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _security_headers(self, content_type: str) -> None:
            self.send_header("Content-Type", content_type)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; style-src 'unsafe-inline'; "
                "script-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'",
            )

        def log_message(self, format_string: str, *args: Any) -> None:
            print(f"review: {format_string % args}")

    server = ThreadingHTTPServer((host, port), ReviewHandler)
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}/?token={quote(token)}"
    print(f"Review UI: {url}")
    print("Data stays local. Press Ctrl+C to stop.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RunFormatError(f"Missing or unsafe run artifact: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunFormatError(f"Invalid run artifact {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RunFormatError(f"Run artifact must contain an object: {path.name}")
    return payload


def _read_optional_object(path: Path) -> dict[str, Any] | None:
    return _read_object(path) if path.is_file() else None


def _review_html(token: str) -> str:
    safe_token = json.dumps(token)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Detection Failure Probe Review</title><style>
:root{{--bg:#090d18;--panel:#131a2a;--panel2:#1a2337;--ink:#edf3ff;--muted:#98a5bf;--line:#2c3853;--cyan:#55d6ff;--red:#ff6078}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:2;background:#090d18ee;backdrop-filter:blur(12px);border-bottom:1px solid var(--line);padding:16px 22px}}
h1{{font-size:19px;margin:0 0 12px}}.filters{{display:flex;gap:10px;flex-wrap:wrap}}label{{color:var(--muted)}}
select,input,textarea,button{{background:var(--panel2);color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:7px}}
main{{max-width:1250px;margin:auto;padding:20px}}#status{{color:var(--muted);margin-bottom:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:16px}}article{{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}}
.meta{{padding:12px}}.title{{font-weight:650;word-break:break-all}}.tags{{color:var(--muted);font-size:12px;margin:5px 0}}
.stage{{position:relative;background:#05070d;min-height:220px;display:grid;place-items:center}}canvas{{display:block;max-width:100%;height:auto}}
textarea{{width:100%;min-height:65px;resize:vertical}}button{{cursor:pointer;color:var(--cyan)}}.flags{{display:flex;gap:12px;flex-wrap:wrap;margin:7px 0}}.flags label{{color:var(--ink)}}
.legend{{font-size:12px;color:var(--muted);margin-top:8px}}.saved{{color:#6ee7a8;margin-left:8px}}@media(max-width:500px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>Detection Failure Probe · Local Review</h1><div class="filters">
<label>Failure <select id="failure"><option value="all">All</option><option value="fp">Any FP</option><option value="fn">FN</option><option value="localization_error">Localization</option><option value="classification_error">Classification</option><option value="duplicate_detection">Duplicate detection</option><option value="suspicious">Suspicious annotation</option></select></label>
<label>Class <select id="class"><option value="all">All classes</option></select></label>
<label>Min confidence <input id="confidence" type="number" min="0" max="1" step="0.05" value="0"></label>
</div></header><main><div id="status">Loading…</div><div id="grid" class="grid"></div></main>
<script>
const TOKEN={safe_token}; let DATA; const colors={{tp:'#42d9c8',fn:'#ff6078',background_false_positive:'#ff8c5a',localization_error:'#ffd166',classification_error:'#eb72ff',duplicate_detection:'#a78bfa',gt:'#68a8ff'}};
const api=(path,options={{}})=>fetch(path+(path.includes('?')?'&':'?')+'token='+encodeURIComponent(TOKEN),options);
const el=(name,className,text)=>{{const node=document.createElement(name);if(className)node.className=className;if(text!==undefined)node.textContent=text;return node}};
function merged(){{const analysis=new Map((DATA.analysis?.images||[]).map(x=>[x.image,x]));return DATA.audit.images.map(a=>({{...a,analysis:analysis.get(a.image)}}))}}
function statuses(item){{return new Set([...(item.analysis?.ground_truth||[]).map(x=>x.status),...(item.analysis?.predictions||[]).map(x=>x.status)])}}
function passes(item){{const failure=document.querySelector('#failure').value,cls=document.querySelector('#class').value,min=Number(document.querySelector('#confidence').value||0),set=statuses(item);if(failure==='fp'&&!['background_false_positive','localization_error','classification_error','duplicate_detection'].some(x=>set.has(x)))return false;if(failure==='fn'&&!set.has('fn'))return false;if(!['all','fp','fn','suspicious'].includes(failure)&&!set.has(failure))return false;if(failure==='suspicious'&&!item.issue_types.some(x=>['tiny_box','extreme_aspect_ratio','duplicate_annotation','near_duplicate_annotation'].includes(x)))return false;const boxes=[...(item.analysis?.ground_truth||item.annotations||[]),...(item.analysis?.predictions||[])];if(cls!=='all'&&!boxes.some(x=>String(x.class_id)===cls))return false;if(!((item.analysis?.predictions||[]).some(x=>x.confidence>=min)||min===0))return false;return true}}
function render(){{const grid=document.querySelector('#grid');grid.replaceChildren();const items=merged().filter(passes);document.querySelector('#status').textContent=`${{items.length}} / ${{DATA.audit.images.length}} images`;items.forEach(item=>grid.append(card(item)))}}
function card(item){{const article=el('article'),stage=el('div','stage'),canvas=el('canvas'),meta=el('div','meta');stage.append(canvas);article.append(stage,meta);meta.append(el('div','title',item.image),el('div','tags',item.issue_types.length?'Audit: '+item.issue_types.join(', '):'No audit flags'));
const noteState=DATA.notes.notes[item.image]||{{note:'',flags:[]}},ta=el('textarea');ta.placeholder='Reviewer note';ta.value=noteState.note;meta.append(ta);const flags=el('div','flags');['reviewed','confirmed_suspicious','confirmed_duplicate'].forEach(name=>{{const label=el('label'),input=document.createElement('input');input.type='checkbox';input.value=name;input.checked=noteState.flags.includes(name);label.append(input,document.createTextNode(' '+name.replaceAll('_',' ')));flags.append(label)}});meta.append(flags);const save=el('button','', 'Save note'),saved=el('span','saved');save.onclick=async()=>{{save.disabled=true;const selected=[...flags.querySelectorAll('input:checked')].map(x=>x.value);const response=await api('/api/notes',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{image:item.image,note:ta.value,flags:selected}})}});saved.textContent=response.ok?'Saved':'Save failed';save.disabled=false}};meta.append(save,saved,el('div','legend','GT blue · TP teal · FN/red · localization yellow · classification magenta · duplicate purple'));draw(canvas,item);return article}}
function draw(canvas,item){{const image=new Image();image.onload=()=>{{const max=600,scale=Math.min(1,max/image.naturalWidth),width=Math.round(image.naturalWidth*scale),height=Math.round(image.naturalHeight*scale),ratio=window.devicePixelRatio||1;canvas.width=width*ratio;canvas.height=height*ratio;canvas.style.width=width+'px';canvas.style.height=height+'px';const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);ctx.drawImage(image,0,0,width,height);const boxScale=width/item.width;const gt=item.analysis?.ground_truth||item.annotations.filter(x=>x.valid).map(x=>({{...x,status:'gt'}}));gt.forEach(x=>box(ctx,x.bbox_xyxy,boxScale,colors[x.status]||colors.gt,'GT '+x.class_id+(x.status!=='gt'?' '+x.status:'')));(item.analysis?.predictions||[]).forEach(x=>box(ctx,x.bbox_xyxy,boxScale,colors[x.status]||colors.background_false_positive,`${{x.status}} c${{x.class_id}} ${{x.confidence.toFixed(2)}}`))}};image.onerror=()=>{{canvas.replaceWith(el('div','meta','Image could not be loaded'))}};image.src='/asset/'+item.image.split('/').map(encodeURIComponent).join('/')+'?token='+encodeURIComponent(TOKEN)}}
function box(ctx,b,s,color,label){{if(!b)return;const [x1,y1,x2,y2]=b.map(v=>v*s);ctx.strokeStyle=color;ctx.lineWidth=2;ctx.strokeRect(x1,y1,x2-x1,y2-y1);ctx.font='11px system-ui';const w=ctx.measureText(label).width+6;ctx.fillStyle=color;ctx.fillRect(x1,Math.max(0,y1-15),w,15);ctx.fillStyle='#05070d';ctx.fillText(label,x1+3,Math.max(11,y1-4))}}
async function init(){{const response=await api('/api/data');if(!response.ok){{document.querySelector('#status').textContent='Could not load run data';return}}DATA=await response.json();const classes=DATA.audit.class_distribution,select=document.querySelector('#class');classes.forEach(x=>{{const option=el('option','',`${{x.class_id}} · ${{x.name}}`);option.value=String(x.class_id);select.append(option)}});['failure','class','confidence'].forEach(id=>document.querySelector('#'+id).addEventListener('input',render));render()}}init();
</script></body></html>"""
