// FaceScan browser UI. Plain DOM; all server text is inserted with textContent.
import { Viewer } from "/static/viewer.js";

const $ = (id) => document.getElementById(id);
const DEFAULTS = {
  voxel_size_mm: 0.7, back_cut_mm: -40, bottom_cut_mm: -110, smoothing_iterations: 4,
  target_faces: 600000, scale_percent: 100, target_height_mm: "",
};
const STATE_LABEL = {
  waiting_upload: "Waiting for upload", queued: "Queued", processing: "Processing", done: "Ready", failed: "Failed",
};

let scans = [];
let selectedId = null;
let viewer = null;
let loadedPreviewKey = null;
let pollTimer = null;
let waitingScanId = null;

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const type = res.headers.get("content-type") || "";
  const body = type.includes("json") ? await res.json() : await res.text();
  if (!res.ok) throw new Error((body && body.detail) || res.statusText);
  return body;
}

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), 4000);
}

function el(tag, props = {}, ...children) {
  const e = document.createElement(tag);
  Object.assign(e, props);
  for (const c of children) e.append(c);
  return e;
}

const fmtSize = (b) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.round(b / 1e3)} kB`);
const fmtDate = (t) => new Date(t * 1000).toLocaleString();

// ----------------------------------------------------------------- list
function renderList() {
  const ul = $("scan-list");
  ul.replaceChildren(
    ...scans.map((s) => {
      const li = el("li", { className: s.id === selectedId ? "active" : "" },
        el("div", { className: "title", textContent: s.name }),
        el("div", { className: "sub", textContent: `${STATE_LABEL[s.state] || s.state} · ${fmtDate(s.created_at)}` }));
      li.onclick = () => select(s.id);
      return li;
    }),
  );
  $("empty-list").hidden = scans.length > 0;
}

async function refresh() {
  try {
    scans = await api("/api/scans");
  } catch (e) {
    toast(`Cannot reach the server: ${e.message}`);
    return;
  }
  renderList();
  if (selectedId) {
    const s = scans.find((x) => x.id === selectedId);
    if (s) renderDetail(s);
    else { selectedId = null; $("detail").hidden = true; }
  }
  if (waitingScanId) {
    const s = scans.find((x) => x.id === waitingScanId);
    if (s && s.state !== "waiting_upload") {
      waitingScanId = null;
      $("new-dialog").close();
      select(s.id);
      toast("Scan received. Processing has started.");
    }
  }
  schedulePoll();
}

function schedulePoll() {
  clearTimeout(pollTimer);
  const busy = waitingScanId || scans.some((s) => s.state === "queued" || s.state === "processing");
  pollTimer = setTimeout(refresh, busy ? 1500 : 8000);
}

// --------------------------------------------------------------- detail
function select(id) {
  selectedId = id;
  renderList();
  const s = scans.find((x) => x.id === id);
  if (s) {
    fillParams(s.params || DEFAULTS);
    renderDetail(s);
  }
}

function renderDetail(s) {
  $("detail").hidden = false;
  $("d-name").textContent = s.name;
  const badge = $("d-state");
  badge.textContent = STATE_LABEL[s.state] || s.state;
  badge.className = `badge ${s.state}`;

  const busy = s.state === "queued" || s.state === "processing";
  $("d-progress").hidden = !busy;
  $("d-bar").style.width = `${Math.round((s.progress || 0) * 100)}%`;
  $("d-stage").textContent = busy ? `${s.stage || "Waiting"} · ${Math.round((s.progress || 0) * 100)}%` : "";
  $("d-error").hidden = !s.error;
  $("d-error").textContent = s.error || "";
  $("d-reprocess").disabled = busy || !s.has_upload;
  $("d-delete").disabled = s.state === "processing";
  $("d-capture").hidden = !s.has_upload;
  $("d-capture").href = `/api/scans/${s.id}/files/capture.zip`;

  const r = s.report;
  const dl = $("d-report");
  dl.replaceChildren();
  $("d-warnings").replaceChildren();
  if (r && s.state === "done") {
    const m = r.mesh;
    const rows = [
      ["Size", `${m.dimensions_mm.join(" × ")} mm`],
      ["Scale", `${m.scale_percent_of_life_size}% of life size`],
      ["Volume", `${m.volume_cm3} cm³`],
      ["Printable", m.watertight ? "Yes (watertight)" : "Needs repair"],
      ["Triangles", m.faces.toLocaleString()],
      ["Frames used", `${r.frames_used} of ${r.frames_total}`],
      ["Alignment error", r.registration_rmse_mm != null ? `${r.registration_rmse_mm} mm RMS` : "n/a"],
      ["Processing time", `${Math.round(r.timings.total_s)} s`],
    ];
    for (const [k, v] of rows) dl.append(el("dt", { textContent: k }), el("dd", { textContent: v }));
    $("d-warnings").replaceChildren(...r.warnings.map((w) => el("li", { textContent: w })));
  } else {
    dl.append(el("dd", { className: "muted", textContent: busy ? "Processing…" : "No model yet." }));
  }

  $("d-files").replaceChildren(
    ...(s.files || []).map((f) =>
      el("li", {},
        el("a", { href: `/api/scans/${s.id}/files/${f.name}`, textContent: f.label, className: "link" }),
        el("span", { className: "size", textContent: fmtSize(f.size) }))),
  );

  const key = s.has_preview ? `${s.id}:${r && r.timings && r.timings.total_s}:${s.report && s.report.mesh.faces}` : null;
  $("viewer-empty").hidden = !!s.has_preview;
  if (key !== loadedPreviewKey) {
    loadedPreviewKey = key;
    if (!viewer) viewer = new Viewer($("viewer"));
    viewer.clear();
    if (key) {
      viewer.load(`/api/scans/${s.id}/files/preview.ply?k=${encodeURIComponent(key)}`)
        .then(() => viewer.setColor($("v-color").checked))
        .catch((e) => toast(`Could not load preview: ${e.message}`));
    }
  }
}

function fillParams(p) {
  const form = $("d-params");
  for (const [k, v] of Object.entries({ ...DEFAULTS, ...p })) {
    if (form.elements[k]) form.elements[k].value = v ?? "";
  }
}

function readParams() {
  const out = {};
  const form = $("d-params");
  for (const k of Object.keys(DEFAULTS)) {
    const v = form.elements[k].value.trim();
    out[k] = v === "" ? null : Number(v);
  }
  return out;
}

// --------------------------------------------------------------- actions
$("d-params").onsubmit = async (ev) => {
  ev.preventDefault();
  try {
    await api(`/api/scans/${selectedId}/process`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(readParams()),
    });
    refresh();
  } catch (e) {
    toast(e.message);
  }
};
$("d-defaults").onclick = () => fillParams(DEFAULTS);

$("d-delete").onclick = async () => {
  const s = scans.find((x) => x.id === selectedId);
  if (!s || !confirm(`Delete "${s.name}" and all of its files?`)) return;
  try {
    await api(`/api/scans/${s.id}`, { method: "DELETE" });
    selectedId = null;
    loadedPreviewKey = null;
    viewer?.clear();
    $("detail").hidden = true;
    refresh();
  } catch (e) {
    toast(e.message);
  }
};

$("d-rename").onclick = async () => {
  const s = scans.find((x) => x.id === selectedId);
  const name = s && prompt("Scan name", s.name);
  if (!name) return;
  try {
    await api(`/api/scans/${s.id}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }),
    });
    refresh();
  } catch (e) {
    toast(e.message);
  }
};

$("new-scan").onclick = async () => {
  try {
    const res = await api("/api/scans", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    // The SVG is generated by the server's QR library from the scan URL.
    $("qr").innerHTML = res.qr_svg;
    $("phone-url").textContent = res.phone_url;
    $("lan-note").hidden = !res.is_lan_url;
    waitingScanId = res.scan.id;
    $("new-dialog").showModal();
    refresh();
  } catch (e) {
    toast(e.message);
  }
};
$("new-close").onclick = () => $("new-dialog").close();
$("new-dialog").addEventListener("close", async () => {
  // Closing before the phone uploads cancels that scan.
  const id = waitingScanId;
  waitingScanId = null;
  const s = scans.find((x) => x.id === id);
  if (id && (!s || s.state === "waiting_upload")) {
    try {
      const fresh = await api(`/api/scans/${id}`);
      if (fresh.state === "waiting_upload") await api(`/api/scans/${id}`, { method: "DELETE" });
    } catch { /* already gone */ }
    refresh();
  }
});

$("import-file").onchange = async (ev) => {
  const file = ev.target.files[0];
  ev.target.value = "";
  if (!file) return;
  toast(`Uploading ${file.name}…`);
  try {
    const res = await api("/api/scans/import", { method: "POST", headers: { "Content-Type": "application/zip" }, body: file });
    await refresh();
    select(res.scan_id);
  } catch (e) {
    toast(`Import failed: ${e.message}`);
  }
};

$("v-color").onchange = (e) => viewer?.setColor(e.target.checked);
$("v-wire").onchange = (e) => viewer?.setWireframe(e.target.checked);
$("v-reset").onclick = () => viewer?.resetView();

refresh();
