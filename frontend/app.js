const sceneSelect = document.getElementById("scene-select");
const playBtn = document.getElementById("play-btn");
const timeSlider = document.getElementById("time-slider");
const nVariantsInput = document.getElementById("n-variants");
const generateBtn = document.getElementById("generate-btn");
const variantToggles = document.getElementById("variant-toggles");
const canvas = document.getElementById("bev-canvas");
const ctx = canvas.getContext("2d");
const genBadge = document.getElementById("generator-badge");

const VARIANT_COLORS = ["#ff5c8a", "#8a6bff", "#ffd166", "#06d6a0", "#ef476f", "#118ab2", "#c77dff", "#f4a261"];
const CLASS_COLORS = { vehicle: "#4f8cff", bicycle: "#ffb454", pedestrian: "#4fd18b", other: "#999999" };

let currentScene = null;
let currentVariants = []; // [{variant_id, agent_futures}]
let visibleVariants = new Set();
let playing = false;
let playTimer = null;
let transform = null;

let mapBackgroundImg = null;
let mapBackgroundBounds = null;

async function loadMapBackground() {
  try {
    // Cache-Busting: das Hintergrundbild wird von den Datenpipeline-Skripten
    // gelegentlich neu erzeugt (gleicher Dateiname) -- ohne Query-Parameter
    // wuerde der Browser sonst eine veraltete, gecachte Version weiterzeigen.
    const cacheBust = Date.now();
    const meta = await fetch(`/map_background.json?v=${cacheBust}`).then((r) => (r.ok ? r.json() : null));
    if (!meta) return;
    const img = new Image();
    img.onload = () => {
      mapBackgroundImg = img;
      mapBackgroundBounds = meta;
      draw();
    };
    img.src = `/map_background.png?v=${cacheBust}`;
  } catch {
    // Kein Hintergrundbild vorhanden (z.B. Szenen-Pipeline noch nicht gelaufen) -- kein Problem, BEV bleibt leer.
  }
}

async function loadHealth() {
  const res = await fetch("/api/health");
  const data = await res.json();
  genBadge.textContent = `Generator: ${data.generator_mode}`;
}

async function loadSceneList() {
  const res = await fetch("/api/scenes");
  const scenes = await res.json();
  sceneSelect.innerHTML = "";
  for (const s of scenes) {
    const opt = document.createElement("option");
    opt.value = s.scene_id;
    opt.textContent = `${s.scene_id} (${s.n_agents} Agenten: ${s.classes.join(", ")})`;
    sceneSelect.appendChild(opt);
  }
  if (scenes.length > 0) {
    await loadScene(scenes[0].scene_id);
  }
}

async function loadScene(sceneId) {
  stopPlaying();
  const res = await fetch(`/api/scenes/${sceneId}`);
  currentScene = await res.json();
  currentVariants = [];
  visibleVariants = new Set();
  renderVariantToggles();

  const totalFrames = maxHistoryLen() + currentScene.horizon_steps;
  timeSlider.min = 0;
  timeSlider.max = Math.max(totalFrames - 1, 1);
  timeSlider.value = 0;

  computeTransform();
  draw();
}

function maxHistoryLen() {
  return Math.max(...currentScene.agents.map((a) => a.history.length), 1);
}

function computeTransform() {
  const b = currentScene.map_bounds;
  const w = canvas.width;
  const h = canvas.height;
  const pad = 30;
  const availW = w - 2 * pad;
  const availH = h - 2 * pad;
  const worldW = Math.max(b.xmax - b.xmin, 1);
  const worldH = Math.max(b.ymax - b.ymin, 1);
  const scale = Math.min(availW / worldW, availH / worldH);

  // Restraum (weil Szenen-Seitenverhaeltnis selten exakt zur Canvas passt) gleichmaessig
  // auf beide Seiten verteilen, statt links/unten anzupinnen -- sonst wirkt die Szene
  // nicht zentriert.
  const offsetX = pad + (availW - worldW * scale) / 2;
  const offsetY = pad + (availH - worldH * scale) / 2;

  transform = (x, y) => {
    const cx = offsetX + (x - b.xmin) * scale;
    const cy = h - offsetY - (y - b.ymin) * scale; // y-Achse spiegeln (Northing nach oben)
    return [cx, cy];
  };
}

function agentPathAt(agent, frame) {
  // Kombiniert history + future (real) zu einer Zeitreihe, gibt Punkt bei frame zurueck.
  const combined = agent.history.concat(agent.future);
  if (frame < 0 || frame >= combined.length) return null;
  return combined[frame];
}

function draw() {
  if (!currentScene) return;
  const frame = parseInt(timeSlider.value, 10);
  const histLen = maxHistoryLen();

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawMapBackground();

  for (const agent of currentScene.agents) {
    drawRealTrail(agent, frame);
    for (const variantId of visibleVariants) {
      drawVariantTrail(agent, variantId, frame, histLen);
    }
    drawAgentMarker(agent, frame);
  }
}

function drawMapBackground() {
  // Rasterkarte aus der Dichte echter Trajektorien (siehe scripts/build_map_background.py):
  // zeigt tatsaechlich befahrene Fahrspuren/Rad-/Fusswege statt einer groben Silhouette.
  // Deckt den GESAMTEN Kreuzungsbereich ab; hier wird nur der fuer die aktuelle Szene
  // relevante Ausschnitt eingeblendet, exakt im selben Massstab wie die Agentenpunkte.
  if (mapBackgroundImg && mapBackgroundBounds) {
    const b = mapBackgroundBounds;
    const [x0, y0] = transform(b.xmin, b.ymax); // oben links (Norden)
    const [x1, y1] = transform(b.xmax, b.ymin); // unten rechts (Sueden)
    ctx.drawImage(mapBackgroundImg, x0, y0, x1 - x0, y1 - y0);
    return;
  }

  // Fallback, falls noch keine Rasterkarte generiert wurde (z.B. Pipeline noch nicht gelaufen):
  // schwache Konvexe-Huellen-Polygone, sofern die Szene welche mitliefert.
  ctx.fillStyle = "rgba(79, 140, 255, 0.06)";
  ctx.strokeStyle = "rgba(79, 140, 255, 0.25)";
  ctx.lineWidth = 1;
  for (const poly of currentScene.lane_polygons || []) {
    if (poly.length < 3) continue;
    ctx.beginPath();
    poly.forEach(([x, y], i) => {
      const [cx, cy] = transform(x, y);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    });
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
}

function drawRealTrail(agent, frame) {
  const combined = agent.history.concat(agent.future);
  const upto = Math.min(frame + 1, combined.length);
  if (upto < 2) return;
  ctx.strokeStyle = "rgba(231, 235, 243, 0.55)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < upto; i++) {
    const [cx, cy] = transform(combined[i].x, combined[i].y);
    if (i === 0) ctx.moveTo(cx, cy);
    else ctx.lineTo(cx, cy);
  }
  ctx.stroke();
}

function drawVariantTrail(agent, variantId, frame, histLen) {
  const variant = currentVariants.find((v) => v.variant_id === variantId);
  if (!variant) return;
  const future = variant.agent_futures[agent.agent_id];
  if (!future) return;

  const futureFrame = frame - histLen;
  if (futureFrame < 0) return;
  const upto = Math.min(futureFrame + 1, future.length);
  if (upto < 1) return;

  const color = variantColor(variantId);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  const start = transform(agent.history[agent.history.length - 1].x, agent.history[agent.history.length - 1].y);
  ctx.moveTo(start[0], start[1]);
  for (let i = 0; i < upto; i++) {
    const [cx, cy] = transform(future[i].x, future[i].y);
    ctx.lineTo(cx, cy);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  if (upto > 0) {
    const [px, py] = transform(future[upto - 1].x, future[upto - 1].y);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawAgentMarker(agent, frame) {
  const point = agentPathAt(agent, frame) || agent.history[agent.history.length - 1];
  if (!point) return;
  const [cx, cy] = transform(point.x, point.y);
  const color = CLASS_COLORS[agent.agent_class] || CLASS_COLORS.other;

  ctx.fillStyle = color;
  ctx.strokeStyle = "rgba(255,255,255,0.8)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  if (agent.agent_class === "vehicle") {
    ctx.rect(cx - 5, cy - 5, 10, 10);
  } else {
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
  }
  ctx.fill();
  ctx.stroke();
}

function variantColor(variantId) {
  const idx = currentVariants.findIndex((v) => v.variant_id === variantId);
  return VARIANT_COLORS[idx % VARIANT_COLORS.length];
}

function renderVariantToggles() {
  variantToggles.innerHTML = "";
  currentVariants.forEach((v, i) => {
    const id = `toggle-${v.variant_id}`;
    const label = document.createElement("label");
    label.style.display = "inline-flex";
    label.style.alignItems = "center";
    label.style.gap = "4px";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = id;
    checkbox.checked = visibleVariants.has(v.variant_id);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) visibleVariants.add(v.variant_id);
      else visibleVariants.delete(v.variant_id);
      draw();
    });

    const swatch = document.createElement("span");
    swatch.style.display = "inline-block";
    swatch.style.width = "10px";
    swatch.style.height = "10px";
    swatch.style.borderRadius = "50%";
    swatch.style.background = VARIANT_COLORS[i % VARIANT_COLORS.length];

    label.appendChild(checkbox);
    label.appendChild(swatch);
    label.appendChild(document.createTextNode(`Variante ${i + 1}`));
    variantToggles.appendChild(label);
  });
}

async function generateVariants() {
  if (!currentScene) return;
  generateBtn.disabled = true;
  generateBtn.textContent = "… generiere";
  try {
    const n = parseInt(nVariantsInput.value, 10) || 3;
    const res = await fetch(`/api/scenes/${currentScene.scene_id}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ n_variants: n }),
    });
    const data = await res.json();
    currentVariants = data.variants;
    visibleVariants = new Set(currentVariants.map((v) => v.variant_id));
    renderVariantToggles();
    draw();
  } finally {
    generateBtn.disabled = false;
    generateBtn.textContent = "✨ Generiere Varianten";
  }
}

function stopPlaying() {
  playing = false;
  playBtn.textContent = "▶ Play";
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
  }
}

function togglePlay() {
  if (playing) {
    stopPlaying();
    return;
  }
  playing = true;
  playBtn.textContent = "⏸ Pause";
  playTimer = setInterval(() => {
    let v = parseInt(timeSlider.value, 10) + 1;
    if (v > parseInt(timeSlider.max, 10)) v = 0;
    timeSlider.value = v;
    draw();
  }, 90);
}

sceneSelect.addEventListener("change", (e) => loadScene(e.target.value));
timeSlider.addEventListener("input", draw);
playBtn.addEventListener("click", togglePlay);
generateBtn.addEventListener("click", generateVariants);

loadHealth();
loadSceneList();
loadMapBackground();
