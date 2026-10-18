const navButtons = [...document.querySelectorAll(".nav-link")];
const pages = [...document.querySelectorAll(".page")];
const topbar = document.querySelector(".topbar");
const form = document.querySelector("#analyze-form");
const videoInput = document.querySelector("#video-input");
const backgroundInput = document.querySelector("#background-input");
const confidenceInput = document.querySelector("#confidence");
const confidenceValue = document.querySelector("#confidence-value");
const enhancementInput = document.querySelector("#enhancement");
const imgszInput = document.querySelector("#imgsz");
const speciesInput = document.querySelector("#species-input");
const runButton = document.querySelector("#run-button");
const progressCard = document.querySelector("#progress-card");
const errorBanner = document.querySelector("#error-banner");
let currentJobId = new URLSearchParams(location.search).get("job") || localStorage.getItem("seal-job-id");
let pollTimer = null;
let latestStatistics = null;
let speciesTableRows = [];
let speciesReviewLabels = {};
let speciesReviewJobId = null;
let speciesRangePayload = null;
let rangeMapMode = "local";
const speciesTablePages = { results: 0, species: 0, "species-labelled": 0, "species-unlabelled": 0 };

const updateTopbarShape = () => topbar?.classList.toggle("topbar-morphed", window.scrollY > 72);
window.addEventListener("scroll", updateTopbarShape, { passive: true });
updateTopbarShape();

function navigateTo(page) {
  const valid = ["Home", "Upload", "Live", "Results", "Statistics", "Species"];
  if (!valid.includes(page)) page = "Home";
  navButtons.forEach((button) => button.classList.toggle("active", button.dataset.page === page));
  pages.forEach((section) => section.classList.toggle("active", section.id === `page-${page}`));
  document.querySelector(".depth-rail")?.classList.toggle("visible", page === "Home" && Boolean(document.querySelector(".ocean-descent")?.classList.contains("descent-active")));
  history.replaceState(null, "", `?page=${page}`);
  window.scrollTo({ top: 0, behavior: "smooth" });
  updateTopbarShape();
  if (page === "Statistics" && latestStatistics) renderStatisticsCharts();
}

document.querySelectorAll("[data-page]").forEach((button) => button.addEventListener("click", () => navigateTo(button.dataset.page)));
document.querySelectorAll(".home-upload, .go-upload").forEach((button) => button.addEventListener("click", () => navigateTo("Upload")));
document.querySelector(".explore-species")?.addEventListener("click", () => document.querySelector("#zone-sunlight")?.scrollIntoView({ behavior: "smooth" }));
document.querySelector(".open-species")?.addEventListener("click", () => navigateTo("Species"));
const initialPage = new URLSearchParams(location.search).get("page") || "Home";
navigateTo(initialPage);

const depthRail = document.querySelector(".depth-rail");
const oceanDescent = document.querySelector(".ocean-descent");
if (oceanDescent && "IntersectionObserver" in window) {
  const zones = [...document.querySelectorAll(".ocean-layer")];
  const railDots = [...document.querySelectorAll(".depth-dot")];
  // Build a light particle field once; scrolling only updates transforms, never inference.
  zones.forEach((zone, zoneIndex) => {
    const field = document.createElement("div");
    field.className = "current-field";
    field.setAttribute("aria-hidden", "true");
    for (let index = 0; index < 18; index += 1) {
      const particle = document.createElement("i");
      particle.className = "water-particle";
      particle.style.setProperty("--x", `${(index * 47 + zoneIndex * 13) % 100}%`);
      particle.style.setProperty("--delay", `${-((index * 1.7) % 13)}s`);
      particle.style.setProperty("--duration", `${12 + (index % 7) * 2}s`);
      particle.style.setProperty("--drift", `${(index % 2 ? 1 : -1) * (12 + (index % 5) * 7)}px`);
      field.append(particle);
    }
    zone.prepend(field);
  });

  let motionFrame = 0;
  const updateOceanMotion = () => {
    motionFrame = 0;
    const viewportCenter = window.innerHeight * 0.5;
    zones.forEach((zone) => {
      const rect = zone.getBoundingClientRect();
      const centerDelta = (rect.top + rect.height * 0.5 - viewportCenter) / Math.max(window.innerHeight, 1);
      zone.style.setProperty("--parallax", `${Math.max(-90, Math.min(90, centerDelta * -72))}px`);
      zone.style.setProperty("--scene-progress", `${Math.max(-1, Math.min(1, centerDelta))}`);
    });
  };
  const scheduleOceanMotion = () => {
    if (!motionFrame) motionFrame = requestAnimationFrame(updateOceanMotion);
  };
  window.addEventListener("scroll", scheduleOceanMotion, { passive: true });
  window.addEventListener("resize", scheduleOceanMotion, { passive: true });
  scheduleOceanMotion();

  const descentObserver = new IntersectionObserver(([entry]) => {
    oceanDescent.classList.toggle("descent-active", entry.isIntersecting);
    depthRail?.classList.toggle("visible", entry.isIntersecting && document.querySelector("#page-Home")?.classList.contains("active"));
  }, { threshold: 0.02 });
  descentObserver.observe(oceanDescent);

  const zoneObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("in-view");
      const activeIndex = zones.indexOf(entry.target);
      railDots.forEach((dot, index) => dot.classList.toggle("active", index === activeIndex));
      const depthLabel = document.querySelector("#depth-rail-label");
      if (depthLabel) depthLabel.textContent = entry.target.dataset.depth || "";
    });
  }, { rootMargin: "-38% 0px -42% 0px", threshold: 0 });
  zones.forEach((zone) => zoneObserver.observe(zone));
}

videoInput.addEventListener("change", () => {
  document.querySelector("#video-filename").textContent = videoInput.files[0]?.name || "Choose a video to upload";
});

// Give the video drop zone a clear response while a file is dragged over it.
const videoDropzone = document.querySelector("label.dropzone");
["dragenter", "dragover"].forEach((eventName) => videoDropzone.addEventListener(eventName, (event) => {
  event.preventDefault();
  videoDropzone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((eventName) => videoDropzone.addEventListener(eventName, () => {
  videoDropzone.classList.remove("dragging");
}));
backgroundInput.addEventListener("change", () => {
  document.querySelector("#background-filename").textContent = backgroundInput.files[0]?.name || "Counts detector outputs on an assumed empty-water clip.";
});
confidenceInput.addEventListener("input", () => { confidenceValue.value = Number(confidenceInput.value).toFixed(2); });

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.classList.remove("hidden");
  progressCard.classList.add("hidden");
  runButton.disabled = false;
  runButton.innerHTML = 'Run detection <span aria-hidden="true">→</span>';
}

function showProgress(job) {
  progressCard.classList.remove("hidden");
  const message = job.message || "";
  const isClassifying = job.state === "classifying" || /classif/i.test(message);
  const isDetecting = job.state === "processing" || /detecting and tracking/i.test(message);
  document.querySelector("#progress-title").textContent = isClassifying ? "Identifying species" : isDetecting ? "Detecting and tracking" : "Loading detector";
  document.querySelector("#progress-message").textContent = job.message || "Processing video…";
  const percent = Math.round((job.progress || 0) * 100);
  document.querySelector("#progress-fill").style.width = `${percent}%`;
  document.querySelector("#progress-percent").textContent = job.total_frames ? `${Number(job.processed_frames).toLocaleString()} / ${Number(job.total_frames).toLocaleString()} ${isClassifying ? "tracks" : "frames"}` : `${percent}%`;
  const livePreview = document.querySelector("#live-preview");
  const previewWrap = document.querySelector("#live-preview-wrap");
  if (job.preview_url) {
    previewWrap.classList.remove("hidden");
    const nextPreviewUrl = `${job.preview_url}?frame=${job.preview_frame || Date.now()}`;
    if (livePreview.dataset.frame !== String(job.preview_frame)) {
      livePreview.src = nextPreviewUrl;
      livePreview.dataset.frame = String(job.preview_frame);
    }
    document.querySelector("#live-preview-frame").textContent = `FRAME ${Number(job.preview_frame || 0).toLocaleString()}`;
  }
}

function parseCsvRows(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const split = (line) => {
    const values = [];
    let value = "", quoted = false;
    for (let i = 0; i < line.length; i += 1) {
      const char = line[i];
      if (char === '"' && quoted && line[i + 1] === '"') { value += '"'; i += 1; }
      else if (char === '"') quoted = !quoted;
      else if (char === "," && !quoted) { values.push(value); value = ""; }
      else value += char;
    }
    values.push(value);
    return values;
  };
  const header = split(lines[0]);
  return lines.slice(1).filter(Boolean).map((line) => {
    const values = split(line);
    return Object.fromEntries(header.map((key, index) => [key, values[index] ?? ""]));
  }).filter((row) => Number.isFinite(Number(row.frame)) && Number.isFinite(Number(row.confidence)));
}

function setStatistics(jobId, job, rows) {
  latestStatistics = { jobId, job, rows };
  const empty = document.querySelector("#empty-statistics");
  const content = document.querySelector("#statistics-content");
  empty.classList.toggle("hidden", true);
  content.classList.remove("hidden");
  const metrics = job.metrics || {};
  const avgCount = metrics.mean_detections_per_frame ?? 0;
  const meanConfidence = metrics.mean_confidence;
  document.querySelector(".statistics-report-link").href = `/api/jobs/${jobId}/report.pdf`;
  const values = [
    ["Frames", Number(metrics.frames_processed || 0).toLocaleString()],
    ["Unique track IDs", Number(metrics.unique_track_ids || 0).toLocaleString()],
    ["Detection rows", Number(metrics.detection_rows || rows.length).toLocaleString()],
    ["Mean fish detections / frame", Number(avgCount).toFixed(2)],
    ["Mean confidence", meanConfidence == null ? "—" : `${(Number(meanConfidence) * 100).toFixed(1)}%`],
  ];
  document.querySelector("#statistics-metrics").innerHTML = values.map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`).join("");
  document.querySelector("#statistics-title").textContent = `${metrics.resolution || "Video"} · ${Number(metrics.elapsed_seconds || 0).toFixed(1)} s processing`;
  if (document.querySelector("#page-Statistics").classList.contains("active")) renderStatisticsCharts();
}

function renderSpeciesTable(containerId, pageKey) {
  const container = document.getElementById(containerId);
  if (!container) return;
  if (pageKey === "species") {
    const labelled = speciesTableRows.filter(([, decision]) => !decision.unlabelled);
    const unlabelled = speciesTableRows.filter(([, decision]) => decision.unlabelled);
    const reviewJobId = speciesReviewJobId || latestStatistics?.jobId || currentJobId || localStorage.getItem("seal-job-id");
    const group = (rows, key, title, includeCrops = false) => {
      const pageSize = 10;
      const pageCount = Math.max(1, Math.ceil(rows.length / pageSize));
      const page = Math.max(0, Math.min(speciesTablePages[key] || 0, pageCount - 1));
      speciesTablePages[key] = page;
      const visible = rows.slice(page * pageSize, (page + 1) * pageSize);
      const pager = `<div class="species-pager"><span>Showing ${rows.length ? page * pageSize + 1 : 0}–${Math.min((page + 1) * pageSize, rows.length)} of ${rows.length}</span><div><button class="button secondary" type="button" data-species-target="${key}" data-species-page="${page - 1}" ${page === 0 ? "disabled" : ""}>Previous</button><span>Batch ${page + 1} / ${pageCount}</span><button class="button secondary" type="button" data-species-target="${key}" data-species-page="${page + 1}" ${page + 1 >= pageCount ? "disabled" : ""}>Next 10</button></div></div>`;
      const table = `<div class="species-table-wrap"><table class="species-table"><thead><tr><th>Track ID</th><th>${includeCrops ? "Review crop" : "Species label"}</th><th>Confidence</th><th>Agreement</th><th>Status</th></tr></thead><tbody>${visible.map(([trackId, decision]) => {
        const cropUrl = `/api/jobs/${encodeURIComponent(reviewJobId || "")}/species-crops/${encodeURIComponent(trackId)}`;
        const saved = speciesReviewLabels[String(trackId)];
        const label = includeCrops ? `<div class="species-crop-review"><a class="species-crop-link" href="${cropUrl}" target="_blank" rel="noopener"><img class="species-review-crop" src="${cropUrl}" loading="lazy" alt="Review crop for unlabelled fish, track ${escapeHTML(trackId)}"><span>Open crop</span></a><small>Top candidate: ${escapeHTML(decision.candidate || "Unavailable")}</small></div><form class="species-label-form" data-review-track="${escapeHTML(trackId)}"><label><span>Review label</span><select name="label_type" aria-label="Review type for track ${escapeHTML(trackId)}"><option value="species" ${saved?.label_type !== "false_positive" ? "selected" : ""}>Species name</option><option value="false_positive" ${saved?.label_type === "false_positive" ? "selected" : ""}>False positive</option></select></label><label class="species-review-name ${saved?.label_type === "false_positive" ? "hidden" : ""}"><span>Species</span><input name="label" type="text" maxlength="100" placeholder="e.g. Lutjanus fulviflamma" value="${saved?.label_type !== "false_positive" ? escapeHTML(saved?.label || "") : ""}" ${saved?.label_type === "false_positive" ? "disabled" : "required"}></label><button class="button secondary" type="submit">${saved ? "Update label" : "Save label"}</button><small class="species-label-status" role="status">${saved ? `Saved: ${escapeHTML(saved.label)} · included in training export` : "Saved for later review; this does not change model predictions."}</small></form>` : escapeHTML(decision.label);
        const status = includeCrops ? saved ? "Reviewed · saved" : "Unlabelled · review" : "Labelled candidate";
        return `<tr><td>${escapeHTML(trackId)}</td><td>${label}</td><td>${(Number(decision.score) * 100).toFixed(0)}%</td><td>${(Number(decision.agreement) * 100).toFixed(0)}%</td><td>${status}</td></tr>`;
      }).join("") || `<tr><td colspan="5">${includeCrops ? "No unlabelled tracks in this clip." : "No labelled tracks in this clip."}</td></tr>`}</tbody></table></div>`;
      return `<section class="species-review-group"><div class="species-review-heading"><h2>${title}</h2><span>${rows.length} tracks</span></div>${includeCrops ? `<div class="species-review-tools"><p class="disclaimer">Add a species name or mark a false positive. Labels and crops are saved with this run for later dataset review; they do not change this model’s predictions.</p><a class="button secondary ${Object.keys(speciesReviewLabels).length ? "" : "hidden"}" href="/api/jobs/${encodeURIComponent(reviewJobId || "")}/species-review-dataset.zip" download>Download reviewed crops + labels</a></div>` : ""}${rows.length > pageSize ? pager : ""}${table}${rows.length > pageSize ? pager : ""}</section>`;
    };
    container.innerHTML = group(labelled, "species-labelled", "Labelled candidates") + group(unlabelled, "species-unlabelled", "Unlabelled detections", true);
    return;
  }
  const pageSize = 10;
  const pageCount = Math.max(1, Math.ceil(speciesTableRows.length / pageSize));
  const page = Math.max(0, Math.min(speciesTablePages[pageKey] || 0, pageCount - 1));
  speciesTablePages[pageKey] = page;
  const visibleRows = speciesTableRows.slice(page * pageSize, (page + 1) * pageSize);
  const table = `<div class="species-pager"><span>Showing ${speciesTableRows.length ? page * pageSize + 1 : 0}–${Math.min((page + 1) * pageSize, speciesTableRows.length)} of ${speciesTableRows.length} track candidates</span><div><button class="button secondary" type="button" data-species-target="${pageKey}" data-species-page="${page - 1}" ${page === 0 ? "disabled" : ""}>Previous</button><span>Batch ${page + 1} / ${pageCount}</span><button class="button secondary" type="button" data-species-target="${pageKey}" data-species-page="${page + 1}" ${page + 1 >= pageCount ? "disabled" : ""}>Next 10</button></div></div><div class="species-table-wrap"><table class="species-table"><thead><tr><th>Track ID</th><th>One label for track</th><th>Confidence</th><th>Agreement</th><th>Status</th></tr></thead><tbody>${visibleRows.map(([trackId, decision]) => `<tr><td>${escapeHTML(trackId)}</td><td>${escapeHTML(decision.label)}</td><td>${(Number(decision.score) * 100).toFixed(0)}%</td><td>${(Number(decision.agreement) * 100).toFixed(0)}%</td><td>${decision.unlabelled ? "Flagged" : "Confirmed"}</td></tr>`).join("") || '<tr><td colspan="5">No classifiable tracks in this clip.</td></tr>'}</tbody></table></div>`;
  container.innerHTML = table;
}

document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-species-target][data-species-page]");
  if (!button || button.disabled) return;
  speciesTablePages[button.dataset.speciesTarget] = Number(button.dataset.speciesPage);
  const isResults = button.dataset.speciesTarget === "results";
  renderSpeciesTable(isResults ? "species-table" : "species-page-table", isResults ? "results" : "species");
});

document.addEventListener("change", (event) => {
  const select = event.target.closest(".species-label-form select[name=label_type]");
  if (!select) return;
  const input = select.form.elements.label;
  const nameField = select.form.querySelector(".species-review-name");
  const isFalsePositive = select.value === "false_positive";
  input.disabled = isFalsePositive;
  input.required = !isFalsePositive;
  nameField.classList.toggle("hidden", isFalsePositive);
});

document.addEventListener("submit", async (event) => {
  const reviewForm = event.target.closest(".species-label-form[data-review-track]");
  if (!reviewForm) return;
  event.preventDefault();
  const jobId = speciesReviewJobId || latestStatistics?.jobId || currentJobId || localStorage.getItem("seal-job-id");
  const status = reviewForm.querySelector(".species-label-status");
  const button = reviewForm.querySelector("button[type=submit]");
  if (!jobId) { status.textContent = "No analysis run is selected."; return; }
  button.disabled = true;
  status.textContent = "Saving label…";
  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/species-review-labels`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ track_id: reviewForm.dataset.reviewTrack, label_type: reviewForm.elements.label_type.value, label: reviewForm.elements.label.value }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Could not save this label.");
    speciesReviewLabels[String(result.entry.track_id)] = result.entry;
    renderSpeciesTable("species-table", "results");
    renderSpeciesTable("species-page-table", "species");
  } catch (error) {
    status.textContent = error.message || "Could not save this label.";
    button.disabled = false;
  }
});

async function loadSpeciesReviewLabels(jobId) {
  speciesReviewJobId = jobId;
  speciesReviewLabels = {};
  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/species-review-labels`);
    if (!response.ok) return;
    const result = await response.json();
    if (speciesReviewJobId !== jobId) return;
    speciesReviewLabels = result.labels || {};
    renderSpeciesTable("species-table", "results");
    renderSpeciesTable("species-page-table", "species");
  } catch { /* Keep the review UI usable if saved annotations cannot be loaded. */ }
}

function greatCircleKm(a, b) {
  const rad = (degrees) => degrees * Math.PI / 180;
  const dLat = rad(b.lat - a.lat), dLon = rad(b.lon - a.lon);
  const value = Math.sin(dLat / 2) ** 2 + Math.cos(rad(a.lat)) * Math.cos(rad(b.lat)) * Math.sin(dLon / 2) ** 2;
  return 6371.0088 * 2 * Math.atan2(Math.sqrt(value), Math.sqrt(1 - value));
}

function renderRangeSpecies() {
  const select = document.querySelector("#range-species-select");
  const entry = speciesRangePayload?.species?.find((item) => item.species === select?.value);
  if (!entry) return;
  const { longitude, latitude, radius_km: radius } = speciesRangePayload;
  const points = entry.points || [];
  const fishbaseName = String(entry.species || "").trim().replace(/\s+/g, "_");
  document.querySelector("#regional-status-title").textContent = `Native / migratory status: unverified for ${entry.species}`;
  document.querySelector("#regional-status-source").href = `https://www.fishbase.se/summary/${encodeURIComponent(fishbaseName)}.html`;
  const nearby = points.filter((point) => greatCircleKm({ lon: longitude, lat: latitude }, point) <= radius);
  const distances = points.map((point) => ({ ...point, distance: greatCircleKm({ lon: longitude, lat: latitude }, point) })).sort((a, b) => a.distance - b.distance);
  document.querySelector("#range-source-count").textContent = `OBIS total: ${Number(entry.total || 0).toLocaleString()} · Mapped sample: ${points.length} points · Regional query: ${Number(entry.local_window_total || 0).toLocaleString()} records`;
  document.querySelector("#range-local-summary").textContent = nearby.length
    ? `${nearby.length} of the mapped sample points fall within ${radius} km. OBIS reports ${Number(entry.local_window_total || 0).toLocaleString()} records in the surrounding regional query window. These are reported occurrences, not proof of a resident population.`
    : Number(entry.local_window_total || 0) > 0
      ? `OBIS reports ${Number(entry.local_window_total).toLocaleString()} records in the surrounding regional query window, but no returned point sample falls inside the exact ${radius} km circle. This is inconclusive; inspect the records and source datasets.`
      : `No records were returned from the surrounding regional query window for this taxon. That does not establish absence; coverage can be sparse.`;
  const W = 720, H = 340, pad = { l: 42, r: 18, t: 18, b: 34 }, innerW = W - pad.l - pad.r, innerH = H - pad.t - pad.b;
  const localLatSpan = Math.min(85, Math.max(0.2, radius / 110.574 * 1.3));
  const localLonSpan = Math.min(180, localLatSpan / Math.max(0.1, Math.cos(latitude * Math.PI / 180)));
  const mapBounds = rangeMapMode === "local"
    ? { west: Math.max(-180, longitude - localLonSpan), east: Math.min(180, longitude + localLonSpan), south: Math.max(-90, latitude - localLatSpan), north: Math.min(90, latitude + localLatSpan) }
    : { west: -180, east: 180, south: -90, north: 90 };
  const x = (lon) => pad.l + (lon - mapBounds.west) / Math.max(0.001, mapBounds.east - mapBounds.west) * innerW;
  const y = (lat) => pad.t + (mapBounds.north - lat) / Math.max(0.001, mapBounds.north - mapBounds.south) * innerH;
  const continents = [
    "-168,70 -145,60 -130,52 -124,40 -117,32 -105,25 -97,18 -83,9 -78,25 -80,34 -70,45 -60,52 -75,60 -100,72 -130,72",
    "-81,12 -70,8 -61,-2 -50,-8 -45,-20 -53,-34 -68,-55 -75,-42 -79,-20",
    "-18,36 -5,37 10,34 25,31 34,20 45,11 50,-12 38,-35 22,-35 12,-27 5,-7 -5,5 -16,15",
    "-10,70 5,72 26,68 40,58 32,48 20,42 8,45 -2,55",
    "40,55 58,60 80,55 95,48 115,50 130,44 145,50 160,60 178,52 165,40 145,35 130,22 120,8 105,5 95,20 80,8 68,25 55,25 45,40",
    "112,-11 130,-12 145,-18 153,-28 145,-39 130,-35 115,-25",
    "-50,82 -20,80 -25,62 -42,58 -55,68"
  ];
  const lonStep = rangeMapMode === "local" ? (mapBounds.east - mapBounds.west) / 4 : 60;
  const latStep = rangeMapMode === "local" ? (mapBounds.north - mapBounds.south) / 4 : 30;
  const lonTicks = Array.from({ length: 5 }, (_, index) => mapBounds.west + lonStep * index);
  const latTicks = Array.from({ length: 5 }, (_, index) => mapBounds.south + latStep * index);
  const grid = lonTicks.map((v) => `<line x1="${x(v)}" y1="${pad.t}" x2="${x(v)}" y2="${H-pad.b}"/><text x="${x(v)}" y="${H-12}" text-anchor="middle">${v.toFixed(1)}°</text>`).join("") + latTicks.map((v) => `<line x1="${pad.l}" y1="${y(v)}" x2="${W-pad.r}" y2="${y(v)}"/><text x="${pad.l-8}" y="${y(v)+3}" text-anchor="end">${v.toFixed(1)}°</text>`).join("");
  const shapes = rangeMapMode === "global" ? continents.map((coords) => `<polygon points="${coords.trim().split(/\s+/).map((pair) => { const [lon, lat] = pair.split(",").map(Number); return `${x(lon)},${y(lat)}`; }).join(" ")}"/>`).join("") : "";
  const markers = points.map((point, index) => {
    const isLocal = greatCircleKm({ lon: longitude, lat: latitude }, point) <= radius;
    const inView = point.lon >= mapBounds.west && point.lon <= mapBounds.east && point.lat >= mapBounds.south && point.lat <= mapBounds.north;
    if (!inView) return "";
    return `<circle class="range-point${isLocal ? " local" : ""}" data-range-point="${index}" cx="${x(point.lon)}" cy="${y(point.lat)}" r="${isLocal ? 5 : 3.5}" tabindex="0" role="button" aria-label="Inspect ${escapeHTML(entry.species)} record at ${point.lat.toFixed(3)}, ${point.lon.toFixed(3)}"><title>${escapeHTML(entry.species)} · ${point.lat.toFixed(3)}, ${point.lon.toFixed(3)} · ${escapeHTML(point.date || "date unavailable")} · ${escapeHTML(point.dataset)}</title></circle>`;
  }).join("");
  document.querySelectorAll("[data-range-map-mode]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.rangeMapMode === rangeMapMode)));
  const radiusRx = (radius / 110.574) / (mapBounds.east - mapBounds.west) * innerW;
  const radiusRy = (radius / 110.574) / (mapBounds.north - mapBounds.south) * innerH;
  document.querySelector("#range-map").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${rangeMapMode === "local" ? "Local-area" : "Global"} view of sampled marine occurrences for ${escapeHTML(entry.species)}"><rect x="${pad.l}" y="${pad.t}" width="${innerW}" height="${innerH}" rx="10" class="map-ocean"/><g class="map-grid">${grid}</g><g class="map-land">${shapes}</g>${rangeMapMode === "local" ? `<ellipse class="range-radius-ring" cx="${x(longitude)}" cy="${y(latitude)}" rx="${radiusRx}" ry="${radiusRy}"/>` : ""}${markers}<circle class="camera-point" cx="${x(longitude)}" cy="${y(latitude)}" r="7"><title>Camera: ${latitude.toFixed(4)}, ${longitude.toFixed(4)}</title></circle><text x="${x(longitude)+10}" y="${y(latitude)-10}" class="camera-label">Camera</text></svg><div class="map-legend"><span><i class="legend-record"></i> OBIS occurrence</span><span><i class="legend-local"></i> Within ${radius} km</span><span><i class="legend-camera"></i> Camera position</span></div>`;
  const inViewCount = points.filter((point) => point.lon >= mapBounds.west && point.lon <= mapBounds.east && point.lat >= mapBounds.south && point.lat <= mapBounds.north).length;
  document.querySelector("#range-map").setAttribute("aria-label", `${inViewCount} of ${points.length} sampled points are shown in the ${rangeMapMode} map view.`);
  document.querySelectorAll("[data-range-point]").forEach((marker) => {
    const inspect = () => { const point = points[Number(marker.dataset.rangePoint)]; if (!point) return; const distance = greatCircleKm({ lon: longitude, lat: latitude }, point); const source = point.dataset_url ? `<a href="${escapeHTML(point.dataset_url)}" target="_blank" rel="noopener">Open OBIS dataset</a>` : escapeHTML(point.dataset || "OBIS dataset"); document.querySelector("#range-selected-record").innerHTML = `<strong>Selected occurrence</strong><br>${escapeHTML(point.scientific_name || entry.species)} · ${point.lat.toFixed(4)}°, ${point.lon.toFixed(4)}° · ${Math.round(distance).toLocaleString()} km from camera<br>Recorded: ${escapeHTML(point.date || "date unavailable")} · Source: ${source}`; };
    marker.addEventListener("click", inspect);
    marker.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); inspect(); } });
  });
  document.querySelector("#range-selected-record").textContent = `${inViewCount} sampled points shown here. Select a marker for its date, coordinates, distance, and source dataset.`;
  const months = Array(12).fill(0);
  points.forEach((point) => { const month = Number(point.date?.slice(5, 7)); if (month >= 1 && month <= 12) months[month - 1] += 1; });
  const datedCount = months.reduce((sum, count) => sum + count, 0);
  document.querySelector("#range-month-note").textContent = `${datedCount} dated records shown from ${points.length} mapped sample points. The bounded sample is not a complete seasonal or migration pattern.`;
  const max = Math.max(1, ...months);
  const labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  document.querySelector("#range-month-chart").innerHTML = `<div class="month-bars">${months.map((count, index) => `<div class="month-column" title="${labels[index]}: ${count} records"><span>${count || ""}</span><i style="height:${count ? Math.max(4, count / max * 100) : 2}%"></i><small>${labels[index]}</small></div>`).join("")}</div>`;
  const nearest = distances[0];
  document.querySelector("#range-details").innerHTML = nearest
    ? `<div class="range-callout"><strong>Nearest sampled occurrence:</strong> ${nearest.lat.toFixed(3)}°, ${nearest.lon.toFixed(3)}° · approximately ${Math.round(nearest.distance).toLocaleString()} km from the camera${nearest.date ? ` · recorded ${escapeHTML(nearest.date)}` : ""}. This is a comparison point, not a source population or migration origin.</div>`
    : `<div class="range-callout">OBIS returned no usable coordinates for this candidate.</div>`;
}

function updateRangeReportLinks(selectedSpecies) {
  const jobId = latestStatistics?.jobId || currentJobId || localStorage.getItem("seal-job-id");
  if (!jobId) return;
  const base = `/api/jobs/${encodeURIComponent(jobId)}/report.pdf`;
  const href = selectedSpecies ? `${base}?range_species=${encodeURIComponent(selectedSpecies)}` : base;
  document.querySelectorAll(".results-report-link,.statistics-report-link,.species-report-link").forEach((link) => { link.href = href; });
}

document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-range-map-mode]");
  if (!button || !speciesRangePayload) return;
  rangeMapMode = button.dataset.rangeMapMode === "global" ? "global" : "local";
  renderRangeSpecies();
});

document.querySelector("#range-form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const status = document.querySelector("#range-status"), submit = document.querySelector("#range-submit");
  const species = [...new Set(speciesTableRows.filter(([, decision]) => !decision.unlabelled && decision.label && decision.label !== "Unlabelled fish").map(([, decision]) => decision.label))].slice(0, 12);
  if (!species.length) { status.textContent = "This run has no labelled species candidates to compare."; return; }
  submit.disabled = true;
  status.textContent = `Looking up ${species.length} candidate species in OBIS…`;
  document.querySelector("#range-results").classList.add("hidden");
  try {
    const jobId = latestStatistics?.jobId || currentJobId || localStorage.getItem("seal-job-id");
    if (!jobId) throw new Error("Could not identify this analysis run. Refresh the page and try again.");
    const response = await fetch("/api/species/range", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ job_id: jobId, longitude: Number(document.querySelector("#range-lon").value), latitude: Number(document.querySelector("#range-lat").value), radius_km: Number(document.querySelector("#range-radius").value), species }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not query occurrence data.");
    speciesRangePayload = data;
    const select = document.querySelector("#range-species-select");
    select.innerHTML = data.species.map((item) => `<option value="${escapeHTML(item.species)}">${escapeHTML(item.species)}${item.error ? " · lookup unavailable" : ""}</option>`).join("");
    select.onchange = () => { renderRangeSpecies(); updateRangeReportLinks(select.value); };
    updateRangeReportLinks(select.value);
    document.querySelector("#range-results").classList.remove("hidden");
    status.textContent = data.species.some((item) => item.error) ? "Lookup completed; some species did not respond. Results below are a partial sample." : "Occurrence lookup complete.";
    renderRangeSpecies();
  } catch (error) {
    status.textContent = error.message;
  } finally {
    submit.disabled = false;
  }
});

function buildThreeChart(containerId, draw) {
  const host = document.getElementById(containerId);
  if (!host) return;
  host._chartCleanup?.();
  host.replaceChildren();
  if (!window.THREE) {
    host.textContent = "Three.js could not load. Check the internet connection, then refresh this page.";
    return;
  }
  const THREE = window.THREE;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf2f9fb);
  const camera = new THREE.OrthographicCamera(-6, 6, 2.8, -2.8, 0.1, 50);
  camera.position.z = 20;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.setSize(Math.max(host.clientWidth, 320), Math.max(host.clientHeight, 260));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  host.append(renderer.domElement);
  const tooltip = document.createElement("div");
  tooltip.className = "chart-tooltip";
  tooltip.setAttribute("role", "status");
  host.append(tooltip);
  const plot = new THREE.Group();
  scene.add(plot);

  // A flat XY grid keeps the data in 2D while Three.js handles hover interaction.
  const gridPoints = [];
  for (let x = -5; x <= 5; x += 1) gridPoints.push(new THREE.Vector3(x, -2.15, 0), new THREE.Vector3(x, 2.15, 0));
  for (let y = -2; y <= 2; y += 1) gridPoints.push(new THREE.Vector3(-5, y, 0), new THREE.Vector3(5, y, 0));
  plot.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(gridPoints),
    new THREE.LineBasicMaterial({ color: 0xd8e8ed })));
  const axes = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-5, -2.15, 0.01), new THREE.Vector3(5, -2.15, 0.01),
    new THREE.Vector3(-5, -2.15, 0.01), new THREE.Vector3(-5, 2.15, 0.01),
  ]);
  plot.add(new THREE.LineSegments(axes, new THREE.LineBasicMaterial({ color: 0x648390 })));
  draw(THREE, plot);

  const raycaster = new THREE.Raycaster();
  raycaster.params.Line.threshold = 0.12;
  const pointer = new THREE.Vector2();
  const interactive = [];
  plot.traverse((object) => { if (object.userData.tooltip) interactive.push(object); });
  let hovered = null;
  const setHover = (object, event) => {
    if (hovered && hovered !== object) {
      if (!hovered.isLine) hovered.scale.setScalar(1);
      if (hovered.material?.color && hovered.userData.baseColor != null) hovered.material.color.setHex(hovered.userData.baseColor);
    }
    hovered = object;
    if (object) {
      if (!object.isLine) object.scale.setScalar(1.22);
      if (object.material?.color) {
        object.userData.baseColor ??= object.material.color.getHex();
        object.material.color.setHex(0xf08b36);
      }
      tooltip.textContent = object.userData.tooltip;
      const rect = host.getBoundingClientRect();
      tooltip.style.left = Math.min(event.clientX - rect.left + 12, rect.width - 220) + "px";
      tooltip.style.top = Math.max(8, event.clientY - rect.top - 38) + "px";
      tooltip.classList.add("visible");
      renderer.domElement.style.cursor = "pointer";
    } else {
      tooltip.classList.remove("visible");
      renderer.domElement.style.cursor = "crosshair";
    }
    renderer.render(scene, camera);
  };
  const onMove = (event) => {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    setHover(raycaster.intersectObjects(interactive, false)[0]?.object || null, event);
  };
  const onLeave = () => setHover(null, { clientX: 0, clientY: 0 });
  renderer.domElement.addEventListener("pointermove", onMove);
  renderer.domElement.addEventListener("pointerleave", onLeave);
  const resize = new ResizeObserver(() => {
    if (!host.clientWidth || !host.clientHeight) return;
    const aspect = host.clientWidth / host.clientHeight;
    camera.left = -6;
    camera.right = 6;
    camera.top = 6 / aspect;
    camera.bottom = -6 / aspect;
    camera.updateProjectionMatrix();
    renderer.setSize(host.clientWidth, host.clientHeight);
    renderer.render(scene, camera);
  });
  resize.observe(host);
  renderer.render(scene, camera);
  host._chartCleanup = () => {
    resize.disconnect();
    renderer.domElement.removeEventListener("pointermove", onMove);
    renderer.domElement.removeEventListener("pointerleave", onLeave);
    plot.traverse((object) => {
      object.geometry?.dispose?.();
      if (Array.isArray(object.material)) object.material.forEach((material) => material.dispose());
      else object.material?.dispose?.();
    });
    renderer.dispose();
  };
}

function renderStatisticsCharts() {
  if (!latestStatistics || !document.querySelector("#page-Statistics").classList.contains("active")) return;
  const { rows, job } = latestStatistics;
  const frameCount = Math.max(1, Number(job.metrics?.frames_processed || 1));
  const frameWidth = Math.max(1, Number(job.metrics?.frame_width || 1));
  const frameHeight = Math.max(1, Number(job.metrics?.frame_height || 1));
  const fps = Math.max(1, Number(job.metrics?.video_fps || 25));
  const palette = [0x087fa7, 0x19a889, 0xf29f3d, 0x7659c6, 0xe15b72, 0x46a2d7, 0x95ad30, 0xd0773c];
  const dot = (THREE, x, y, radius, color, tooltipText) => {
    const mesh = new THREE.Mesh(new THREE.CircleGeometry(radius, 12), new THREE.MeshBasicMaterial({ color }));
    mesh.position.set(x, y, 0.08);
    mesh.userData.tooltip = tooltipText;
    return mesh;
  };

  buildThreeChart("stat-count-graph", (THREE, plot) => {
    if (!rows.length) return;
    const binFrames = Math.max(1, Math.round(fps * 2));
    const bins = Math.max(1, Math.ceil(frameCount / binFrames));
    const counts = Array(bins).fill(0);
    rows.forEach((row) => { counts[Math.min(bins - 1, Math.floor(Number(row.frame) / binFrames))] += 1; });
    const max = Math.max(...counts, 1);
    counts.forEach((count, index) => {
      const barWidth = Math.min(0.72, 9.2 / bins * 0.72);
      const barHeight = (count / max) * 3.8;
      const bar = new THREE.Mesh(new THREE.PlaneGeometry(barWidth, Math.max(barHeight, 0.035)),
        new THREE.MeshBasicMaterial({ color: 0x0786aa, transparent: true, opacity: 0.88 }));
      const x = -4.6 + (index + 0.5) * (9.2 / bins);
      bar.position.set(x, -2.05 + barHeight / 2, 0.08);
      const startSec = index * binFrames / fps;
      const endSec = Math.min(frameCount, (index + 1) * binFrames) / fps;
      bar.userData.tooltip = startSec.toFixed(1) + "–" + endSec.toFixed(1) + " sec · " + count + " detections";
      plot.add(bar);
    });
  });

  buildThreeChart("stat-track-graph", (THREE, plot) => {
    if (!rows.length) return;
    const groups = new Map();
    rows.forEach((row) => {
      const id = String(row.track_id);
      if (!id || id === "None" || id === "null") return;
      if (!groups.has(id)) groups.set(id, []);
      groups.get(id).push(row);
    });
    [...groups.entries()].forEach(([id, points], index) => {
      points.sort((a, b) => Number(a.frame) - Number(b.frame));
      const xy = points.map((row) => new THREE.Vector3((Number(row.center_x) / frameWidth - 0.5) * 9.2,
        (0.5 - Number(row.center_y) / frameHeight) * 4.0, 0.08));
      if (xy.length < 2) return;
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(xy),
        new THREE.LineBasicMaterial({ color: palette[index % palette.length], transparent: true, opacity: 0.84 }));
      line.userData.tooltip = "Track " + id + " · " + points.length + " detections · frames " + points[0].frame + "–" + points[points.length - 1].frame;
      plot.add(line);
      const last = points[points.length - 1];
      plot.add(dot(THREE, xy[xy.length - 1].x, xy[xy.length - 1].y, 0.075, palette[index % palette.length],
        "Track " + id + " · latest position (" + Math.round(Number(last.center_x)) + ", " + Math.round(Number(last.center_y)) + ") px"));
    });
  });

  buildThreeChart("stat-confidence-graph", (THREE, plot) => {
    if (!rows.length) return;
    const step = Math.max(1, Math.ceil(rows.length / 1400));
    rows.filter((_, index) => index % step === 0).forEach((row) => {
      const confidence = Math.max(0, Math.min(1, Number(row.confidence)));
      const frame = Number(row.frame);
      const x = (frame / frameCount - 0.5) * 9.2;
      const y = (confidence - 0.5) * 4.0;
      const color = new THREE.Color().setHSL(0.53 - confidence * 0.12, 0.78, 0.42);
      const track = row.track_id && row.track_id !== "None" ? "Track " + row.track_id : "Untracked detection";
      plot.add(dot(THREE, x, y, 0.045, color,
        track + " · frame " + frame + " · " + (frame / fps).toFixed(2) + " sec · " + (confidence * 100).toFixed(1) + "% confidence"));
    });
  });
}
function renderResults(jobId, job) {
  clearInterval(pollTimer);
  document.querySelector("#live-preview-wrap").classList.add("hidden");
  localStorage.setItem("seal-job-id", jobId);
  const reportUrl = `/api/jobs/${jobId}/report.pdf`;
  document.querySelector(".results-report-link").href = reportUrl;
  document.querySelector(".statistics-report-link").href = reportUrl;
  document.querySelector(".species-report-link").href = reportUrl;
  document.querySelector("#empty-results").classList.add("hidden");
  const content = document.querySelector("#results-content");
  content.classList.remove("hidden");
  const metrics = job.metrics || {};
  const species = metrics.species_classification;
  const cards = [
    ["Frames processed", Number(metrics.frames_processed || 0).toLocaleString()],
    ["Processing speed", `${metrics.processing_fps ?? "—"} FPS`],
    ["Mean inference", `${metrics.mean_inference_ms_per_frame ?? "—"} ms / frame`],
    ["Detection rows", Number(metrics.detection_rows || 0).toLocaleString()],
    ["Track IDs", Number(metrics.unique_track_ids || 0).toLocaleString()],
  ];
  if (species) cards.push(["Species candidates", `${species.species_count} · ${species.classified_tracks} tracks`]);
  document.querySelector("#metrics-grid").innerHTML = cards.map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`).join("");
  const speciesSummary = document.querySelector("#species-summary");
  const emptySpecies = document.querySelector("#empty-species");
  const speciesPage = document.querySelector("#species-page-content");
  speciesSummary.classList.toggle("hidden", !species);
  emptySpecies.classList.toggle("hidden", Boolean(species));
  speciesPage.classList.toggle("hidden", !species);
  if (species) {
    const decisions = Object.entries(species.track_decisions || {}).sort((a, b) => Number(a[0]) - Number(b[0]));
    speciesTableRows = decisions;
    speciesReviewJobId = jobId;
    speciesReviewLabels = {};
    speciesTablePages.results = 0;
    speciesTablePages.species = 0;
    speciesTablePages["species-labelled"] = 0;
    speciesTablePages["species-unlabelled"] = 0;
    renderSpeciesTable("species-table", "results");
    renderSpeciesTable("species-page-table", "species");
    document.querySelector("#species-note").textContent = `${species.classifier} · ${species.classifier_device.toUpperCase()} · ${species.classified_tracks} classified tracks. ${species.note}`;
    document.querySelector("#species-page-note").textContent = `${species.classifier} · ${species.classifier_device.toUpperCase()} · ${species.classified_tracks} classified tracks. ${species.note}`;
    document.querySelector("#species-page-title").textContent = `${species.species_count} candidate species across ${species.classified_tracks} tracks`;
    loadSpeciesReviewLabels(jobId);
  }
  const videoUrl = job.video_url || `/api/jobs/${jobId}/video`;
  const csvUrl = job.csv_url || `/api/jobs/${jobId}/tracks.csv`;
  const jsonUrl = job.benchmark_url || `/api/jobs/${jobId}/benchmark.json`;
  const artifact = (key, fallback) => job[key] || `/api/jobs/${jobId}/${fallback}`;
  const video = document.querySelector("#result-video");
  const playbackMessage = document.querySelector("#video-playback-message");
  let triedApiVideo = false;
  playbackMessage?.classList.add("hidden");
  video.onerror = () => {
    if (!triedApiVideo) {
      triedApiVideo = true;
      video.src = `${videoUrl}?v=${Date.now()}`;
      document.querySelector(".download-video").href = videoUrl;
      video.load();
      return;
    }
    if (playbackMessage) {
      playbackMessage.textContent = "This browser could not decode the annotated file. Download it or rerun this clip to create a browser-ready WebM.";
      playbackMessage.classList.remove("hidden");
    }
  };
  video.onloadeddata = () => playbackMessage?.classList.add("hidden");
  const extension = job.video_extension || job.metrics?.annotated_video_extension || ".webm";
  const localVideoUrl = `/result_videos/${jobId}${extension}`;
  document.querySelector(".download-video").href = localVideoUrl;
  video.src = `${localVideoUrl}?v=${Date.now()}`;
  video.load();
  document.querySelector(".download-csv").href = csvUrl;
  document.querySelector(".download-json").href = jsonUrl;
  document.querySelector(".download-bundle").href = artifact("bundle_url", "bundle.zip");
  document.querySelector(".download-heatmap").href = artifact("heatmap_url", "heatmap.png");
  document.querySelector(".download-trajectories").href = artifact("trajectories_url", "trajectories.png");
  document.querySelector("#heatmap-image").src = `${artifact("heatmap_url", "heatmap.png")}?v=${Date.now()}`;
  document.querySelector("#trajectory-image").src = `${artifact("trajectories_url", "trajectories.png")}?v=${Date.now()}`;
  fetch(csvUrl, { cache: "no-store" }).then(async (response) => {
    if (!response.ok) throw new Error("Could not load this run’s tracks.");
    setStatistics(jobId, job, parseCsvRows(await response.text()));
  }).catch((error) => {
    document.querySelector("#empty-statistics h2").textContent = "Statistics unavailable";
    document.querySelector("#empty-statistics p").textContent = error.message;
    document.querySelector("#empty-statistics").classList.remove("hidden");
  });
  const backgroundSummary = document.querySelector("#background-summary");
  const bg = metrics.background_video_check;
  backgroundSummary.classList.toggle("hidden", !bg);
  backgroundSummary.textContent = bg ? `Fish-free background check · ${Number(bg.detection_count).toLocaleString()} detections across ${Number(bg.frames_processed).toLocaleString()} frames (${bg.detections_per_frame} per frame). This is an assumed-background false-positive count, not a precision score.` : "";
  document.querySelector("#result-title").textContent = `${Number(metrics.frames_processed || 0).toLocaleString()} frames analyzed`;
  document.querySelector(".progress-card")?.classList.add("hidden");
  navigateTo("Results");
}

function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
}

async function pollJob(jobId) {
  try {
    const response = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || "Could not find this analysis session.");
    if (job.state === "done") {
      renderResults(jobId, job);
      runButton.disabled = false;
      runButton.innerHTML = 'Run detection <span aria-hidden="true">→</span>';
      return;
    }
    if (job.state === "error") {
      showError(job.message || "Analysis failed.");
      return;
    }
    showProgress(job);
  } catch (error) {
    showError(error.message || "Could not connect to the local analysis server.");
    clearInterval(pollTimer);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!videoInput.files[0]) return;
  errorBanner.classList.add("hidden");
  progressCard.classList.remove("hidden");
  runButton.disabled = true;
  runButton.textContent = "Uploading…";
  navigateTo("Upload");
  document.querySelector("#progress-title").textContent = "Uploading video";
  document.querySelector("#progress-message").textContent = "Sending the selected files to the local processing service…";
  document.querySelector("#progress-fill").style.width = "3%";
  const data = new FormData();
  data.append("video", videoInput.files[0]);
  data.append("confidence", confidenceInput.value);
  data.append("enhancement", enhancementInput.value);
  data.append("imgsz", imgszInput.value);
  data.append("species", speciesInput.checked ? "true" : "false");
  if (backgroundInput.files[0]) data.append("background", backgroundInput.files[0]);
  try {
    const response = await fetch("/api/jobs", { method: "POST", body: data });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || "Upload failed.");
    currentJobId = body.job_id;
    localStorage.setItem("seal-job-id", currentJobId);
    runButton.textContent = "Processing…";
    pollTimer = setInterval(() => pollJob(currentJobId), 900);
    await pollJob(currentJobId);
  } catch (error) {
    showError(error.message || "Could not start analysis.");
  }
});

async function restoreLatestAnalysis() {
  let job = null;
  if (currentJobId) {
    const response = await fetch(`/api/jobs/${currentJobId}`, { cache: "no-store" });
    if (response.ok) job = await response.json();
  }
  if (!job || job.state === "error") {
    const latest = await fetch("/api/latest", { cache: "no-store" });
    if (latest.ok) {
      job = await latest.json();
      currentJobId = job.job_id;
    } else {
      const pointerResponse = await fetch("/latest.json", { cache: "no-store" });
      if (!pointerResponse.ok) return;
      const pointer = await pointerResponse.json();
      currentJobId = pointer.job_id;
      const restored = await fetch(`/api/jobs/${currentJobId}`, { cache: "no-store" });
      if (!restored.ok) return;
      job = await restored.json();
    }
    localStorage.setItem("seal-job-id", currentJobId);
  }
  if (job.state === "done") {
    renderResults(currentJobId || job.job_id, job);
    if (initialPage !== "Results") navigateTo(initialPage);
  }
  else if (["queued", "loading", "processing", "classifying"].includes(job.state)) {
    navigateTo("Upload");
    runButton.disabled = true;
    runButton.textContent = "Processing…";
    showProgress(job);
    pollTimer = setInterval(() => pollJob(currentJobId), 900);
  }
}
restoreLatestAnalysis().catch(() => {});

const liveInput = document.querySelector("#live-video-input");
const liveStart = document.querySelector("#live-start");
const liveStop = document.querySelector("#live-stop");
const liveImage = document.querySelector("#live-stream");
const liveReportLink = document.querySelector("#live-report-link");
const livePollTimer = { id: null };
let liveSessionId = null;
const storedLiveReportId = localStorage.getItem("seal-live-session-id");
if (storedLiveReportId && liveReportLink) {
  liveReportLink.href = `/api/live/${storedLiveReportId}/report.pdf`;
  liveReportLink.classList.remove("hidden");
}

fetch("/api/live/demo", { cache: "no-store" }).then((response) => response.json()).then((demo) => {
  const note = document.querySelector("#live-source-note");
  if (note) note.textContent = demo.available ? `Using generated demo clip by default: ${demo.filename}. Choose another clip to override it.` : "Choose a local clip to start the simulated feed.";
}).catch(() => {});

liveInput?.addEventListener("change", () => {
  const note = document.querySelector("#live-source-note");
  if (note) note.textContent = liveInput.files[0]?.name || "The generated demo clip will be used automatically when available.";
});
document.querySelector("#live-confidence")?.addEventListener("input", (event) => {
  document.querySelector("#live-confidence-value").value = Number(event.target.value).toFixed(2);
});

function setLiveState(message, active = false) {
  document.querySelector("#live-state-text").textContent = message;
  document.querySelector("#live-state").classList.toggle("is-live", active);
}

async function pollLiveStatus() {
  if (!liveSessionId) return;
  try {
    const response = await fetch(`/api/live/${liveSessionId}`, { cache: "no-store" });
    const state = await response.json();
    if (!response.ok) throw new Error(state.error || "Live session ended.");
    updateLiveAnalytics(state);
    document.querySelector("#live-detections").textContent = state.frame_number ? state.detection_count : "—";
    document.querySelector("#live-tracks").textContent = state.frame_number ? state.track_count : "—";
    document.querySelector("#live-loop").textContent = state.frame_number ? state.loop_count : "—";
    if (state.state === "loading") setLiveState("Loading Fishial detector…");
    if (state.state === "live") setLiveState(`Simulated feed running · loop ${state.loop_count}${state.species_enabled ? ` · ${Object.keys(state.species_labels || {}).length} species tracks` : ""}${state.warning ? " · CPU fallback after CUDA error" : ""}`, true);
    if (state.state === "error") {
      document.querySelector("#live-error").textContent = state.error || "Live monitoring stopped unexpectedly.";
      document.querySelector("#live-error").classList.remove("hidden");
      stopLiveUi("Feed stopped · check the error above");
    }
    if (state.state === "stopped") stopLiveUi("Feed stopped");
  } catch (error) {
    document.querySelector("#live-error").textContent = error.message;
    document.querySelector("#live-error").classList.remove("hidden");
    stopLiveUi("Feed stopped");
  }
}

function prepareLiveCanvas(id) {
  const canvas = document.getElementById(id);
  if (!canvas) return null;
  const width = Math.max(260, canvas.clientWidth);
  const height = Math.max(180, canvas.clientHeight);
  const ratio = Math.min(window.devicePixelRatio || 1, 1.5);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  return { canvas, context, width, height };
}

function drawLiveCountChart(history = []) {
  const surface = prepareLiveCanvas("live-count-chart");
  if (!surface) return;
  const { context: ctx, width, height } = surface;
  const left = 38, right = 14, top = 18, bottom = 27;
  const plotW = width - left - right, plotH = height - top - bottom;
  ctx.strokeStyle = "#d6e7ec"; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = top + plotH * i / 4;
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(width - right, y); ctx.stroke();
  }
  ctx.fillStyle = "#718895"; ctx.font = "10px sans-serif"; ctx.fillText("detections", 7, 12);
  ctx.fillText("time →", width - 49, height - 7);
  if (!history.length) { ctx.fillText("Start the feed to collect live samples", left + 12, top + plotH / 2); return; }
  const max = Math.max(1, ...history.map((item) => Number(item.detections || 0)), ...history.map((item) => Number(item.tracks || 0)));
  const drawSeries = (key, color) => {
    ctx.strokeStyle = color; ctx.lineWidth = 2.2; ctx.beginPath();
    history.forEach((item, index) => {
      const x = left + (history.length < 2 ? 0 : index / (history.length - 1)) * plotW;
      const y = top + plotH - (Number(item[key] || 0) / max) * plotH;
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  drawSeries("detections", "#0788ad"); drawSeries("tracks", "#18a889");
  ctx.fillStyle = "#0788ad"; ctx.fillRect(left, height - 14, 8, 3); ctx.fillStyle = "#496778"; ctx.fillText("Fish detections", left + 13, height - 10);
  ctx.fillStyle = "#18a889"; ctx.fillRect(left + 105, height - 14, 8, 3); ctx.fillStyle = "#496778"; ctx.fillText("Tracked IDs", left + 118, height - 10);
}

function drawLiveHeatmap(points = []) {
  const surface = prepareLiveCanvas("live-heatmap-chart");
  if (!surface) return;
  const { context: ctx, width, height } = surface;
  const pad = 18, x = pad, y = pad, w = width - pad * 2, h = height - pad * 2;
  ctx.fillStyle = "#e8f4f7"; ctx.fillRect(x, y, w, h);
  ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
  points.slice(-400).forEach((point) => {
    const px = x + Number(point.x || 0) * w, py = y + Number(point.y || 0) * h;
    const gradient = ctx.createRadialGradient(px, py, 1, px, py, 22);
    gradient.addColorStop(0, "rgba(3,126,164,.24)"); gradient.addColorStop(1, "rgba(3,126,164,0)");
    ctx.fillStyle = gradient; ctx.fillRect(px - 22, py - 22, 44, 44);
  });
  ctx.restore(); ctx.strokeStyle = "#86afbd"; ctx.lineWidth = 1; ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = "#6e8995"; ctx.font = "10px sans-serif"; ctx.fillText("0,0", x + 4, y + 12); ctx.fillText("1,1", x + w - 22, y + h - 5);
  if (!points.length) { ctx.fillStyle = "#718895"; ctx.fillText("No detection centroids yet", x + 12, y + h / 2); }
}

function drawLiveTrajectory(points = []) {
  const surface = prepareLiveCanvas("live-trajectory-chart");
  if (!surface) return;
  const { context: ctx, width, height } = surface;
  const pad = 18, x = pad, y = pad, w = width - pad * 2, h = height - pad * 2;
  ctx.fillStyle = "#f1f8fa"; ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = "#d7e8ed"; ctx.lineWidth = 1;
  for (let i = 1; i < 4; i += 1) { ctx.beginPath(); ctx.moveTo(x + w * i / 4, y); ctx.lineTo(x + w * i / 4, y + h); ctx.stroke(); ctx.beginPath(); ctx.moveTo(x, y + h * i / 4); ctx.lineTo(x + w, y + h * i / 4); ctx.stroke(); }
  ctx.strokeStyle = "#86afbd"; ctx.strokeRect(x, y, w, h);
  const groups = new Map();
  points.filter((point) => point.track_id != null).forEach((point) => {
    const key = `${point.loop}-${point.track_id}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(point);
  });
  const palette = ["#0788ad", "#18a889", "#ef9b3e", "#7659c6", "#df5a72", "#378ac8", "#8ca52a", "#c36d39"];
  [...groups.values()].slice(-30).forEach((path, index) => {
    if (path.length < 2) return;
    ctx.strokeStyle = palette[index % palette.length]; ctx.lineWidth = 2; ctx.beginPath();
    path.forEach((point, pointIndex) => {
      const px = x + Number(point.x || 0) * w, py = y + Number(point.y || 0) * h;
      if (pointIndex === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
  });
  if (!groups.size) { ctx.fillStyle = "#718895"; ctx.font = "11px sans-serif"; ctx.fillText("Track centroids will appear here", x + 12, y + h / 2); }
}

function updateLiveAnalytics(state) {
  const set = (id, value) => { const element = document.getElementById(id); if (element) element.textContent = value; };
  set("live-metric-frames", Number(state.processed_frames || 0).toLocaleString());
  set("live-metric-inference", state.mean_inference_ms == null ? "—" : `${Number(state.mean_inference_ms).toFixed(1)} ms`);
  set("live-metric-fps", state.processing_fps == null ? "—" : `${Number(state.processing_fps).toFixed(2)} FPS`);
  set("live-metric-ids", state.unique_track_ids_this_loop == null ? "—" : Number(state.unique_track_ids_this_loop).toLocaleString());
  set("live-metric-elapsed", state.elapsed_seconds == null ? "—" : `${Number(state.elapsed_seconds).toFixed(1)} s`);
  set("live-metric-hardware", state.hardware || "—");
  drawLiveCountChart(state.count_history || []);
  drawLiveHeatmap(state.centroid_history || []);
  drawLiveTrajectory(state.centroid_history || []);
}

function stopLiveUi(message = "Feed stopped") {
  clearInterval(livePollTimer.id);
  livePollTimer.id = null;
  liveStart.disabled = false;
  liveStart.classList.remove("hidden");
  liveStop.classList.add("hidden");
  setLiveState(message);
}

async function restoreActiveLive() {
  if (!liveStart || !liveStop) return;
  try {
    const response = await fetch("/api/live/active", { cache: "no-store" });
    const active = await response.json();
    if (!response.ok || !active.active) return;
    liveSessionId = active.session_id;
    localStorage.setItem("seal-live-session-id", liveSessionId);
    liveReportLink.href = `/api/live/${liveSessionId}/report.pdf`;
    liveReportLink.classList.remove("hidden");
    liveImage.src = `${active.stream_url}?v=${Date.now()}`;
    liveImage.classList.add("ready");
    document.querySelector("#live-placeholder").classList.add("hidden");
    liveStart.classList.add("hidden");
    liveStop.classList.remove("hidden");
    setLiveState(active.state === "loading" ? "Loading Fishial detector…" : "Simulated feed reconnected", active.state === "live");
    livePollTimer.id = setInterval(pollLiveStatus, 700);
    pollLiveStatus();
  } catch {
    // Keep the page usable if the local server is temporarily unavailable.
  }
}

liveStart?.addEventListener("click", async () => {
  document.querySelector("#live-error").classList.add("hidden");
  liveStart.disabled = true;
  setLiveState("Starting simulated feed…");
  const formData = new FormData();
  if (liveInput.files[0]) formData.append("video", liveInput.files[0]);
  formData.append("confidence", document.querySelector("#live-confidence").value);
  formData.append("imgsz", document.querySelector("#live-imgsz").value);
  formData.append("species", document.querySelector("#live-species")?.checked ? "true" : "false");
  try {
    const response = await fetch("/api/live/start", { method: "POST", body: formData });
    const raw = await response.text();
    let data;
    try { data = JSON.parse(raw); } catch {
      if (response.status === 405 || raw.includes("Method Not Allowed")) {
        throw new Error("This Seal server is still running the previous version. Stop it and restart web_server.py, then refresh this page.");
      }
      throw new Error(`The local server returned an unexpected response (${response.status}). Restart web_server.py and refresh.`);
    }
    if (!response.ok) throw new Error(data.error || "Could not start the feed.");
    liveSessionId = data.session_id;
    localStorage.setItem("seal-live-session-id", liveSessionId);
    liveReportLink.href = `/api/live/${liveSessionId}/report.pdf`;
    liveReportLink.classList.remove("hidden");
    liveImage.src = `${data.stream_url}?v=${Date.now()}`;
    liveImage.classList.add("ready");
    document.querySelector("#live-placeholder").classList.add("hidden");
    liveStart.classList.add("hidden");
    liveStart.disabled = false;
    liveStop.classList.remove("hidden");
    livePollTimer.id = setInterval(pollLiveStatus, 700);
    pollLiveStatus();
  } catch (error) {
    document.querySelector("#live-error").textContent = error.message;
    document.querySelector("#live-error").classList.remove("hidden");
    stopLiveUi("Could not start feed");
  }
});

liveStop?.addEventListener("click", async () => {
  if (!liveSessionId) return stopLiveUi();
  liveStop.disabled = true;
  setLiveState("Stopping feed…");
  try {
    await fetch(`/api/live/${liveSessionId}/stop`, { method: "POST" });
  } finally {
    liveImage.removeAttribute("src");
    liveImage.classList.remove("ready");
    document.querySelector("#live-placeholder").classList.remove("hidden");
    liveStop.disabled = false;
    stopLiveUi("Feed stopped");
    liveSessionId = null;
  }
});

restoreActiveLive();
