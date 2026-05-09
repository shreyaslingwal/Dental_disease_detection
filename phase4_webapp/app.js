
// Persistent storage for analysis records
const STORAGE_KEY = "dentiscan_records";
let serverUrl = "";
let selectedFile = null;

// ---- Storage Helpers ----
function getRecords() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || []; }
    catch { return []; }
}
function saveRecord(record) {
    const records = getRecords();
    records.unshift(record);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(records));
}
function clearRecords() {
    localStorage.removeItem(STORAGE_KEY);
}

// ---- Navigation ----
document.querySelectorAll(".nav-item").forEach(item => {
    item.addEventListener("click", e => {
        e.preventDefault();
        navigateTo(item.dataset.page);
    });
});

function navigateTo(page) {
    document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
    const nav = document.querySelector(`[data-page="${page}"]`);
    if (nav) nav.classList.add("active");
    document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
    const target = document.getElementById("page" + page.charAt(0).toUpperCase() + page.slice(1));
    if (target) target.classList.add("active");
    const breadcrumb = document.getElementById("breadcrumb");
    const titles = { dashboard: "Dashboard", upload: "Upload & Analyze", records: "Patient Records", reports: "Clinical Reports", settings: "Settings" };
    breadcrumb.innerHTML = `<a href="#">${titles[page] || page}</a>`;
    document.getElementById("sidebar").classList.remove("open");
    // Refresh dynamic content on navigation
    if (page === "dashboard") refreshDashboard();
    if (page === "records") renderRecordsList();
}

// Mobile menu
document.getElementById("menuToggle")?.addEventListener("click", () => {
    document.getElementById("sidebar").classList.toggle("open");
});

// ---- Server Connection ----
async function connectToServer() {
    const input = document.getElementById("serverUrl");
    const dot = document.getElementById("connDot");
    const text = document.getElementById("connText");
    const btn = document.getElementById("connectBtn");
    serverUrl = input.value.trim().replace(/\/+$/, "");
    if (!serverUrl) { showToast("Please enter the Colab ngrok URL."); return; }
    btn.textContent = "Connecting..."; btn.disabled = true;
    try {
        const res = await fetch(serverUrl + "/health", { method: "GET", headers: { "ngrok-skip-browser-warning": "true" } });
        const data = await res.json();
        dot.className = "conn-dot online";
        text.textContent = `Connected (${data.device || "GPU"})`;
        btn.textContent = "Connected";
        input.style.borderColor = "var(--success)";
        showToast("Server connected successfully");
    } catch (err) {
        dot.className = "conn-dot offline";
        text.textContent = "Connection failed";
        btn.textContent = "Retry"; btn.disabled = false;
        input.style.borderColor = "var(--error)";
        showToast("Connection failed - check the URL");
    }
}

// ---- File Upload & Drag-Drop ----
const dropzone = document.getElementById("uploadDropzone");
const fileInput = document.getElementById("fileInput");

dropzone?.addEventListener("click", (e) => {
    // Don't trigger file dialog again if the click came from the Browse button
    if (e.target.closest(".upload-browse-btn")) return;
    fileInput.click();
});
dropzone?.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("dragover"); });
dropzone?.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone?.addEventListener("drop", e => {
    e.preventDefault(); dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});
fileInput?.addEventListener("change", e => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith("image/")) { showToast("Please upload an image file."); return; }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = e => {
        document.getElementById("previewImage").src = e.target.result;
        document.getElementById("fileName").textContent = file.name;
        document.getElementById("fileSize").textContent = formatBytes(file.size);
        dropzone.style.display = "none";
        document.getElementById("uploadPreview").style.display = "block";
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null; fileInput.value = "";
    dropzone.style.display = "";
    document.getElementById("uploadPreview").style.display = "none";
    document.getElementById("resultsContainer").style.display = "none";
    document.getElementById("uploadContainer").style.display = "";
}

// ---- Analysis ----
async function analyzeImage() {
    if (!selectedFile) { showToast("Please upload an image first."); return; }
    if (!serverUrl) { showToast("Connect to the Colab server first."); return; }
    const analyzeBtn = document.getElementById("analyzeBtn");
    const btnText = analyzeBtn.querySelector(".btn-text");
    const btnLoader = analyzeBtn.querySelector(".btn-loader");
    const overlay = document.getElementById("loadingOverlay");
    const step = document.getElementById("loadingStep");
    const progress = document.getElementById("loadingProgress");

    analyzeBtn.disabled = true;
    btnText.textContent = "Analyzing...";
    btnLoader.style.display = "inline-block";
    overlay.style.display = "flex";

    const steps = [
        { text: "Sending image to GPU server...", p: "20%" },
        { text: "Running Swin-S classification...", p: "40%" },
        { text: "Generating GradCAM heatmap...", p: "60%" },
        { text: "Building clinical report...", p: "80%" },
        { text: "Post-processing findings...", p: "90%" },
    ];
    let idx = 0;
    const si = setInterval(() => { if (idx < steps.length) { step.textContent = steps[idx].text; progress.style.width = steps[idx].p; idx++; } }, 1500);

    try {
        const fd = new FormData(); fd.append("file", selectedFile);
        const res = await fetch(serverUrl + "/analyze", { method: "POST", body: fd, headers: { "ngrok-skip-browser-warning": "true" } });
        if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || `Server error: ${res.status}`); }
        const data = await res.json();
        clearInterval(si); progress.style.width = "100%"; step.textContent = "Complete!";

        // Save to records
        const patientId = document.getElementById("patientId")?.value.trim() || "PT-" + Math.floor(10000 + Math.random() * 90000);
        const scanType = document.getElementById("scanType")?.value || "Unknown";
        const notes = document.getElementById("clinicalNotes")?.value.trim() || "";
        const record = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            patientId: patientId,
            scanType: scanType,
            notes: notes,
            fileName: selectedFile.name,
            topPrediction: data.top_prediction || "No finding",
            confidenceScores: data.confidence_scores || {},
            originalImage: data.original_image,
            heatmapImage: data.heatmap_image,
            vizPanels: data.viz_panels || [],
            report: data.generated_report || "No report generated."
        };
        saveRecord(record);

        setTimeout(() => { overlay.style.display = "none"; displayResults(data); refreshDashboard(); }, 500);
    } catch (err) {
        clearInterval(si); overlay.style.display = "none";
        showToast("Analysis failed: " + err.message);
    } finally {
        analyzeBtn.disabled = false; btnText.textContent = "Run Analysis";
        btnLoader.style.display = "none"; progress.style.width = "0%";
    }
}

// ---- Display Results ----
function displayResults(data) {
    document.getElementById("uploadContainer").style.display = "none";
    document.getElementById("resultsContainer").style.display = "block";

    // Build horizontal scroll panels (Phase 2 style)
    const track = document.getElementById("vizPanelsTrack");
    track.innerHTML = "";

    const panels = data.viz_panels || [];
    if (panels.length > 0) {
        panels.forEach((panel, idx) => {
            const div = document.createElement("div");
            div.className = "viz-panel" + (idx > 0 && idx < panels.length - 1 ? " highlight-card" : "");
            const badgeHTML = panel.badge
                ? `<span class="viz-panel-badge${panel.title === 'Predicted Findings' ? ' findings' : ''}">${panel.badge}</span>`
                : "";
            div.innerHTML = `
                <div class="viz-panel-header">
                    <span class="viz-panel-title">${panel.title}</span>
                    ${badgeHTML}
                </div>
                <div class="viz-panel-body">
                    <div class="xray-frame">
                        <img src="data:image/jpeg;base64,${panel.image}" alt="${panel.title}">
                    </div>
                </div>`;
            track.appendChild(div);
        });
    } else {
        // Fallback: legacy 2-image mode
        const orig = data.original_image;
        const heat = data.heatmap_image;
        [{ title: "Original Radiograph", img: orig }, { title: "GradCAM Heatmap", img: heat }].forEach(p => {
            const div = document.createElement("div");
            div.className = "viz-panel";
            div.innerHTML = `
                <div class="viz-panel-header">
                    <span class="viz-panel-title">${p.title}</span>
                </div>
                <div class="viz-panel-body">
                    <div class="xray-frame">
                        <img src="data:image/jpeg;base64,${p.img}" alt="${p.title}">
                    </div>
                </div>`;
            track.appendChild(div);
        });
    }

    const grid = document.getElementById("scoresGrid"); grid.innerHTML = "";
    const scores = data.confidence_scores || {};
    for (const [disease, score] of Object.entries(scores)) {
        const pct = (score * 100).toFixed(1);
        const color = score > 0.85 ? "var(--error)" : score > 0.65 ? "var(--warning)" : score > 0.3 ? "var(--primary)" : "var(--success)";
        const item = document.createElement("div"); item.className = "score-item";
        item.innerHTML = `<span class="score-label">${disease}</span><div class="score-bar-bg"><div class="score-bar-fill" style="width:${pct}%;background:${color};"></div></div><span class="score-value" style="color:${color}">${pct}%</span>`;
        grid.appendChild(item);
    }
    document.getElementById("reportContent").textContent = data.generated_report || "No report generated.";
    document.getElementById("resultsContainer").scrollIntoView({ behavior: "smooth" });
}

// ---- Dashboard Updates ----
function refreshDashboard() {
    const records = getRecords();
    const totalReports = records.length;
    const uniquePatients = new Set(records.map(r => r.patientId)).size;
    const criticalCount = records.filter(r => {
        const scores = r.confidenceScores || {};
        return Object.values(scores).some(s => s > 0.85);
    }).length;
    const pendingCount = Math.max(0, totalReports - Math.floor(totalReports * 0.7));

    // Update stat values
    animateValue("statReports", totalReports);
    animateValue("statPatients", uniquePatients);
    animateValue("statCritical", criticalCount);
    animateValue("statPending", pendingCount);

    // Update trends
    updateTrend("trendReports", totalReports > 0 ? `${totalReports} total` : "");
    updateTrend("trendPatients", uniquePatients > 0 ? `${uniquePatients} unique` : "");
    updateTrend("trendCritical", criticalCount > 0 ? "Needs attention" : "", criticalCount > 0);
    updateTrend("trendPending", pendingCount > 0 ? `${pendingCount} pending` : "");

    // Update latest case card
    const caseBody = document.getElementById("latestCaseBody");
    if (records.length > 0) {
        const latest = records[0];
        const date = new Date(latest.timestamp).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
        const reportSnippet = (latest.report || "").substring(0, 200);
        caseBody.innerHTML = `
            <div class="case-section">
                <h4>Latest Finding: ${latest.topPrediction}</h4>
                <div class="note-item">
                    <span class="note-label">Patient:</span>
                    <span class="note-text">${latest.patientId} | ${latest.scanType || "Unknown"} | ${date}</span>
                </div>
                <div class="note-item">
                    <span class="note-label">Report Preview:</span>
                    <span class="note-text">${reportSnippet}...</span>
                </div>
            </div>
            <div class="case-footer">
                <button class="btn-secondary" onclick="navigateTo('upload')">
                    <span class="material-symbols-outlined">add_photo_alternate</span>
                    Analyze New Scan
                </button>
                <button class="btn-ghost" onclick="openRecordDetail(${latest.id})">View Full Record</button>
            </div>`;
    } else {
        caseBody.innerHTML = `
            <div class="empty-state-inline">
                <span class="material-symbols-outlined">biotech</span>
                <p>No analyses yet. Upload a dental X-ray to get started.</p>
                <button class="btn-primary" onclick="navigateTo('upload')">
                    <span class="material-symbols-outlined">add_photo_alternate</span>
                    Analyze New Scan
                </button>
            </div>`;
    }

    // Update recent reports list (show last 5)
    const reportList = document.getElementById("dashboardReportList");
    if (records.length > 0) {
        reportList.innerHTML = records.slice(0, 5).map(r => {
            const date = new Date(r.timestamp).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
            const hasCritical = Object.values(r.confidenceScores || {}).some(s => s > 0.85);
            const iconClass = hasCritical ? "warning-icon" : "routine";
            const iconName = hasCritical ? "warning" : "check_circle";
            const chipClass = hasCritical ? "chip-urgent" : "chip-complete";
            const chipText = hasCritical ? "Critical" : "Complete";
            return `<div class="report-row" onclick="openRecordDetail(${r.id})" style="cursor:pointer">
                <div class="report-icon ${iconClass}"><span class="material-symbols-outlined">${iconName}</span></div>
                <div class="report-info">
                    <span class="report-title">${r.topPrediction}</span>
                    <span class="report-meta">${r.patientId} &middot; ${date}</span>
                </div>
                <span class="chip ${chipClass}">${chipText}</span>
            </div>`;
        }).join("");
    } else {
        reportList.innerHTML = `<div class="empty-state-inline"><span class="material-symbols-outlined">description</span><p>No reports generated yet.</p></div>`;
    }
}

function animateValue(elId, target) {
    const el = document.getElementById(elId);
    if (!el) return;
    const current = parseInt(el.textContent) || 0;
    if (current === target) { el.textContent = target; return; }
    let val = current;
    const diff = target - current;
    const stepCount = Math.min(Math.abs(diff), 20);
    const stepSize = diff / stepCount;
    let i = 0;
    const interval = setInterval(() => {
        i++;
        val = i >= stepCount ? target : Math.round(current + stepSize * i);
        el.textContent = val;
        if (i >= stepCount) clearInterval(interval);
    }, 40);
}

function updateTrend(elId, text, isAlert) {
    const el = document.getElementById(elId);
    if (!el) return;
    el.textContent = text;
    el.className = "stat-trend" + (isAlert ? " alert" : text ? " up" : "");
}

// ---- Patient Records ----
function renderRecordsList() {
    const records = getRecords();
    const container = document.getElementById("recordsList");
    const empty = document.getElementById("recordsEmpty");

    if (records.length === 0) {
        container.innerHTML = "";
        container.appendChild(createEmptyRecordsState());
        return;
    }

    container.innerHTML = records.map(r => {
        const date = new Date(r.timestamp).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit" });
        const snippet = (r.report || "").substring(0, 80);
        const hasCritical = Object.values(r.confidenceScores || {}).some(s => s > 0.85);
        return `<div class="record-card" onclick="openRecordDetail(${r.id})">
            <div class="record-thumb">
                <img src="data:image/jpeg;base64,${r.heatmapImage}" alt="Heatmap">
            </div>
            <div class="record-body">
                <span class="record-prediction">${r.topPrediction}</span>
                <span class="record-patient">${r.patientId} | ${r.scanType || "N/A"} | ${r.fileName}</span>
                <span class="record-date">${date}</span>
                <span class="record-snippet">${snippet}...</span>
            </div>
            <div class="record-actions-col">
                <span class="chip ${hasCritical ? 'chip-urgent' : 'chip-complete'}">${hasCritical ? 'Critical' : 'Complete'}</span>
            </div>
        </div>`;
    }).join("");
}

function createEmptyRecordsState() {
    const div = document.createElement("div");
    div.className = "empty-state";
    div.id = "recordsEmpty";
    div.innerHTML = `<span class="material-symbols-outlined">folder_shared</span><h3>No Records Yet</h3><p>Completed analyses will appear here automatically.</p>`;
    return div;
}

function openRecordDetail(recordId) {
    const records = getRecords();
    const r = records.find(rec => rec.id === recordId);
    if (!r) return;

    const date = new Date(r.timestamp).toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit" });

    const overlay = document.createElement("div");
    overlay.className = "record-modal-overlay";
    overlay.onclick = e => { if (e.target === overlay) overlay.remove(); };
    overlay.innerHTML = `<div class="record-modal">
        <div class="modal-header">
            <h3>${r.topPrediction} - ${r.patientId}</h3>
            <button class="btn-icon" onclick="this.closest('.record-modal-overlay').remove()">
                <span class="material-symbols-outlined">close</span>
            </button>
        </div>
        <div class="modal-body">
            <p style="font-size:12px;color:var(--outline);margin-bottom:16px">${date} | ${r.scanType || "N/A"} | ${r.fileName}</p>
            <div class="viz-scroll-track" style="display:flex;gap:12px;overflow-x:auto;padding-bottom:8px;margin-bottom:20px">
                ${(r.vizPanels && r.vizPanels.length > 0
            ? r.vizPanels.map(p => `<div style="flex:0 0 auto;width:260px;border-radius:8px;overflow:hidden;border:1px solid var(--outline-variant);background:#1e293b">
                        <div style="padding:6px 10px;background:var(--surface-bright);border-bottom:1px solid var(--outline-variant);font-size:11px;font-weight:600">${p.title}${p.badge ? ' <span style="font-size:9px;color:var(--primary);margin-left:4px">' + p.badge + '</span>' : ''}</div>
                        <img src="data:image/jpeg;base64,${p.image}" alt="${p.title}" style="width:100%;display:block">
                    </div>`).join('')
            : `<img src="data:image/jpeg;base64,${r.originalImage}" alt="Original" style="width:260px;border-radius:8px">
                       <img src="data:image/jpeg;base64,${r.heatmapImage}" alt="Heatmap" style="width:260px;border-radius:8px">`
        )}
            </div>
            <h4 style="font-size:13px;font-weight:600;margin-bottom:8px;color:var(--on-surface-variant);text-transform:uppercase;letter-spacing:.04em">Clinical Report</h4>
            <div class="modal-report">${r.report}</div>
        </div>
    </div>`;
    document.body.appendChild(overlay);
}

function clearAllRecords() {
    if (!confirm("Clear all saved analysis records? This cannot be undone.")) return;
    clearRecords();
    refreshDashboard();
    renderRecordsList();
    showToast("All records cleared");
}

// ---- Utilities ----
function copyReport() {
    const text = document.getElementById("reportContent").textContent;
    navigator.clipboard.writeText(text).then(() => showToast("Report copied to clipboard"));
}
function newAnalysis() { resetUpload(); window.scrollTo({ top: 0, behavior: "smooth" }); }
function formatBytes(b) { if (b === 0) return "0 B"; const k = 1024, s = ["B", "KB", "MB", "GB"]; const i = Math.floor(Math.log(b) / Math.log(k)); return parseFloat((b / Math.pow(k, i)).toFixed(1)) + " " + s[i]; }
function showToast(msg) {
    const c = document.getElementById("toastContainer");
    const t = document.createElement("div"); t.className = "toast"; t.textContent = msg; c.appendChild(t);
    setTimeout(() => { t.style.opacity = "0"; setTimeout(() => t.remove(), 300); }, 3000);
}

// ---- Init ----
document.addEventListener("DOMContentLoaded", () => {
    refreshDashboard();
});
