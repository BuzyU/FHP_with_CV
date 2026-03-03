/**
 * FHP Detection — Web Frontend
 *
 * Captures webcam frames, sends to FastAPI backend,
 * displays results with skeleton overlay and stats.
 */

// ============================================
// Configuration
// ============================================

const CONFIG = {
    API_URL: `${window.location.protocol}//${window.location.host}`,

    // Detection
    MIN_FRAME_INTERVAL_MS: 33,
    API_CAPTURE_WIDTH: 320,
    API_CAPTURE_HEIGHT: 240,
    JPEG_QUALITY: 0.55,
    ALERT_COOLDOWN_MS: 8000,
    ALERT_DURATION_MS: 2000,
    TIMELINE_MAX_BARS: 80,

    // Skeleton overlay — MediaPipe landmark indices
    // 0=nose, 7=L_ear, 8=R_ear, 11=L_shoulder, 12=R_shoulder
    // 13=L_elbow, 14=R_elbow, 15=L_wrist, 16=R_wrist
    // 23=L_hip, 24=R_hip, 33=virtual NECK (computed by API)

    // FHP assessment chain (head → neck → shoulders)
    FHP_LINES: [
        [0, 7], [0, 8],       // Nose → Ears
        [7, 33], [8, 33],     // Ears → Neck
        [33, 11], [33, 12],   // Neck → Shoulders
        [11, 12],              // Shoulder line
    ],

    // Body skeleton (arms + torso)
    BODY_LINES: [
        [11, 13], [13, 15],   // Left arm
        [12, 14], [14, 16],   // Right arm
        [11, 23], [12, 24],   // Torso sides (shoulders → hips)
        [23, 24],              // Hip line
    ],

    // Primary joints (FHP chain: nose, ears, neck, shoulders)
    PRIMARY_JOINTS: [0, 7, 8, 33, 11, 12],
    // Secondary joints (arms + hips)
    SECONDARY_JOINTS: [13, 14, 15, 16, 23, 24],

    JOINT_COLORS: {
        0:  '#fbbf24',                    // Nose — amber
        7:  '#c084fc', 8:  '#c084fc',    // Ears — purple
        33: '#e2e8f0',                    // Neck — light
        11: '#38bdf8', 12: '#38bdf8',    // Shoulders — cyan
        13: '#818cf8', 14: '#818cf8',    // Elbows — indigo
        15: '#a78bfa', 16: '#a78bfa',    // Wrists — violet
        23: '#34d399', 24: '#34d399',    // Hips — green
    },
};

// ============================================
// State
// ============================================

const state = {
    detecting: false,
    sessionId: null,
    stream: null,
    detectionLoopRunning: false,
    cameraFacingMode: 'user',

    startTime: null,
    totalFrames: 0,
    normalFrames: 0,
    fhpFrames: 0,
    lastAlertTime: 0,
    lastResult: null,
    detectionFpsEma: 0,
    timeline: [],

    smoothNormalPct: 50,
    smoothFhpPct: 50,

    fhpHistory: [],
    FHP_HISTORY_MAX: 300,

    // Capture dimensions (for skeleton scaling)
    captureWidth: 640,
    captureHeight: 480,
};

// ============================================
// DOM
// ============================================

const $ = (id) => document.getElementById(id);
const webcam = $('webcam');
const displayCanvas = $('displayCanvas');
const ctx = displayCanvas.getContext('2d');
let displayAnimFrame = null;

// ============================================
// Start / Stop Detection
// ============================================

async function startDetection() {
    if (state.detecting) return;

    $('hero').classList.add('hidden');
    $('infoSection').classList.add('hidden');
    $('detectionPanel').classList.remove('hidden');
    $('videoLoading').classList.remove('hidden');

    try {
        state.stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: state.cameraFacingMode,
                width: { ideal: 640 },
                height: { ideal: 480 },
            },
            audio: false,
        });

        webcam.srcObject = state.stream;
        await webcam.play();

        displayCanvas.width = webcam.videoWidth;
        displayCanvas.height = webcam.videoHeight;

        // Match container aspect-ratio to webcam
        const container = document.querySelector('.video-container');
        container.style.aspectRatio = `${webcam.videoWidth} / ${webcam.videoHeight}`;

        // Live video render loop (mirrored)
        function renderFrame() {
            if (!state.detecting) return;

            ctx.save();
            ctx.translate(displayCanvas.width, 0);
            ctx.scale(-1, 1);
            ctx.drawImage(webcam, 0, 0, displayCanvas.width, displayCanvas.height);
            ctx.restore();

            // Overlay skeleton
            if (state.lastResult && state.lastResult.detected) {
                drawSkeleton(state.lastResult);
            }
            displayAnimFrame = requestAnimationFrame(renderFrame);
        }

        $('videoLoading').classList.add('hidden');
        $('statusDot').classList.add('active');
        $('statusLabel').textContent = 'Live';

        state.detecting = true;
        state.startTime = Date.now();
        state.sessionId = crypto.randomUUID();

        renderFrame();
        state.detectionLoopRunning = true;
        runDetectionLoop();
        startStatsTimer();

    } catch (err) {
        console.error('Camera error:', err);
        alert('Camera access denied. Please allow camera and try again.');
        stopDetection();
    }
}

function stopDetection() {
    state.detecting = false;
    state.detectionLoopRunning = false;

    if (_statsIntervalId) {
        clearInterval(_statsIntervalId);
        _statsIntervalId = null;
    }

    if (displayAnimFrame) {
        cancelAnimationFrame(displayAnimFrame);
        displayAnimFrame = null;
    }

    if (state.stream) {
        state.stream.getTracks().forEach(t => t.stop());
        state.stream = null;
    }

    $('statusDot').classList.remove('active');
    $('statusLabel').textContent = 'Offline';
    $('detectionPanel').classList.add('hidden');
    $('hero').classList.remove('hidden');
    $('infoSection').classList.remove('hidden');

    ctx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
}

async function toggleCamera() {
    state.cameraFacingMode = state.cameraFacingMode === 'user' ? 'environment' : 'user';
    if (state.detecting) {
        stopDetection();
        await startDetection();
    }
}

function resetSession() {
    state.totalFrames = 0;
    state.normalFrames = 0;
    state.fhpFrames = 0;
    state.startTime = Date.now();
    state.timeline = [];
    state.fhpHistory = [];
    state.smoothNormalPct = 50;
    state.smoothFhpPct = 50;
    state.detectionFpsEma = 0;
    state.sessionId = crypto.randomUUID();
    $('postureTimeline').innerHTML = '';
    drawFhpChart();
    updateStats();
}

// ============================================
// Capture & Send
// ============================================

let _captureCanvas = null;
let _captureCtx = null;

function getCaptureCanvas() {
    if (!_captureCanvas) {
        _captureCanvas = document.createElement('canvas');
        _captureCanvas.width = CONFIG.API_CAPTURE_WIDTH;
        _captureCanvas.height = CONFIG.API_CAPTURE_HEIGHT;
        _captureCtx = _captureCanvas.getContext('2d', { willReadFrequently: false });
    }
    return { canvas: _captureCanvas, ctx: _captureCtx };
}

async function runDetectionLoop() {
    let lastLoopTime = performance.now();
    while (state.detectionLoopRunning && state.detecting) {
        const frameStart = performance.now();
        await captureAndDetect();
        const elapsed = performance.now() - frameStart;

        const remaining = CONFIG.MIN_FRAME_INTERVAL_MS - elapsed;
        if (remaining > 0) {
            await new Promise(r => setTimeout(r, remaining));
        }

        const now = performance.now();
        const cycleDt = now - lastLoopTime;
        lastLoopTime = now;
        if (cycleDt > 0) {
            const instantFps = 1000 / cycleDt;
            state.detectionFpsEma = state.detectionFpsEma === 0
                ? instantFps
                : state.detectionFpsEma * 0.9 + instantFps * 0.1;
        }
    }
}

async function captureAndDetect() {
    if (!state.detecting || webcam.readyState < 2) return;

    const { canvas, ctx: capCtx } = getCaptureCanvas();
    const cw = canvas.width;
    const ch = canvas.height;

    // Mirror capture (same as display)
    capCtx.save();
    capCtx.translate(cw, 0);
    capCtx.scale(-1, 1);
    capCtx.drawImage(webcam, 0, 0, cw, ch);
    capCtx.restore();

    state.captureWidth = webcam.videoWidth || 640;
    state.captureHeight = webcam.videoHeight || 480;

    const dataUrl = canvas.toDataURL('image/jpeg', CONFIG.JPEG_QUALITY);

    try {
        const response = await fetch(`${CONFIG.API_URL}/api/detect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: dataUrl,
                session_id: state.sessionId,
            }),
        });

        if (!response.ok) {
            console.warn('API error:', response.status);
            return;
        }

        const result = await response.json();
        state.lastResult = result;

        updateResult(result);
        updateAngles(result);
        updateTechPanel(result);
        updateStats();
        updateTimeline(result);
        checkAlert(result);

    } catch (err) {
        console.warn('Detection error:', err.message);
    }
}

// ============================================
// Skeleton Drawing
// ============================================

function drawSkeleton(result) {
    if (!result.detected || !result.keypoints_2d) return;

    const kps = result.keypoints_2d;
    const canvasW = displayCanvas.width;
    const canvasH = displayCanvas.height;

    // Keypoints from API are in the capture resolution (320x240)
    // with coordinates already in pixel space (x*w, y*h of capture).
    // We need to scale to display canvas size.
    const scaleX = canvasW / CONFIG.API_CAPTURE_WIDTH;
    const scaleY = canvasH / CONFIG.API_CAPTURE_HEIGHT;

    const isNormal = result.classification !== 'FHP';

    // Margin: reject joints that are outside the visible canvas (bad MediaPipe guesses)
    const margin = 10;

    // Get scaled position for a landmark index, with bounds checking
    const getPos = (idx) => {
        if (idx < kps.length && kps[idx]) {
            const x = kps[idx][0] * scaleX;
            const y = kps[idx][1] * scaleY;
            // Reject points clearly outside the visible frame
            if (x < -margin || x > canvasW + margin || y < -margin || y > canvasH + margin) {
                return null;
            }
            return [x, y];
        }
        return null;
    };

    // Compute shoulder span for plausibility checks on arms/hips
    const lsh = getPos(11), rsh = getPos(12);
    let shoulderSpan = 0;
    if (lsh && rsh) {
        shoulderSpan = Math.hypot(rsh[0] - lsh[0], rsh[1] - lsh[1]);
    }
    const midShoulderY = (lsh && rsh) ? (lsh[1] + rsh[1]) / 2 : canvasH * 0.4;

    // Plausibility check: reject arm/hip keypoints that are too far from shoulders
    // (MediaPipe guesses wildly when limbs are off-screen)
    const getCheckedPos = (idx) => {
        const pos = getPos(idx);
        if (!pos || shoulderSpan === 0) return null;
        const maxDist = shoulderSpan * 2.5;  // arm/hip can't be >2.5x shoulder width away
        const ref = (idx === 23 || idx === 24) ? midShoulderY : null;
        // For hips: must be BELOW shoulders
        if ((idx === 23 || idx === 24) && pos[1] < midShoulderY - shoulderSpan * 0.3) return null;
        // Distance check from nearest shoulder
        const anchor = (idx === 13 || idx === 15 || idx === 23) ? lsh : rsh;
        if (anchor) {
            const d = Math.hypot(pos[0] - anchor[0], pos[1] - anchor[1]);
            if (d > maxDist) return null;
        }
        return pos;
    };

    ctx.lineCap = 'round';
    ctx.setLineDash([]);

    // --- Body skeleton (arms + torso — drawn first, behind FHP lines) ---
    const bodyColor = isNormal
        ? 'rgba(34, 197, 94, 0.5)'
        : 'rgba(239, 68, 68, 0.5)';
    ctx.strokeStyle = bodyColor;
    ctx.lineWidth = 1.8;

    for (const [i, j] of CONFIG.BODY_LINES) {
        // Use plausibility-checked positions for arm/hip joints
        const p1 = [11, 12].includes(i) ? getPos(i) : getCheckedPos(i);
        const p2 = [11, 12].includes(j) ? getPos(j) : getCheckedPos(j);
        if (p1 && p2) {
            ctx.beginPath();
            ctx.moveTo(p1[0], p1[1]);
            ctx.lineTo(p2[0], p2[1]);
            ctx.stroke();
        }
    }

    // --- FHP assessment lines (prominent, on top) ---
    const fhpColor = isNormal
        ? 'rgba(250, 204, 21, 0.85)'     // Gold
        : 'rgba(251, 146, 60, 0.9)';      // Orange

    for (const [i, j] of CONFIG.FHP_LINES) {
        const p1 = getPos(i), p2 = getPos(j);
        if (p1 && p2) {
            const isShoulder = (i === 11 && j === 12);
            ctx.strokeStyle = isShoulder
                ? (isNormal ? 'rgba(56, 189, 248, 0.6)' : 'rgba(239, 68, 68, 0.6)')
                : fhpColor;
            ctx.lineWidth = isShoulder ? 2.0 : 2.5;
            ctx.beginPath();
            ctx.moveTo(p1[0], p1[1]);
            ctx.lineTo(p2[0], p2[1]);
            ctx.stroke();
        }
    }

    // --- Primary joint dots (nose, ears, neck, shoulders) ---
    for (const idx of CONFIG.PRIMARY_JOINTS) {
        const pos = getPos(idx);
        if (!pos) continue;
        const [px, py] = pos;

        ctx.beginPath();
        ctx.arc(px, py, 5, 0, Math.PI * 2);
        ctx.fillStyle = CONFIG.JOINT_COLORS[idx] || '#e2e8f0';
        ctx.fill();
        ctx.strokeStyle = 'rgba(0,0,0,0.4)';
        ctx.lineWidth = 1.2;
        ctx.stroke();
    }

    // --- Secondary joint dots (elbows, wrists, hips — smaller, with plausibility check) ---
    for (const idx of CONFIG.SECONDARY_JOINTS) {
        const pos = getCheckedPos(idx);
        if (!pos) continue;
        const [px, py] = pos;

        ctx.beginPath();
        ctx.arc(px, py, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = CONFIG.JOINT_COLORS[idx] || (isNormal ? '#4ade80' : '#f87171');
        ctx.globalAlpha = 0.7;
        ctx.fill();
        ctx.globalAlpha = 1.0;
        ctx.strokeStyle = 'rgba(0,0,0,0.3)';
        ctx.lineWidth = 1;
        ctx.stroke();
    }
}

// ============================================
// UI Updates
// ============================================

function updateResult(result) {
    const classEl = $('resultClass');
    const probFillNormal = $('probFillNormal');
    const probFillFHP = $('probFillFHP');
    const probValNormal = $('probValNormal');
    const probValFHP = $('probValFHP');
    const severityBadge = $('severityBadge');

    if (!result.detected) {
        classEl.textContent = 'No Person Detected';
        classEl.className = 'result-class';
        probFillNormal.style.width = '0%';
        probFillFHP.style.width = '0%';
        probValNormal.textContent = '\u2014';
        probValFHP.textContent = '\u2014';
        severityBadge.textContent = '\u2014';
        severityBadge.className = 'severity-badge';
        return;
    }

    const cls = result.classification;
    const probs = result.probabilities || { Normal: 0.5, FHP: 0.5 };
    const rawNormal = probs.Normal * 100;
    const rawFhp = probs.FHP * 100;

    // EMA smooth probability bars
    state.smoothNormalPct = state.smoothNormalPct * 0.7 + rawNormal * 0.3;
    state.smoothFhpPct = state.smoothFhpPct * 0.7 + rawFhp * 0.3;
    const normalPct = state.smoothNormalPct;
    const fhpPct = state.smoothFhpPct;

    // FHP history for chart
    if (state.startTime) {
        const timeSec = (Date.now() - state.startTime) / 1000;
        state.fhpHistory.push({ time: timeSec, fhpPct: rawFhp });
        if (state.fhpHistory.length > state.FHP_HISTORY_MAX) state.fhpHistory.shift();
        drawFhpChart();
    }

    // Classification label
    if (cls === 'FHP') {
        classEl.textContent = 'FHP Detected';
        classEl.className = 'result-class fhp';
    } else {
        classEl.textContent = 'Good Posture';
        classEl.className = 'result-class normal';
    }

    // Probability bars
    probFillNormal.style.width = `${normalPct}%`;
    probFillFHP.style.width = `${fhpPct}%`;
    probValNormal.textContent = `${normalPct.toFixed(1)}%`;
    probValFHP.textContent = `${fhpPct.toFixed(1)}%`;

    // Severity
    const severity = result.cva_severity || 'normal';
    const labels = { normal: 'Normal', moderate: 'Moderate', severe: 'Severe' };
    severityBadge.textContent = labels[severity] || severity;
    severityBadge.className = `severity-badge severity-${severity}`;

    // Frame counters
    state.totalFrames++;
    if (cls === 'FHP') state.fhpFrames++;
    else state.normalFrames++;
}

function updateAngles(result) {
    if (!result.angles) return;

    const mapping = {
        'cva_proxy_angle': 'angleCVA',
        'shoulder_rounding_angle': 'angleShoulder',
        'head_tilt_angle': 'angleHeadTilt',
        'neck_flexion_angle': 'angleNeckFlex',
        'head_forward_displacement': 'angleHeadFwd',
        'shoulder_symmetry_angle': 'angleSymmetry',
    };

    for (const [key, elemId] of Object.entries(mapping)) {
        const el = $(elemId);
        if (!el || result.angles[key] === undefined) continue;
        const val = result.angles[key];
        el.textContent = `${val.toFixed(1)}\u00B0`;

        // Color code CVA
        if (key === 'cva_proxy_angle') {
            el.style.color = val > 49 ? '#22c55e' : val > 44 ? '#f59e0b' : '#ef4444';
        }
    }
}

function updateStats() {
    if (state.startTime) {
        const elapsed = Math.floor((Date.now() - state.startTime) / 1000);
        const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
        const secs = (elapsed % 60).toString().padStart(2, '0');
        $('statTime').textContent = `${mins}:${secs}`;
    }

    if (state.detectionFpsEma > 0) {
        $('statFPS').textContent = Math.round(state.detectionFpsEma);
    }

    const total = state.totalFrames || 1;
    $('statNormal').textContent = `${Math.round(state.normalFrames / total * 100)}%`;
    $('statFHP').textContent = `${Math.round(state.fhpFrames / total * 100)}%`;
}

let _statsIntervalId = null;
function startStatsTimer() {
    if (_statsIntervalId) clearInterval(_statsIntervalId);
    _statsIntervalId = setInterval(() => {
        if (state.detecting) updateStats();
    }, 1000);
}

function updateTimeline(result) {
    if (!result.detected) return;
    const cls = result.classification === 'FHP' ? 'fhp' : 'normal';
    state.timeline.push(cls);

    const container = $('postureTimeline');
    if (state.timeline.length > CONFIG.TIMELINE_MAX_BARS) {
        state.timeline.shift();
        if (container.firstChild) container.removeChild(container.firstChild);
    }

    const bar = document.createElement('div');
    bar.className = `timeline-bar ${cls}`;
    container.appendChild(bar);
}

function checkAlert(result) {
    if (!result.detected || result.classification !== 'FHP') return;
    if ((result.confidence || 0) < 0.7) return;

    const now = Date.now();
    if (now - state.lastAlertTime < CONFIG.ALERT_COOLDOWN_MS) return;
    state.lastAlertTime = now;

    const alertEl = $('alertOverlay');
    alertEl.classList.remove('hidden');
    setTimeout(() => alertEl.classList.add('hidden'), CONFIG.ALERT_DURATION_MS);
}

// ============================================
// Keyboard Shortcuts
// ============================================

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') stopDetection();
    if (e.key === ' ' && !state.detecting) {
        e.preventDefault();
        startDetection();
    }
});

// ============================================
// Technical Panel
// ============================================

function updateTechPanel(result) {
    if (!result.score_breakdown) return;
    const b = result.score_breakdown;

    $('scoreNeck').style.width = `${Math.min((b.neck / 30) * 100, 100)}%`;
    $('scoreEar').style.width = `${Math.min((b.ear / 25) * 100, 100)}%`;
    $('scoreCVA').style.width = `${Math.min((b.cva / 25) * 100, 100)}%`;
    $('scoreNose').style.width = `${Math.min((b.nose / 10) * 100, 100)}%`;
    $('scoreShoulder').style.width = `${Math.min((b.shoulder / 10) * 100, 100)}%`;
    $('scoreTotal').style.width = `${Math.min(b.total, 100)}%`;

    $('scoreNeckVal').textContent = `+${b.neck.toFixed(1)}`;
    $('scoreEarVal').textContent = `+${b.ear.toFixed(1)}`;
    $('scoreCVAVal').textContent = `+${b.cva.toFixed(1)}`;
    $('scoreNoseVal').textContent = `+${b.nose.toFixed(1)}`;
    $('scoreShoulderVal').textContent = `+${b.shoulder.toFixed(1)}`;
    $('scoreTotalVal').textContent = b.total.toFixed(1);

    if (result.probabilities && result.probabilities.FHP !== undefined) {
        $('smoothedVal').textContent = `val = ${(result.probabilities.FHP * 100).toFixed(1)}%`;
    }
}

let techPanelOpen = true;
function toggleTechPanel() {
    techPanelOpen = !techPanelOpen;
    const content = $('techContent');
    const btn = $('techToggle');
    if (techPanelOpen) {
        content.classList.remove('hidden');
        btn.innerHTML = '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>';
    } else {
        content.classList.add('hidden');
        btn.innerHTML = '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>';
    }
}

// ============================================
// FHP Over-Time Chart
// ============================================

function drawFhpChart() {
    const canvas = $('fhpChart');
    if (!canvas) return;
    const c = canvas.getContext('2d');
    const W = canvas.width = canvas.parentElement.clientWidth || 300;
    const H = canvas.height = 90;
    const pad = { top: 8, right: 8, bottom: 16, left: 28 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    c.clearRect(0, 0, W, H);

    // Background
    c.fillStyle = 'rgba(0,0,0,0.2)';
    c.beginPath();
    if (c.roundRect) c.roundRect(0, 0, W, H, 4);
    else c.rect(0, 0, W, H);
    c.fill();

    const data = state.fhpHistory;
    if (data.length < 2) {
        c.fillStyle = 'rgba(255,255,255,0.15)';
        c.font = '10px Inter, system-ui, sans-serif';
        c.textAlign = 'center';
        c.fillText('Collecting data\u2026', W / 2, H / 2 + 3);
        return;
    }

    // Threshold line @ 40%
    const threshY = pad.top + plotH * (1 - 40 / 100);
    c.strokeStyle = 'rgba(251, 191, 36, 0.2)';
    c.setLineDash([3, 3]);
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(pad.left, threshY);
    c.lineTo(pad.left + plotW, threshY);
    c.stroke();
    c.setLineDash([]);

    // Y-axis labels
    c.fillStyle = 'rgba(255,255,255,0.25)';
    c.font = '9px JetBrains Mono, monospace';
    c.textAlign = 'right';
    c.fillText('100%', pad.left - 4, pad.top + 7);
    c.fillText('40%', pad.left - 4, threshY + 3);
    c.fillText('0%', pad.left - 4, pad.top + plotH + 1);

    const tMin = data[0].time;
    const tMax = data[data.length - 1].time;
    const tRange = Math.max(tMax - tMin, 1);

    const toX = (t) => pad.left + ((t - tMin) / tRange) * plotW;
    const toY = (v) => pad.top + plotH * (1 - v / 100);

    // Gradient fill
    const grad = c.createLinearGradient(0, pad.top, 0, pad.top + plotH);
    grad.addColorStop(0, 'rgba(239, 68, 68, 0.2)');
    grad.addColorStop(0.4, 'rgba(239, 68, 68, 0.05)');
    grad.addColorStop(1, 'rgba(34, 197, 94, 0.01)');

    c.beginPath();
    c.moveTo(toX(data[0].time), pad.top + plotH);
    for (const pt of data) c.lineTo(toX(pt.time), toY(pt.fhpPct));
    c.lineTo(toX(data[data.length - 1].time), pad.top + plotH);
    c.closePath();
    c.fillStyle = grad;
    c.fill();

    // Line
    c.beginPath();
    c.moveTo(toX(data[0].time), toY(data[0].fhpPct));
    for (let i = 1; i < data.length; i++) {
        const prev = data[i - 1], curr = data[i];
        const cpx = (toX(prev.time) + toX(curr.time)) / 2;
        c.bezierCurveTo(cpx, toY(prev.fhpPct), cpx, toY(curr.fhpPct), toX(curr.time), toY(curr.fhpPct));
    }
    c.strokeStyle = '#f87171';
    c.lineWidth = 1.5;
    c.stroke();

    // Endpoint dot
    const last = data[data.length - 1];
    c.beginPath();
    c.arc(toX(last.time), toY(last.fhpPct), 3, 0, Math.PI * 2);
    c.fillStyle = last.fhpPct > 40 ? '#ef4444' : '#4ade80';
    c.fill();
    c.strokeStyle = 'rgba(255,255,255,0.6)';
    c.lineWidth = 1;
    c.stroke();

    // Time label
    c.fillStyle = 'rgba(255,255,255,0.2)';
    c.font = '8px JetBrains Mono, monospace';
    c.textAlign = 'center';
    const durSec = Math.round(tRange);
    const label = durSec < 60 ? `${durSec}s` : `${Math.floor(durSec / 60)}m ${durSec % 60}s`;
    c.fillText(label, pad.left + plotW / 2, H - 2);
}

// ============================================
// Init
// ============================================

window.addEventListener('load', () => {
    fetch(`${CONFIG.API_URL}/health`)
        .then(r => r.json())
        .then(data => console.log('API status:', data))
        .catch(() => console.warn('API not reachable at', CONFIG.API_URL));
});
