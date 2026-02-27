/**
 * FHP Detection — Web Frontend Application
 *
 * Captures webcam frames, sends to FastAPI backend,
 * displays results with skeleton overlay and stats.
 */

// ============================================
// Configuration
// ============================================

const CONFIG = {
    // API endpoint — Render backend URL (change in production)
    API_URL: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        ? 'http://localhost:8000'
        : (window.API_URL || 'https://fhp-detection-api.onrender.com'),

    // Detection settings
    FRAME_INTERVAL_MS: 200,      // 5 FPS to API (throttle)
    ALERT_COOLDOWN_MS: 8000,     // 8 seconds between alerts
    ALERT_DURATION_MS: 2000,     // Alert shows for 2 seconds
    TIMELINE_MAX_BARS: 80,       // Max bars in timeline

    // Canvas drawing — ALL 33 MediaPipe landmarks for pixel-perfect overlay
    // MediaPipe Pose Landmark indices:
    // 0=nose, 7=L_ear, 8=R_ear, 11=L_shoulder, 12=R_shoulder
    // 13=L_elbow, 14=R_elbow, 15=L_wrist, 16=R_wrist
    // 23=L_hip, 24=R_hip

    // === FHP ASSESSMENT CHAIN (the core posture lines) ===
    // These are the anatomical lines used for clinical FHP detection:
    // Head → Ear Tragus → Neck → Shoulder → Hip
    // Index 33 = virtual NECK point (midpoint of shoulders, computed by API)
    FHP_LINES: [
        [0, 7],    // Nose → Left Ear (head forward position)
        [0, 8],    // Nose → Right Ear
        [7, 33],   // Left Ear → Neck (upper neck alignment)
        [8, 33],   // Right Ear → Neck
        [33, 11],  // Neck → Left Shoulder
        [33, 12],  // Neck → Right Shoulder
        [11, 23],  // Left Shoulder → Left Hip (trunk alignment)
        [12, 24],  // Right Shoulder → Right Hip
    ],

    // Regular skeleton connections
    SKELETON_EDGES: [
        [11, 12],              // Shoulder line
        [23, 24],              // Hip line
        [11, 13], [13, 15],   // Left arm (shoulder → elbow → wrist)
        [12, 14], [14, 16],   // Right arm
    ],

    // Upper body joints to draw as dots
    UPPER_BODY_JOINTS: [0, 7, 8, 33, 11, 12, 13, 14, 15, 16, 23, 24],
    JOINT_COLORS: {
        0: '#fbbf24',  // Nose (gold)
        7: '#c084fc', 8: '#c084fc',   // Ears (purple)
        33: '#ffffff',                 // Neck (white)
        11: '#38bdf8', 12: '#38bdf8', // Shoulders (cyan)
        13: '#818cf8', 14: '#818cf8', // Elbows (indigo)
        15: '#a78bfa', 16: '#a78bfa', // Wrists (violet)
        23: '#34d399', 24: '#34d399', // Hips (green)
    },
};

// ============================================
// State
// ============================================

let state = {
    detecting: false,
    sessionId: null,
    stream: null,
    intervalId: null,
    cameraFacingMode: 'user',

    // Stats
    startTime: null,
    totalFrames: 0,
    normalFrames: 0,
    fhpFrames: 0,
    lastAlertTime: 0,
    lastResult: null,
    fpsHistory: [],
    timeline: [],
};


// ============================================
// DOM Elements
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

        // Set display canvas size to match video
        displayCanvas.width = webcam.videoWidth;
        displayCanvas.height = webcam.videoHeight;

        // Set container aspect-ratio to match webcam
        const container = document.querySelector('.video-container');
        container.style.aspectRatio = `${webcam.videoWidth} / ${webcam.videoHeight}`;

        // Start live video rendering loop (mirrored)
        function renderFrame() {
            if (!state.detecting) return;
            ctx.save();
            ctx.translate(displayCanvas.width, 0);
            ctx.scale(-1, 1);
            ctx.drawImage(webcam, 0, 0, displayCanvas.width, displayCanvas.height);
            ctx.restore();

            // Redraw skeleton on top if we have a result
            if (state.lastResult && state.lastResult.detected) {
                drawSkeletonOnCanvas(state.lastResult);
            }
            displayAnimFrame = requestAnimationFrame(renderFrame);
        }

        $('videoLoading').classList.add('hidden');
        $('statusDot').classList.add('active');

        state.detecting = true;
        state.startTime = Date.now();
        state.sessionId = crypto.randomUUID();

        // Start video render + detection loop
        renderFrame();
        state.intervalId = setInterval(captureAndDetect, CONFIG.FRAME_INTERVAL_MS);

        // Start stats timer
        updateStatsTimer();

    } catch (err) {
        console.error('Camera error:', err);
        alert('Camera access denied. Please allow camera access and try again.');
        stopDetection();
    }
}

function stopDetection() {
    state.detecting = false;

    if (state.intervalId) {
        clearInterval(state.intervalId);
        state.intervalId = null;
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
    $('detectionPanel').classList.add('hidden');
    $('hero').classList.remove('hidden');
    $('infoSection').classList.remove('hidden');

    // Clear canvas
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
    state.sessionId = crypto.randomUUID();
    $('postureTimeline').innerHTML = '';
    updateStats();
}

// ============================================
// Capture & Send Frame
// ============================================

async function captureAndDetect() {
    if (!state.detecting || webcam.readyState < 2) return;

    const frameStart = performance.now();

    // Capture frame at webcam resolution (mirrored for selfie)
    const vw = webcam.videoWidth || 640;
    const vh = webcam.videoHeight || 480;

    const captureCanvas = document.createElement('canvas');
    captureCanvas.width = vw;
    captureCanvas.height = vh;
    const captureCtx = captureCanvas.getContext('2d');

    // MIRROR the capture — same as display
    captureCtx.translate(vw, 0);
    captureCtx.scale(-1, 1);
    captureCtx.drawImage(webcam, 0, 0, vw, vh);
    captureCtx.setTransform(1, 0, 0, 1, 0, 0);

    // Store capture dimensions
    state.captureWidth = vw;
    state.captureHeight = vh;

    // Convert to base64 JPEG
    const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.7);

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

        // Update FPS
        const elapsed = performance.now() - frameStart;
        state.fpsHistory.push(1000 / elapsed);
        if (state.fpsHistory.length > 30) state.fpsHistory.shift();

        updateResult(result);
        updateAngles(result);
        updateTechPanel(result);
        // Skeleton is drawn in renderFrame loop
        updateStats();
        updateTimeline(result);
        checkAlert(result);

    } catch (err) {
        console.warn('Detection error:', err.message);
    }
}

// ============================================
// UI Updates
// ============================================

function updateResult(result) {
    const classEl = $('resultClass');
    const confFill = $('confidenceFill');
    const confVal = $('confidenceValue');
    const card = $('resultCard');

    if (!result.detected) {
        classEl.textContent = 'No Person';
        classEl.className = 'result-class';
        confFill.style.width = '0%';
        confVal.textContent = '—';
        return;
    }

    const cls = result.classification;
    const conf = result.confidence || 0;

    classEl.textContent = cls === 'FHP' ? '⚠️ FHP Detected' : '✅ Good Posture';
    classEl.className = `result-class ${cls === 'FHP' ? 'fhp' : 'normal'}`;

    confFill.style.width = `${conf * 100}%`;
    confFill.style.background = cls === 'FHP'
        ? 'linear-gradient(90deg, #f87171, #ef4444)'
        : 'linear-gradient(90deg, #4ade80, #22c55e)';
    confVal.textContent = `${(conf * 100).toFixed(1)}%`;

    // Update stats
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
        if (el && result.angles[key] !== undefined) {
            const val = result.angles[key];
            el.textContent = `${val.toFixed(1)}°`;

            // Color code based on severity
            if (key === 'cva_proxy_angle') {
                el.style.color = val < 44 ? '#ef4444' : val < 49 ? '#f59e0b' : '#22c55e';
            }
        }
    }
}

function drawSkeletonOnCanvas(result) {
    if (!result.detected || !result.keypoints_2d) return;

    const kps = result.keypoints_2d;
    // Coordinates are in capture canvas space = displayCanvas space (both = webcam resolution)
    // No scaling needed since displayCanvas.width = webcam.videoWidth

    const isNormal = result.classification !== 'FHP';

    // Helper to get position (returns null if keypoint invisible)
    const getPos = (idx) => {
        if (idx < kps.length && kps[idx]) {
            return [kps[idx][0], kps[idx][1]];
        }
        return null;
    };

    // 1. Draw FHP ASSESSMENT CHAIN (prominent, thicker)
    const fhpColor = isNormal ? 'rgba(250, 204, 21, 0.9)' : 'rgba(251, 146, 60, 0.95)';
    ctx.strokeStyle = fhpColor;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.setLineDash([]);

    for (const [i, j] of CONFIG.FHP_LINES) {
        const p1 = getPos(i), p2 = getPos(j);
        if (p1 && p2) {
            ctx.beginPath();
            ctx.moveTo(p1[0], p1[1]);
            ctx.lineTo(p2[0], p2[1]);
            ctx.stroke();
        }
    }

    // 2. Draw regular skeleton edges (thinner)
    const boneColor = isNormal ? 'rgba(34, 197, 94, 0.6)' : 'rgba(239, 68, 68, 0.6)';
    ctx.strokeStyle = boneColor;
    ctx.lineWidth = 2;

    for (const [i, j] of CONFIG.SKELETON_EDGES) {
        const p1 = getPos(i), p2 = getPos(j);
        if (p1 && p2) {
            ctx.beginPath();
            ctx.moveTo(p1[0], p1[1]);
            ctx.lineTo(p2[0], p2[1]);
            ctx.stroke();
        }
    }

    // 3. Draw joints
    const fhpKeyJoints = new Set([0, 7, 8, 33, 11, 12]);
    for (const idx of CONFIG.UPPER_BODY_JOINTS) {
        const pos = getPos(idx);
        if (pos) {
            const [px, py] = pos;
            const radius = fhpKeyJoints.has(idx) ? 6 : 4;

            ctx.beginPath();
            ctx.arc(px, py, radius, 0, Math.PI * 2);
            ctx.fillStyle = CONFIG.JOINT_COLORS[idx] || (isNormal ? '#4ade80' : '#f87171');
            ctx.fill();

            ctx.beginPath();
            ctx.arc(px, py, radius, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(255,255,255,0.7)';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
    }
}

function updateStats() {
    // Duration
    if (state.startTime) {
        const elapsed = Math.floor((Date.now() - state.startTime) / 1000);
        const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
        const secs = (elapsed % 60).toString().padStart(2, '0');
        $('statTime').textContent = `${mins}:${secs}`;
    }

    // FPS
    if (state.fpsHistory.length > 0) {
        const avgFps = state.fpsHistory.reduce((a, b) => a + b) / state.fpsHistory.length;
        $('statFPS').textContent = Math.round(avgFps);
    }

    // Posture percentages
    const total = state.totalFrames || 1;
    $('statNormal').textContent = `${Math.round(state.normalFrames / total * 100)}%`;
    $('statFHP').textContent = `${Math.round(state.fhpFrames / total * 100)}%`;
}

function updateStatsTimer() {
    setInterval(() => {
        if (state.detecting) updateStats();
    }, 1000);
}

function updateTimeline(result) {
    if (!result.detected) return;

    const cls = result.classification === 'FHP' ? 'fhp' : 'normal';
    state.timeline.push(cls);

    if (state.timeline.length > CONFIG.TIMELINE_MAX_BARS) {
        state.timeline.shift();
    }

    const container = $('postureTimeline');
    container.innerHTML = state.timeline
        .map(c => `<div class="timeline-bar ${c}"></div>`)
        .join('');
}

function checkAlert(result) {
    if (!result.detected || result.classification !== 'FHP') return;
    if ((result.confidence || 0) < 0.7) return;

    const now = Date.now();
    if (now - state.lastAlertTime < CONFIG.ALERT_COOLDOWN_MS) return;

    state.lastAlertTime = now;

    const alertEl = $('alertOverlay');
    alertEl.classList.remove('hidden');

    setTimeout(() => {
        alertEl.classList.add('hidden');
    }, CONFIG.ALERT_DURATION_MS);
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

    // Update widths (percentages of max possible)
    $('scoreNeck').style.width = `${Math.min((b.neck / 35.0) * 100, 100)}%`;
    $('scoreEar').style.width = `${Math.min((b.ear / 25.0) * 100, 100)}%`;
    $('scoreCVA').style.width = `${Math.min((b.cva / 25.0) * 100, 100)}%`;
    $('scoreNose').style.width = `${Math.min((b.nose / 10.0) * 100, 100)}%`;
    $('scoreShoulder').style.width = `${Math.min((b.shoulder / 15.0) * 100, 100)}%`;
    $('scoreTotal').style.width = `${Math.min(b.total, 100)}%`;

    // Update text values
    $('scoreNeckVal').textContent = `+${b.neck.toFixed(1)}`;
    $('scoreEarVal').textContent = `+${b.ear.toFixed(1)}`;
    $('scoreCVAVal').textContent = `+${b.cva.toFixed(1)}`;
    $('scoreNoseVal').textContent = `+${b.nose.toFixed(1)}`;
    $('scoreShoulderVal').textContent = `+${b.shoulder.toFixed(1)}`;
    $('scoreTotalVal').textContent = b.total.toFixed(1);

    // Update EMA smoothed value text
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
        btn.textContent = '▼';
    } else {
        content.classList.add('hidden');
        btn.textContent = '▶';
    }
}

// ============================================
// On Load
// ============================================

window.addEventListener('load', () => {
    // Check API health
    fetch(`${CONFIG.API_URL}/health`)
        .then(r => r.json())
        .then(data => {
            console.log('API status:', data);
        })
        .catch(() => {
            console.warn('API not reachable at', CONFIG.API_URL);
        });
});
