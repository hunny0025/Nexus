/* ==========================================================================
   NEXUS Frontend Logic (Vanilla JS)
   ========================================================================== */

// Configuration
const BASE_URL = window.location.origin;
const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`;

// Global State
let state = {
    stations: [],
    tracks: [],
    trains: [],
    activeFaults: {},
    incidents: [],
    pendingEscalations: [],
    systemStats: {},
    activeTab: 'dashboard',
    selectedIncidentId: null,
    wsClientCount: 0
};

// Chart Instance
let learningChart = null;

// Coordinates Projection Constants (mapping India railway stations to 800x550 SVG box)
const projection = {
    minLat: 11.5,
    maxLat: 29.5,
    minLng: 71.5,
    maxLng: 86.5,
    width: 800,
    height: 520,
    
    project: function(lat, lng) {
        // x increases from left to right as lng increases
        const x = ((lng - this.minLng) / (this.maxLng - this.minLng)) * this.width;
        // y increases from top to bottom as lat decreases
        const y = (1 - (lat - this.minLat) / (this.maxLat - this.minLat)) * this.height;
        return { x, y };
    }
};

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

async function initApp() {
    setupTabNavigation();
    setupDrawer();
    setupForms();
    
    // Connect websocket
    connectWebSocket();
    
    // Initial fetch of data
    await loadInitialData();
}

async function loadInitialData() {
    showToast('info', 'Connecting', 'Loading railway database and system metrics...');
    try {
        await Promise.all([
            fetchNetworkState(),
            fetchSystemStats(),
            fetchIncidents(),
            fetchPendingEscalations(),
            fetchDemoStatus()
        ]);
        
        // Render initial topology & charts
        renderRailwayMap();
        populateAssetDropdowns();
        renderNetworkTables();
        renderIncidentsLog();
        initLearningCurveChart();
        
        showToast('success', 'NEXUS Operational', 'Operations dashboard loaded successfully.');
    } catch (err) {
        console.error("Error loading initial data", err);
        showToast('error', 'Initialization Error', 'Failed to load some system states. Please check FastAPI logs.');
    }
}

// -------------------------------------------------------------------------
// API Fetch Handlers
// -------------------------------------------------------------------------

async function fetchNetworkState() {
    const res = await fetch(`${BASE_URL}/api/network/state`);
    const data = await res.json();
    state.stations = data.stations || [];
    state.tracks = data.tracks || [];
    state.trains = data.trains || [];
}

async function fetchSystemStats() {
    const res = await fetch(`${BASE_URL}/api/analytics/system-stats`);
    state.systemStats = await res.json();
    updateStatsGrid();
}

async function fetchIncidents() {
    const res = await fetch(`${BASE_URL}/api/incidents`);
    const data = await res.json();
    state.incidents = data.incidents || [];
}

async function fetchPendingEscalations() {
    const res = await fetch(`${BASE_URL}/api/interventions/pending`);
    const data = await res.json();
    state.pendingEscalations = data.pending || [];
    renderPendingEscalations();
}

async function fetchDemoStatus() {
    try {
        const res = await fetch(`${BASE_URL}/api/demo/status`);
        const data = await res.json();
        state.activeFaults = data.active_faults || {};
        state.wsClientCount = data.ws_clients_count || 0;
        
        // Update stats
        const activeFaultsCount = Object.keys(state.activeFaults).length;
        document.getElementById('inject-location').disabled = false;
    } catch (e) {
        console.warn("Failed to get demo status", e);
    }
}

// -------------------------------------------------------------------------
// WebSocket Event Handler
// -------------------------------------------------------------------------

function connectWebSocket() {
    const wsStatusDot = document.getElementById('ws-status');
    const wsStatusText = document.getElementById('ws-status-text');
    
    wsStatusDot.className = 'connection-status connecting';
    wsStatusText.textContent = 'Websocket Connecting';
    
    const ws = new WebSocket(WS_URL);
    
    ws.onopen = () => {
        wsStatusDot.className = 'connection-status connected';
        wsStatusText.textContent = 'Websocket Live';
        console.log("WebSocket connection established");
        // Keep alive
        setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send("PING");
            }
        }, 15000);
    };
    
    ws.onmessage = async (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === "PONG") return;
            
            console.log("WebSocket event received", data);
            
            if (data.type === "ORCHESTRATOR_ALERT") {
                showToast('warning', 'Disruption Detected!', `Anomaly confirmed at ${data.anomaly_location} (Score: ${data.anomaly_score.toFixed(2)})`);
                
                // Refresh states
                await fetchNetworkState();
                await fetchSystemStats();
                await fetchIncidents();
                await fetchPendingEscalations();
                
                // Rerender dashboard
                renderRailwayMap();
                renderIncidentsLog();
                renderNetworkTables();
                updateLearningChart();
                
                // If a pending escalation is present, highlight the card
                const pendingCard = document.getElementById('stat-pending-card');
                if (pendingCard) pendingCard.classList.add('active');
                
            } else if (data.type === "FAULT_INJECTED") {
                showToast('info', 'Fault Injected', `Telemetry failure injected on ${data.location_id} (${data.sensors.join(', ')})`);
                await fetchDemoStatus();
                await fetchNetworkState();
                renderRailwayMap();
                
            } else if (data.type === "DEMO_RESET") {
                showToast('success', 'Demo Reset', 'System state reset and Neo4j database successfully re-seeded.');
                
                // Reload everything
                await loadInitialData();
                
                // Remove highlight from pending card
                const pendingCard = document.getElementById('stat-pending-card');
                if (pendingCard) pendingCard.classList.remove('active');
            }
        } catch (e) {
            console.error("Error processing websocket message", e);
        }
    };
    
    ws.onclose = () => {
        wsStatusDot.className = 'connection-status disconnected';
        wsStatusText.textContent = 'Websocket Closed';
        console.log("WebSocket connection closed. Retrying in 5 seconds...");
        setTimeout(connectWebSocket, 5000);
    };
    
    ws.onerror = (err) => {
        console.error("WebSocket error observed", err);
        ws.close();
    };
}

// -------------------------------------------------------------------------
// Tab Navigation
// -------------------------------------------------------------------------

function setupTabNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Remove active class from nav items and tab content panels
            navItems.forEach(n => n.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked nav item
            item.classList.add('active');
            
            // Add active class to corresponding content panel
            const tabId = item.getAttribute('data-tab');
            const tabEl = document.getElementById(`tab-${tabId}`);
            if (tabEl) {
                tabEl.classList.add('active');
            }
            
            state.activeTab = tabId;
            
            // Trigger specific page updates
            if (tabId === 'analytics-insights') {
                setTimeout(updateLearningChart, 100);
            }
        });
    });
}

// -------------------------------------------------------------------------
// Update Stats Grid
// -------------------------------------------------------------------------

function updateStatsGrid() {
    document.getElementById('stat-auto-rate').textContent = `${state.systemStats.autonomous_rate_pct || 0.0}%`;
    document.getElementById('stat-delay-reduction').textContent = `${state.systemStats.avg_delay_reduction_pct || 0.0}%`;
    document.getElementById('stat-incidents-count').textContent = state.systemStats.total_incidents_handled || 0;
    
    const pendingCount = state.systemStats.pending_escalations || 0;
    document.getElementById('stat-pending-count').textContent = pendingCount;
    
    const pendingCard = document.getElementById('stat-pending-card');
    const pendingLabel = document.getElementById('stat-pending-label');
    
    if (pendingCount > 0) {
        pendingCard.classList.add('active');
        pendingLabel.className = 'stat-trend warning-text flashing';
        pendingLabel.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> ACTION REQUIRED';
    } else {
        pendingCard.classList.remove('active');
        pendingLabel.className = 'stat-trend positive';
        pendingLabel.innerHTML = '<i class="fa-solid fa-check"></i> Grid Secure';
    }
}

// -------------------------------------------------------------------------
// Render Railway SVG Topology Map
// -------------------------------------------------------------------------

function renderRailwayMap() {
    const svgTracks = document.getElementById('svg-tracks');
    const svgStations = document.getElementById('svg-stations');
    const svgTrains = document.getElementById('svg-trains');
    
    // Clear existing SVG groups
    svgTracks.innerHTML = '';
    svgStations.innerHTML = '';
    svgTrains.innerHTML = '';
    
    const tooltip = document.getElementById('map-tooltip');
    
    // Map stations for coordinate lookup
    const stationCoords = {};
    state.stations.forEach(station => {
        const coords = projection.project(station.lat, station.lng);
        stationCoords[station.id] = coords;
    });

    // 1. Draw tracks (edges)
    state.tracks.forEach(track => {
        const fromCoord = stationCoords[track.from];
        const toCoord = stationCoords[track.to];
        
        if (!fromCoord || !toCoord) return;
        
        const isAnomalous = state.activeFaults[track.id] !== undefined;
        
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", fromCoord.x);
        line.setAttribute("y1", fromCoord.y);
        line.setAttribute("x2", toCoord.x);
        line.setAttribute("y2", toCoord.y);
        
        let className = "svg-track-line";
        if (isAnomalous) className += " anomalous";
        line.setAttribute("class", className);
        line.setAttribute("id", `svg-track-${track.id}`);
        
        // Interactive tooltips on hover
        line.addEventListener('mousemove', (e) => {
            const faults = state.activeFaults[track.id];
            let faultInfo = "";
            if (faults) {
                faultInfo = `<div class="red-text mt-1"><strong>⚠️ Anomaly Injected:</strong> ${faults.join(', ')}</div>`;
            }
            
            tooltip.innerHTML = `
                <h4>Track Section: ${track.id}</h4>
                <div><strong>From:</strong> ${track.from} ➔ <strong>To:</strong> ${track.to}</div>
                <div><strong>Type:</strong> ${track.track_type.replace('_', ' ')}</div>
                <div><strong>Max Speed:</strong> ${track.max_speed_kmph} km/h</div>
                <div><strong>Distance:</strong> ${track.distance_km} km</div>
                ${faultInfo}
            `;
            tooltip.style.opacity = 1;
            tooltip.style.left = `${e.pageX - document.getElementById('map-parent').getBoundingClientRect().left + 15}px`;
            tooltip.style.top = `${e.pageY - document.getElementById('map-parent').getBoundingClientRect().top + 15}px`;
        });
        
        line.addEventListener('mouseleave', () => {
            tooltip.style.opacity = 0;
        });
        
        line.addEventListener('click', () => {
            document.getElementById('inject-location').value = track.id;
        });
        
        svgTracks.appendChild(line);
    });

    // 2. Draw stations (nodes)
    state.stations.forEach(station => {
        const coord = stationCoords[station.id];
        if (!coord) return;
        
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        group.setAttribute("class", "svg-station-group");
        
        // Node circle
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", coord.x);
        circle.setAttribute("cy", coord.y);
        circle.setAttribute("r", station.is_junction ? "6" : "4.5");
        
        let className = "svg-station-node";
        if (station.is_junction) className += " junction";
        circle.setAttribute("class", className);
        
        // Text label
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", coord.x + 8);
        text.setAttribute("y", coord.y + 3);
        text.setAttribute("class", "svg-station-label");
        text.textContent = station.id;
        
        group.appendChild(circle);
        group.appendChild(text);
        
        // Station hover tooltips
        group.addEventListener('mousemove', (e) => {
            tooltip.innerHTML = `
                <h4>${station.name} (${station.id})</h4>
                <div><strong>Zone:</strong> ${station.zone}</div>
                <div><strong>Platforms:</strong> ${station.platform_count}</div>
                <div><strong>Junction:</strong> ${station.is_junction ? "Yes" : "No"}</div>
            `;
            tooltip.style.opacity = 1;
            tooltip.style.left = `${e.pageX - document.getElementById('map-parent').getBoundingClientRect().left + 15}px`;
            tooltip.style.top = `${e.pageY - document.getElementById('map-parent').getBoundingClientRect().top + 15}px`;
        });
        
        group.addEventListener('mouseleave', () => {
            tooltip.style.opacity = 0;
        });
        
        svgStations.appendChild(group);
    });

    // 3. Draw trains (moving objects)
    state.trains.forEach(train => {
        const currentCoord = stationCoords[train.current_station];
        const nextCoord = stationCoords[train.next_station];
        
        if (!currentCoord) return;
        
        let tx = currentCoord.x;
        let ty = currentCoord.y;
        
        // Interpolate position along the track if the train is moving (next_station exists)
        if (nextCoord) {
            // Place 40% of the way along the line for visual separation
            tx = currentCoord.x + 0.4 * (nextCoord.x - currentCoord.x);
            ty = currentCoord.y + 0.4 * (nextCoord.y - currentCoord.y);
        }
        
        const marker = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        marker.setAttribute("cx", tx);
        marker.setAttribute("cy", ty);
        marker.setAttribute("r", "5.5");
        
        let className = "svg-train-marker";
        if (train.status !== "ON_TIME") className += " delayed";
        marker.setAttribute("class", className);
        marker.setAttribute("id", `svg-train-${train.id}`);
        
        // Train hover tooltip
        marker.addEventListener('mousemove', (e) => {
            tooltip.innerHTML = `
                <h4>Train ${train.id}: ${train.name}</h4>
                <div><strong>Status:</strong> <span class="${train.status === 'ON_TIME' ? 'text-success' : 'red-text'}">${train.status}</span></div>
                <div><strong>Speed:</strong> ${train.speed_kmph} km/h</div>
                <div><strong>Passengers:</strong> ${train.passenger_count}</div>
                <div><strong>Position:</strong> ${train.current_station} ${train.next_station ? '➔ ' + train.next_station : '(Stopped)'}</div>
            `;
            tooltip.style.opacity = 1;
            tooltip.style.left = `${e.pageX - document.getElementById('map-parent').getBoundingClientRect().left + 15}px`;
            tooltip.style.top = `${e.pageY - document.getElementById('map-parent').getBoundingClientRect().top + 15}px`;
        });
        
        marker.addEventListener('mouseleave', () => {
            tooltip.style.opacity = 0;
        });
        
        marker.addEventListener('click', () => {
            document.getElementById('inject-location').value = train.id;
        });
        
        svgTrains.appendChild(marker);
    });
}

// -------------------------------------------------------------------------
// Populate Dropdowns & Lists
// -------------------------------------------------------------------------

function populateAssetDropdowns() {
    const select = document.getElementById('inject-location');
    
    // Clear options but keep default
    select.innerHTML = '<option value="">-- Select Asset (Track/Train) --</option>';
    
    // Tracks option group
    const optGroupTracks = document.createElement('optgroup');
    optGroupTracks.label = "Track Segments";
    state.tracks.forEach(track => {
        const opt = document.createElement('option');
        opt.value = track.id;
        opt.textContent = `${track.id} (${track.from} ➔ ${track.to})`;
        optGroupTracks.appendChild(opt);
    });
    
    // Trains option group
    const optGroupTrains = document.createElement('optgroup');
    optGroupTrains.label = "Active Trains";
    state.trains.forEach(train => {
        const opt = document.createElement('option');
        opt.value = train.id;
        opt.textContent = `${train.id} - ${train.name}`;
        optGroupTrains.appendChild(opt);
    });
    
    select.appendChild(optGroupTracks);
    select.appendChild(optGroupTrains);
}

// -------------------------------------------------------------------------
// Render Network Explorer Tables
// -------------------------------------------------------------------------

function renderNetworkTables() {
    // Trains Table
    const tbodyTrains = document.querySelector('#table-trains tbody');
    tbodyTrains.innerHTML = '';
    
    state.trains.forEach(t => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${t.id}</strong></td>
            <td>${t.name}</td>
            <td><span class="badge">${t.current_station}</span></td>
            <td><span class="badge">${t.next_station || 'None'}</span></td>
            <td>${t.speed_kmph} km/h</td>
            <td>${t.passenger_count}</td>
            <td><span class="stat-trend ${t.status === 'ON_TIME' ? 'positive' : 'warning-text'}">${t.status}</span></td>
        `;
        tbodyTrains.appendChild(tr);
    });

    // Stations Table
    const tbodyStations = document.querySelector('#table-stations tbody');
    tbodyStations.innerHTML = '';
    
    state.stations.forEach(s => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${s.id}</strong></td>
            <td>${s.name}</td>
            <td>${s.zone}</td>
            <td>${s.lat.toFixed(4)}, ${s.lng.toFixed(4)}</td>
            <td>${s.platform_count}</td>
            <td>${s.is_junction ? '<i class="fa-solid fa-check text-success"></i>' : '<i class="fa-solid fa-xmark text-muted"></i>'}</td>
        `;
        tbodyStations.appendChild(tr);
    });
}

// -------------------------------------------------------------------------
// Render Incident Log (Disruption Feed)
// -------------------------------------------------------------------------

function renderIncidentsLog() {
    const list = document.getElementById('incidents-feed-list');
    const tbodyHistory = document.querySelector('#table-incidents-history tbody');
    
    list.innerHTML = '';
    tbodyHistory.innerHTML = '';
    
    if (state.incidents.length === 0) {
        list.innerHTML = `
            <div class="feed-empty-state">
                <i class="fa-solid fa-circle-check text-success"></i>
                <p>All operations normal. No disruptions detected in the grid.</p>
            </div>
        `;
        tbodyHistory.innerHTML = `<tr><td colspan="5" style="text-align: center;">No recorded incidents in learning agent.</td></tr>`;
        return;
    }

    // Populate real-time list on dashboard (reverse order, latest first)
    const reversedIncidents = [...state.incidents].reverse();
    reversedIncidents.forEach(inc => {
        const item = document.createElement('div');
        
        let severityClass = "warning";
        let score = inc.cascade_accuracy;
        if (score < 0.6) severityClass = "critical";
        else if (score >= 0.85) severityClass = "resolved";
        
        const date = new Date(inc.timestamp);
        const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        
        item.className = `feed-item ${severityClass}`;
        item.innerHTML = `
            <div class="feed-item-header">
                <span class="feed-item-title">${inc.incident_id}</span>
                <span class="feed-item-time">${timeStr}</span>
            </div>
            <div class="feed-item-desc">Disruption reported on track topology. AI mitigation initiated.</div>
            <div class="feed-item-footer">
                <span>Cascade Accuracy: ${(inc.cascade_accuracy * 100).toFixed(0)}%</span>
                <span>Optim. Accuracy: ${(inc.intervention_accuracy * 100).toFixed(0)}%</span>
            </div>
        `;
        
        item.addEventListener('click', () => {
            openDrawer(inc.incident_id);
        });
        
        list.appendChild(item);
    });

    // Populate historical tab table
    state.incidents.forEach(inc => {
        const tr = document.createElement('tr');
        
        const date = new Date(inc.timestamp);
        const dateStr = date.toLocaleString();
        
        tr.innerHTML = `
            <td><strong class="text-magic cursor-pointer" onclick="openDrawer('${inc.incident_id}')">${inc.incident_id}</strong></td>
            <td>${dateStr}</td>
            <td>
                <span class="stat-trend ${inc.cascade_accuracy > 0.75 ? 'positive' : 'warning-text'}">
                    ${(inc.cascade_accuracy * 100).toFixed(1)}%
                </span>
            </td>
            <td>
                <span class="stat-trend ${inc.intervention_accuracy > 0.75 ? 'positive' : 'warning-text'}">
                    ${(inc.intervention_accuracy * 100).toFixed(1)}%
                </span>
            </td>
            <td>
                <button class="btn btn-secondary btn-small" onclick="openDrawer('${inc.incident_id}')" style="padding: 0.2rem 0.6rem; font-size: 0.7rem;">
                    <i class="fa-solid fa-eye"></i> View Details
                </button>
            </td>
        `;
        tbodyHistory.appendChild(tr);
    });
}

// -------------------------------------------------------------------------
// Render Human-in-the-Loop Pending Escalations
// -------------------------------------------------------------------------

let selectedInterventionIndex = 0;

function renderPendingEscalations() {
    const card = document.getElementById('pending-escalations-card');
    const container = document.getElementById('pending-brief-container');
    
    if (state.pendingEscalations.length === 0) {
        card.style.display = 'none';
        return;
    }
    
    card.style.display = 'block';
    container.innerHTML = '';
    
    // Handle the first pending brief (simplification for UI)
    const brief = state.pendingEscalations[0];
    
    const div = document.createElement('div');
    div.innerHTML = `
        <div class="brief-meta">
            <div class="meta-field">
                <span class="lbl">Incident ID</span>
                <span class="val text-magic">${brief.incident_id}</span>
            </div>
            <div class="meta-field">
                <span class="lbl">DBN Confidence</span>
                <span class="val red-text">${(brief.confidence * 100).toFixed(1)}%</span>
            </div>
        </div>
        
        <div class="brief-desc">
            <strong>Cascade Threat:</strong> ${brief.cascade_summary}
        </div>
        
        <div class="brief-ai-exp">
            <i class="fa-solid fa-sparkles text-magic"></i> <strong>AI Explainer:</strong> ${brief.explanation_text || "Generating analysis..."}
        </div>
        
        <span class="brief-options-title">Select Mitigation Strategy (MCTS Simulated)</span>
        <div class="intervention-options" id="options-selector">
            <!-- Simulated paths injected here -->
        </div>
        
        <button class="btn btn-primary btn-block" id="btn-approve-intervention">
            <i class="fa-solid fa-check-double"></i> Authorize & Execute Plan
        </button>
    `;
    
    container.appendChild(div);
    
    // Inject MCTS options
    const optionsSelector = document.getElementById('options-selector');
    selectedInterventionIndex = 0; // Default first option
    
    brief.top_3_options.forEach((opt, idx) => {
        const item = document.createElement('div');
        item.className = `opt-card ${idx === 0 ? 'selected' : ''}`;
        item.dataset.index = idx;
        
        const iv = opt.intervention;
        const proj = opt.projected_outcome;
        
        let desc = "";
        if (iv.type === "REROUTE") {
            desc = `Reroute train ${iv.train_id} via path: ${iv.selected_route.path.join(' ➔ ')}`;
        } else if (iv.type === "HOLD") {
            desc = `Hold train ${iv.train_id} at station ${iv.hold_station} for ${iv.estimated_hold_minutes} mins`;
        } else if (iv.type === "MAINTENANCE_DISPATCH") {
            desc = `Dispatch crew ${iv.crew_id} to repair section. (ETA: ${iv.eta_minutes} mins)`;
        } else {
            desc = `Multi-objective synergy hold & reroute (synergy bonus: +${iv.synergy_bonus || 0})`;
        }
        
        item.innerHTML = `
            <div class="opt-header">
                <span class="opt-title">${iv.type.replace('_', ' ')}</span>
                <div class="opt-badges">
                    <span class="opt-badge confidence">Sim Prob: ${(opt.confidence * 100).toFixed(0)}%</span>
                    <span class="opt-badge reduction">Delay Reduction: ${(proj.delay_reduction_pct * 100).toFixed(0)}%</span>
                </div>
            </div>
            <div class="opt-desc">${desc}</div>
        `;
        
        item.addEventListener('click', () => {
            document.querySelectorAll('#options-selector .opt-card').forEach(c => c.classList.remove('selected'));
            item.classList.add('selected');
            selectedInterventionIndex = idx;
        });
        
        optionsSelector.appendChild(item);
    });
    
    // Bind approve action
    document.getElementById('btn-approve-intervention').addEventListener('click', async () => {
        await approveIntervention(brief.incident_id, selectedInterventionIndex);
    });
}

// -------------------------------------------------------------------------
// Action Triggers: Fault, Approve, Reset
// -------------------------------------------------------------------------

async function approveIntervention(incidentId, index) {
    const btn = document.getElementById('btn-approve-intervention');
    const oldText = btn.innerHTML;
    
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Authorizing Execution...`;
    
    try {
        const res = await fetch(`${BASE_URL}/api/interventions/${incidentId}/approve`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ intervention_index: index })
        });
        const data = await res.json();
        
        if (data.error) {
            showToast('error', 'Mitigation Failed', data.error);
        } else {
            showToast('success', 'Plan Authorized', `Intervention plan executed for incident ${incidentId}. Graph sync completed.`);
            
            // Clear pending states in UI
            state.pendingEscalations = state.pendingEscalations.filter(p => p.incident_id !== incidentId);
            renderPendingEscalations();
            
            // Update other states
            await fetchSystemStats();
            await fetchIncidents();
            await fetchNetworkState();
            
            renderRailwayMap();
            renderIncidentsLog();
            renderNetworkTables();
        }
    } catch (e) {
        showToast('error', 'Execution Error', 'Server failed to process the authorization. Check logs.');
    } finally {
        btn.disabled = false;
        btn.innerHTML = oldText;
    }
}

function setupForms() {
    // Fault Ingestion Form
    const form = document.getElementById('fault-injection-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const selectLocation = document.getElementById('inject-location').value;
        if (!selectLocation) {
            showToast('warning', 'Input Required', 'Please select a valid track or train section.');
            return;
        }
        
        const checkedSensors = [];
        document.querySelectorAll('input[name="sensor_type"]:checked').forEach(cb => {
            checkedSensors.push(cb.value);
        });
        
        if (checkedSensors.length === 0) {
            showToast('warning', 'Input Required', 'Please check at least one sensor for fault injection.');
            return;
        }
        
        const btn = document.getElementById('btn-inject-fault');
        btn.disabled = true;
        btn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Injecting Anomaly...`;
        
        try {
            const res = await fetch(`${BASE_URL}/api/demo/inject-fault`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    location_id: selectLocation,
                    sensor_types: checkedSensors
                })
            });
            const data = await res.json();
            
            if (data.status === "FAULT_INJECTED") {
                showToast('success', 'Fault Active', `Injected fault at ${selectLocation}. Kalman filter bank is monitoring...`);
            } else {
                showToast('error', 'Failed Ingestion', data.detail || 'Failed to trigger fault.');
            }
        } catch (err) {
            showToast('error', 'Network Connection Refused', 'Could not establish connection to the backend.');
        } finally {
            btn.disabled = false;
            btn.innerHTML = `<i class="fa-solid fa-bolt"></i> Inject Anomaly`;
        }
    });
    
    // Reset Demo Button
    const btnReset = document.getElementById('btn-reset-demo');
    btnReset.addEventListener('click', async () => {
        btnReset.disabled = true;
        btnReset.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Resetting...`;
        
        showToast('info', 'Resetting Database', 'Clearing faults and rebuilding Neo4j graph nodes...');
        
        try {
            const res = await fetch(`${BASE_URL}/api/demo/reset`, { method: 'POST' });
            const data = await res.json();
            if (data.status === 'RESET_COMPLETED') {
                showToast('success', 'Reset Finished', 'Database successfully re-seeded, faults flushed.');
            } else {
                showToast('error', 'Reset Failed', data.detail || 'Fail to clear backend data.');
            }
        } catch (e) {
            showToast('error', 'Reset Timeout', 'Connection timed out resetting database.');
        } finally {
            btnReset.disabled = false;
            btnReset.innerHTML = `<i class="fa-solid fa-rotate-right"></i> Reset Demo`;
        }
    });
}

// -------------------------------------------------------------------------
// Slideout Explainability Drawer Panel
// -------------------------------------------------------------------------

function setupDrawer() {
    const drawer = document.getElementById('explainer-drawer');
    const btnClose = document.getElementById('btn-close-drawer');
    
    btnClose.addEventListener('click', () => {
        drawer.classList.remove('open');
    });
}

async function openDrawer(incidentId) {
    state.selectedIncidentId = incidentId;
    
    const drawer = document.getElementById('explainer-drawer');
    const loading = document.getElementById('drawer-loading');
    const content = document.getElementById('drawer-content');
    
    document.getElementById('drawer-incident-id').textContent = incidentId;
    
    // Reset visibility
    loading.style.display = 'flex';
    content.style.display = 'none';
    drawer.classList.add('open');
    
    try {
        // Fetch cascade map and AI explanation in parallel
        const [resCascade, resExplanation] = await Promise.all([
            fetch(`${BASE_URL}/api/incidents/${incidentId}/cascade`),
            fetch(`${BASE_URL}/api/incidents/${incidentId}/explanation`)
        ]);
        
        const cascadeData = await resCascade.json();
        const explanationData = await resExplanation.json();
        
        // Hide loading
        loading.style.display = 'none';
        content.style.display = 'block';
        
        // Populate static metrics inside drawer
        // Find current incident record
        const record = state.incidents.find(i => i.incident_id === incidentId);
        
        if (record) {
            document.getElementById('drawer-anomaly-score').textContent = (record.cascade_accuracy * 1.5).toFixed(2); // Mock score scaled
            document.getElementById('drawer-anomaly-location').textContent = incidentId.replace('INCIDENT_', 'TRK_') || 'TRK_NDLS_CNB';
        }
        
        // Populate explanation (converting simple markdown lists to HTML)
        let rawExp = explanationData.explanation || "Gemini Explainer unavailable for this scenario.";
        document.getElementById('drawer-explanation-text').innerHTML = formatMarkdown(rawExp);
        
        // Draw DBN cascade risk table
        const cascadeContainer = document.getElementById('drawer-cascade-map-viz');
        cascadeContainer.innerHTML = '';
        
        const cascadeMap = cascadeData.cascade_map || {};
        const sectors = Object.keys(cascadeMap);
        
        if (sectors.length === 0) {
            cascadeContainer.innerHTML = `<div class="section-desc">No cascade risk detected. Disruption contained.</div>`;
        } else {
            sectors.forEach(sec => {
                const stepProbs = cascadeMap[sec]; // E.g., { "5": 0.8, "10": 0.6, "15": 0.4 }
                
                const row = document.createElement('div');
                row.className = 'cascade-row';
                
                let pBadges = "";
                Object.keys(stepProbs).sort((a,b) => parseInt(a) - parseInt(b)).forEach(t => {
                    const prob = stepProbs[t];
                    let pClass = "low";
                    if (prob > 0.6) pClass = "high";
                    else if (prob > 0.3) pClass = "medium";
                    
                    pBadges += `<span class="step-prob ${pClass}">T+${t}m: ${(prob * 100).toFixed(0)}%</span>`;
                });
                
                row.innerHTML = `
                    <span class="cascade-node-id"><i class="fa-solid fa-shuffle text-muted mr-1"></i> ${sec}</span>
                    <div class="cascade-step-p">${pBadges}</div>
                `;
                cascadeContainer.appendChild(row);
            });
        }
        
    } catch (e) {
        console.error("Error loading drawer detail", e);
        loading.innerHTML = `<i class="fa-solid fa-circle-xmark text-danger" style="font-size:2rem"></i><p>Failed to generate Gemini explainability report.</p>`;
    }
}

// Helper to format simple markdown lists & bold text
function formatMarkdown(text) {
    let html = text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/^\s*-\s+(.*?)$/gm, '<li>$1</li>')
        .replace(/^\s*\*\s+(.*?)$/gm, '<li>$1</li>');
        
    // Wrap lists in <ul> tags
    if (html.includes('<li>')) {
        // Simple regex to group consecutive <li> elements (very basic, fits normal gemini outputs)
        html = html.replace(/(<li>.*?<\/li>)+/gs, (match) => `<ul>${match}</ul>`);
    }
    
    // Convert newlines to breaks except inside lists
    return html.split('\n').map(p => {
        if (p.trim().startsWith('<ul>') || p.trim().startsWith('</ul>') || p.trim().startsWith('<li>')) return p;
        return `<p>${p}</p>`;
    }).join('');
}

// -------------------------------------------------------------------------
// Toast System
// -------------------------------------------------------------------------

function showToast(type, title, desc) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    
    let iconClass = "fa-circle-info";
    if (type === 'success') iconClass = "fa-circle-check";
    else if (type === 'error') iconClass = "fa-circle-xmark";
    else if (type === 'warning') iconClass = "fa-triangle-exclamation";
    
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <div class="toast-icon"><i class="fa-solid ${iconClass}"></i></div>
        <div class="toast-content">
            <span class="toast-title">${title}</span>
            <span class="toast-desc">${desc}</span>
        </div>
    `;
    
    container.appendChild(toast);
    
    // Trigger animation
    setTimeout(() => {
        toast.classList.add('show');
    }, 10);
    
    // Auto remove
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 4500);
}

// -------------------------------------------------------------------------
// Chart.js Chart Implementation
// -------------------------------------------------------------------------

function initLearningCurveChart() {
    const ctx = document.getElementById('learningChart').getContext('2d');
    
    // Prepare data
    const labels = state.incidents.map(i => i.incident_id.replace('INCIDENT_', '#'));
    const cascadeAccs = state.incidents.map(i => i.cascade_accuracy * 100);
    const optimAccs = state.incidents.map(i => i.intervention_accuracy * 100);
    
    // Destroy existing chart to avoid "Canvas is already in use" errors on reset/reload
    if (learningChart) {
        learningChart.destroy();
    }
    
    learningChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Cascade Prediction Accuracy (%)',
                    data: cascadeAccs,
                    borderColor: '#a855f7',
                    backgroundColor: 'rgba(168, 85, 247, 0.1)',
                    tension: 0.3,
                    fill: true,
                    borderWidth: 2
                },
                {
                    label: 'Intervention Prediction Accuracy (%)',
                    data: optimAccs,
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6, 182, 212, 0.1)',
                    tension: 0.3,
                    fill: true,
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#94a3b8',
                        font: { family: 'Plus Jakarta Sans', size: 11 }
                    }
                }
            },
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}

function updateLearningChart() {
    if (!learningChart) return;
    
    const labels = state.incidents.map(i => i.incident_id.replace('INCIDENT_', '#'));
    const cascadeAccs = state.incidents.map(i => i.cascade_accuracy * 100);
    const optimAccs = state.incidents.map(i => i.intervention_accuracy * 100);
    
    learningChart.data.labels = labels;
    learningChart.data.datasets[0].data = cascadeAccs;
    learningChart.data.datasets[1].data = optimAccs;
    
    learningChart.update();
}
