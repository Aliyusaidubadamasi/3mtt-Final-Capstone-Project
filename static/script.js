/**
 * script.js
 * ---------
 * Client-side JavaScript logic for Crop Yield Estimator multi-page web application.
 */

// Preset Baselines
const PRESETS = {
    kaduna_maize: {
        state: 'Kaduna',
        crop: 'Maize (corn)',
        year: 2024,
        area_harvested_ha: 1200,
        rainfall_mm: 1180.5,
        avg_temp_c: 26.8,
        min_temp_c: 20.2,
        max_temp_c: 33.1,
        humidity_pct: 65.0,
        solar_radiation: 18.2,
        nitrogen_n: 80.0,
        phosphorus_p: 35.0,
        potassium_k: 25.0,
        soil_ph: 6.5,
        fertilizer_kg_ha: 95.0,
        pesticide_kg_ha: 4.2
    },
    benue_yam: {
        state: 'Benue',
        crop: 'Yams',
        year: 2024,
        area_harvested_ha: 1500,
        rainfall_mm: 1350.0,
        avg_temp_c: 27.4,
        min_temp_c: 21.5,
        max_temp_c: 33.8,
        humidity_pct: 72.0,
        solar_radiation: 17.8,
        nitrogen_n: 60.0,
        phosphorus_p: 40.0,
        potassium_k: 35.0,
        soil_ph: 6.2,
        fertilizer_kg_ha: 75.0,
        pesticide_kg_ha: 3.5
    },
    kano_rice: {
        state: 'Kano',
        crop: 'Rice',
        year: 2024,
        area_harvested_ha: 2000,
        rainfall_mm: 980.0,
        avg_temp_c: 27.8,
        min_temp_c: 19.8,
        max_temp_c: 35.2,
        humidity_pct: 55.0,
        solar_radiation: 19.5,
        nitrogen_n: 95.0,
        phosphorus_p: 45.0,
        potassium_k: 30.0,
        soil_ph: 6.8,
        fertilizer_kg_ha: 110.0,
        pesticide_kg_ha: 5.0
    },
    oyo_cassava: {
        state: 'Oyo',
        crop: 'Cassava, fresh',
        year: 2024,
        area_harvested_ha: 1800,
        rainfall_mm: 1250.0,
        avg_temp_c: 26.5,
        min_temp_c: 21.0,
        max_temp_c: 32.5,
        humidity_pct: 74.0,
        solar_radiation: 17.5,
        nitrogen_n: 50.0,
        phosphorus_p: 25.0,
        potassium_k: 40.0,
        soil_ph: 6.0,
        fertilizer_kg_ha: 60.0,
        pesticide_kg_ha: 3.0
    }
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    fetchOptions();
    fetchAnalytics();
});

// Mobile Hamburger Menu Toggle
function toggleMobileMenu() {
    const navLinks = document.getElementById('nav-links');
    const hamburger = document.getElementById('hamburger-toggle');
    if (navLinks && hamburger) {
        navLinks.classList.toggle('mobile-open');
        hamburger.classList.toggle('open');
    }
}

// Multi-Page Switcher & Auto-close Mobile Menu
function switchPage(pageId) {
    const sections = document.querySelectorAll('.page-section');
    const navBtns = document.querySelectorAll('.nav-btn');

    sections.forEach(sec => {
        sec.classList.remove('active');
    });

    navBtns.forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-target') === pageId) {
            btn.classList.add('active');
        }
    });

    const targetSection = document.getElementById(`page-${pageId}`);
    if (targetSection) {
        targetSection.classList.add('active');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // Auto-close mobile drawer menu on selection
    const navLinks = document.getElementById('nav-links');
    const hamburger = document.getElementById('hamburger-toggle');
    if (navLinks) navLinks.classList.remove('mobile-open');
    if (hamburger) hamburger.classList.remove('open');
}

// Fetch Dropdown Options from FastAPI
async function fetchOptions() {
    try {
        const response = await fetch('/api/options');
        if (!response.ok) return;

        const data = await response.json();
        const stateSelect = document.getElementById('state');
        const cropSelect = document.getElementById('crop');

        if (stateSelect && data.states) {
            stateSelect.innerHTML = data.states.map(s => `<option value="${s}" ${s === 'Kaduna' ? 'selected' : ''}>${s}</option>`).join('');
        }

        if (cropSelect && data.crops) {
            cropSelect.innerHTML = data.crops.map(c => `<option value="${c}" ${c === 'Maize (corn)' ? 'selected' : ''}>${c}</option>`).join('');
        }
    } catch (err) {
        console.warn('Could not load dynamic options, using defaults.', err);
    }
}

// Load Preset Baseline Values
function loadPreset(presetKey) {
    const data = PRESETS[presetKey];
    if (!data) return;

    for (const [key, val] of Object.entries(data)) {
        const el = document.getElementById(key);
        if (el) {
            el.value = val;
        }
    }
    switchPage('estimator');
}

// Form Submission & Inference Call
async function handlePrediction(event) {
    event.preventDefault();

    const submitBtn = document.getElementById('submit-btn');
    const form = document.getElementById('prediction-form');

    const formData = new FormData(form);
    const payload = {};

    formData.forEach((value, key) => {
        if (key === 'state' || key === 'crop') {
            payload[key] = value;
        } else if (key === 'year') {
            payload[key] = parseInt(value, 10);
        } else {
            payload[key] = parseFloat(value);
        }
    });

    submitBtn.disabled = true;
    submitBtn.innerHTML = '⏳ Computing Yield Forecast...';

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'Prediction request failed');
        }

        const result = await response.json();

        // Render Results Panel
        renderResults(result);

    } catch (err) {
        alert(`Prediction Error: ${err.message}`);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '🔮 Predict Crop Yield';
    }
}

// Render Results & Feature Importances
function renderResults(result) {
    const placeholder = document.getElementById('results-placeholder');
    const container = document.getElementById('results-container');

    placeholder.classList.add('hidden');
    container.classList.remove('hidden');

    document.getElementById('res-crop-state').textContent = `${result.crop} — ${result.state}`;

    // Animate counter values
    animateCounter('res-yield-kg', result.predicted_yield_kg_ha, 0);
    animateCounter('res-yield-tonnes', result.predicted_yield_tonnes_ha, 2);

    // Interpretation
    document.getElementById('res-interpretation').textContent = result.interpretation;

    // Feature Importances Chart
    const fiContainer = document.getElementById('fi-chart-container');
    fiContainer.innerHTML = '';

    if (result.feature_importances && result.feature_importances.length > 0) {
        // Find max importance to compute relative percentage width
        const maxImp = Math.max(...result.feature_importances.map(f => f.importance));

        result.feature_importances.forEach(f => {
            const pct = Math.round((f.importance / maxImp) * 100);
            const impPctStr = (f.importance * 100).toFixed(1) + '%';

            const item = document.createElement('div');
            item.className = 'fi-item';
            item.innerHTML = `
                <div class="fi-label-row">
                    <span class="fi-name">${cleanFeatureName(f.feature)}</span>
                    <span class="fi-pct">${impPctStr}</span>
                </div>
                <div class="fi-track">
                    <div class="fi-fill" style="width: 0%;"></div>
                </div>
            `;
            fiContainer.appendChild(item);

            // Animate bar width
            setTimeout(() => {
                item.querySelector('.fi-fill').style.width = `${pct}%`;
            }, 50);
        });
    }
}

function cleanFeatureName(name) {
    const map = {
        'state_Global Baseline': 'State Baseline',
        'crop_Mixed Grain Baseline': 'Crop Commodity Baseline',
        'phosphorus_p': 'Soil Phosphorus (P)',
        'crop_Yams': 'Crop Commodity (Yam)',
        'area_harvested_ha': 'Harvested Area (ha)',
        'crop_Cassava, fresh': 'Crop Commodity (Cassava)',
        'max_temp_c': 'Maximum Temp (°C)',
        'avg_temp_c': 'Average Temp (°C)',
        'min_temp_c': 'Minimum Temp (°C)',
        'rainfall_mm': 'Annual Rainfall (mm)',
        'nitrogen_n': 'Soil Nitrogen (N)',
        'fertilizer_kg_ha': 'Fertilizer Rate (kg/ha)'
    };
    return map[name] || name;
}

// Counter animation helper
function animateCounter(elementId, targetValue, decimals = 0) {
    const el = document.getElementById(elementId);
    if (!el) return;

    let start = 0;
    const duration = 600;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const current = start + (targetValue - start) * progress;

        el.textContent = current.toLocaleString(undefined, {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        });

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// Fetch Analytics Data for Dashboard
async function fetchAnalytics() {
    try {
        const response = await fetch('/api/analytics-data');
        if (!response.ok) return;

        const data = await response.json();
        const metrics = data.metrics;

        if (metrics) {
            document.getElementById('an-total').textContent = metrics.total_records.toLocaleString();
            document.getElementById('an-ng-count').textContent = metrics.nigeria_records.toLocaleString();
            document.getElementById('an-mean-yield').textContent = `${metrics.mean_yield_kg_ha.toLocaleString()} kg/ha`;
            document.getElementById('an-rainfall').textContent = `${metrics.mean_rainfall_mm.toLocaleString()} mm`;
        }

        // Render Crop Summary Bar Chart List
        if (data.crop_summary) {
            const cropList = document.getElementById('crop-analytics-list');
            cropList.innerHTML = '';
            const maxYield = Math.max(...data.crop_summary.map(c => c.yield_kg_ha));

            data.crop_summary.forEach(c => {
                const pct = Math.round((c.yield_kg_ha / maxYield) * 100);
                const item = document.createElement('div');
                item.className = 'fi-item';
                item.innerHTML = `
                    <div class="fi-label-row">
                        <span class="fi-name">${c.crop}</span>
                        <span class="fi-pct">${c.yield_kg_ha.toLocaleString()} kg/ha</span>
                    </div>
                    <div class="fi-track">
                        <div class="fi-fill" style="width: ${pct}%;"></div>
                    </div>
                `;
                cropList.appendChild(item);
            });
        }

        // Render State Analytics Bar Chart List
        if (data.top_states) {
            const stateList = document.getElementById('state-analytics-list');
            stateList.innerHTML = '';
            const maxYield = Math.max(...data.top_states.map(s => s.yield_kg_ha));

            data.top_states.forEach(s => {
                const pct = Math.round((s.yield_kg_ha / maxYield) * 100);
                const item = document.createElement('div');
                item.className = 'fi-item';
                item.innerHTML = `
                    <div class="fi-label-row">
                        <span class="fi-name">${s.state}</span>
                        <span class="fi-pct">${s.yield_kg_ha.toLocaleString()} kg/ha</span>
                    </div>
                    <div class="fi-track">
                        <div class="fi-fill" style="width: ${pct}%;"></div>
                    </div>
                `;
                stateList.appendChild(item);
            });
        }

    } catch (err) {
        console.warn('Analytics loading error:', err);
    }
}
