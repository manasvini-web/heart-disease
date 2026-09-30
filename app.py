
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field
import joblib
import json
import time
import os
import math
from typing import Any


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "heart_disease_api_model.joblib"
)

MODEL_INFO_PATH = os.path.join(
    BASE_DIR, "models", "model_info.json"
)


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
    print("Model loaded successfully.")
except Exception as exc:
    model = None
    model_loaded = False
    print("Model loading error:", exc)


# ============================================================
# MODEL INFORMATION
# ============================================================

try:
    with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
        model_info = json.load(f)
except Exception as exc:
    model_info = {
        "model_name": "HeartDisease_4373_API_Model",
        "model_version": "1.0",
        "algorithm": "Random Forest",
        "number_of_features": 13,
        "error": str(exc),
    }


# ============================================================
# RUNTIME METRICS
# ============================================================

metrics_data = {
    "total_requests": 0,
    "successful_predictions": 0,
    "failed_requests": 0,
    "total_response_time": 0.0,
}


def clean_json_value(value: Any):
    """Convert NumPy/float values to JSON-safe Python values."""
    try:
        value = float(value)
        if not math.isfinite(value):
            return 0.0
        return value
    except (TypeError, ValueError):
        return value


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="HeartDisease_4373 API",
    description=(
        "HeartDisease_4373 Machine Learning REST API. "
        "Random Forest based heart disease prediction service."
    ),
    version="1.0",
    docs_url=None,
    redoc_url=None,
)


# ============================================================
# INPUT MODEL
# ============================================================

class HeartDiseaseInput(BaseModel):
    age: float = Field(..., ge=0, le=120, description="Age of the patient")
    sex: int = Field(..., ge=0, le=1, description="0 = Female, 1 = Male")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type")
    trestbps: float = Field(..., description="Resting blood pressure")
    chol: float = Field(..., description="Serum cholesterol")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result")
    thalch: float = Field(..., description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., description="ST depression")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST")
    ca: int = Field(..., ge=0, le=3, description="Number of major vessels")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia value")


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/", tags=["System"])
def root():
    return {
        "service": "HeartDisease_4373 API",
        "project": "HeartDisease_4373",
        "version": "1.0",
        "status": "running",
    }


@app.get("/health", tags=["System"])
def health():
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
    }


@app.get("/ready", tags=["System"])
def ready():
    return {
        "status": "ready" if model_loaded else "not_ready",
        "model_loaded": model_loaded,
    }


@app.get("/model-info", tags=["Model"])
def model_info_endpoint():
    return model_info


@app.post("/predict", tags=["Prediction"])
def predict(data: HeartDiseaseInput):
    start_time = time.time()
    metrics_data["total_requests"] += 1

    try:
        if model is None:
            raise RuntimeError(
                f"Model could not be loaded from: {MODEL_PATH}"
            )

        features = [[
            data.age,
            data.sex,
            data.cp,
            data.trestbps,
            data.chol,
            data.fbs,
            data.restecg,
            data.thalch,
            data.exang,
            data.oldpeak,
            data.slope,
            data.ca,
            data.thal,
        ]]

        prediction = model.predict(features)[0]

        response = {
            "prediction": int(prediction),
            "model_output": "Heart Disease class prediction",
            "model_version": model_info.get("model_version", "1.0"),
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            response["probabilities"] = [
                round(clean_json_value(value), 6)
                for value in probabilities
            ]

        metrics_data["successful_predictions"] += 1
        return response

    except Exception as exc:
        metrics_data["failed_requests"] += 1
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )

    finally:
        metrics_data["total_response_time"] += (
            time.time() - start_time
        )


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    total = metrics_data["total_requests"]
    average = (
        metrics_data["total_response_time"] / total
        if total > 0 else 0.0
    )

    return {
        "total_requests": total,
        "successful_predictions": metrics_data["successful_predictions"],
        "failed_requests": metrics_data["failed_requests"],
        "average_response_time_seconds": round(average, 6),
    }


# ============================================================
# SWAGGER
# ============================================================

@app.get("/swagger", include_in_schema=False)
def swagger():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="HeartDisease_4373 API - Swagger",
        swagger_js_url=(
            "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/"
            "swagger-ui-bundle.js"
        ),
        swagger_css_url=(
            "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/"
            "swagger-ui.css"
        ),
    )


# ============================================================
# TARGET DASHBOARD UI
# ============================================================

DASHBOARD_HTML = r"""
<div class="app-shell">

    <aside class="sidebar">
        <div class="side-brand">
            <div class="brand-mark">♥</div>
            <div>
                <div class="brand-title">Heart Disease</div>
                <div class="brand-subtitle">Prediction Platform</div>
            </div>
        </div>

        <div class="side-label">MAIN MENU</div>

        <button class="side-link active" data-panel="dashboard">
            <span class="side-icon">⌂</span>
            <span>Dashboard</span>
        </button>

        <button class="side-link" data-panel="patients">
            <span class="side-icon">♙</span>
            <span>Patients</span>
        </button>

        <button class="side-link" data-panel="predictions">
            <span class="side-icon">⌁</span>
            <span>Predictions</span>
        </button>

        <button class="side-link" data-panel="model">
            <span class="side-icon">◇</span>
            <span>Model</span>
        </button>

        <button class="side-link" data-panel="monitoring">
            <span class="side-icon">⌁</span>
            <span>Analytics</span>
        </button>

        <button class="side-link" data-panel="settings">
            <span class="side-icon">⚙</span>
            <span>Settings</span>
        </button>

        <div class="side-label side-label-space">API</div>

        <a class="side-link side-anchor" href="/swagger">
            <span class="side-icon">▣</span>
            <span>Swagger Docs</span>
        </a>

        <a class="side-link side-anchor" href="/openapi.json" target="_blank">
            <span class="side-icon">{ }</span>
            <span>OpenAPI JSON</span>
        </a>

        <div class="side-footer">
            <div class="admin-avatar">A</div>
            <div>
                <div class="admin-name">Admin User</div>
                <div class="admin-email">admin@heartapi.com</div>
            </div>
        </div>
    </aside>


    <main class="main-content">

        <header class="topbar">
            <div>
                <h1 id="pageTitle">Dashboard</h1>
                <p id="pageSubtitle">Manage your prediction service here</p>
            </div>

            <div class="top-actions">
                <div class="search-box">
                    <span>⌕</span>
                    <input id="globalSearch" type="text"
                           placeholder="Search..." autocomplete="off">
                </div>

                <button class="icon-button" id="refreshBtn" title="Refresh">
                    ↻
                </button>

                <button class="icon-button notification" id="notifyBtn"
                        title="Notifications">
                    ♧
                    <span class="notification-dot"></span>
                </button>

                <div class="profile-circle">A</div>
            </div>
        </header>


        <div id="toast" class="toast"></div>


        <!-- ==================================================
             DASHBOARD
             ================================================== -->

        <section id="panel-dashboard" class="content-panel active">

            <section class="hero-card">
                <div class="hero-copy">
                    <div class="hero-kicker">PRODUCTION MACHINE LEARNING API</div>
                    <h2>Welcome back, Admin</h2>
                    <p>Here's your heart disease prediction platform overview.</p>
                </div>

                <div class="hero-date">
                    <span id="currentDate">Loading date...</span>
                </div>

                <div class="hero-actions">
                    <button class="hero-btn" id="exportBtn">⇩ Export</button>
                    <button class="hero-btn" id="heroRefreshBtn">↻</button>
                </div>

                <div class="hero-metrics">
                    <div class="hero-metric">
                        <div class="metric-icon">⌁</div>
                        <div class="metric-label">Today's Predictions</div>
                        <div class="metric-value" id="heroRequests">0</div>
                        <div class="metric-note">Live API requests</div>
                    </div>

                    <div class="hero-metric">
                        <div class="metric-icon">✓</div>
                        <div class="metric-label">Successful</div>
                        <div class="metric-value" id="heroSuccess">0</div>
                        <div class="metric-note">Completed predictions</div>
                    </div>

                    <div class="hero-metric">
                        <div class="metric-icon">◉</div>
                        <div class="metric-label">Model Accuracy</div>
                        <div class="metric-value" id="heroAccuracy">—</div>
                        <div class="metric-note">Loaded model</div>
                    </div>

                    <div class="hero-metric">
                        <div class="metric-icon">⚡</div>
                        <div class="metric-label">Response Time</div>
                        <div class="metric-value" id="heroLatency">0.00s</div>
                        <div class="metric-note">Average API time</div>
                    </div>
                </div>
            </section>


            <section class="stat-grid">

                <article class="stat-card">
                    <div class="stat-top">
                        <div class="soft-icon blue">♙</div>
                        <span class="growth positive">● Online</span>
                    </div>
                    <div class="stat-value" id="totalRequestsCard">0</div>
                    <div class="stat-title">Total Requests</div>
                    <div class="stat-sub">API requests received</div>
                    <div class="progress-label">
                        <span>Successful</span><span id="successPercent">0%</span>
                    </div>
                    <div class="progress"><div id="successProgress" class="progress-fill blue-fill"></div></div>
                    <div class="progress-foot"><span>Successful predictions</span><span id="successFoot">0</span></div>
                </article>

                <article class="stat-card">
                    <div class="stat-top">
                        <div class="soft-icon green">◇</div>
                        <span class="growth positive">✓ Loaded</span>
                    </div>
                    <div class="stat-value" id="modelFeaturesCard">13</div>
                    <div class="stat-title">Model Features</div>
                    <div class="stat-sub">Input variables</div>
                    <div class="progress-label">
                        <span>Model status</span><span id="modelStatusPercent">100%</span>
                    </div>
                    <div class="progress"><div id="modelProgress" class="progress-fill green-fill"></div></div>
                    <div class="progress-foot"><span>Random Forest</span><span>Ready</span></div>
                </article>

                <article class="stat-card">
                    <div class="stat-top">
                        <div class="soft-icon orange">⌁</div>
                        <span class="growth positive">Live</span>
                    </div>
                    <div class="stat-value" id="predictionCountCard">0</div>
                    <div class="stat-title">Predictions</div>
                    <div class="stat-sub">Successful model calls</div>
                    <div class="progress-label">
                        <span>Completion</span><span id="predictionPercent">0%</span>
                    </div>
                    <div class="progress"><div id="predictionProgress" class="progress-fill orange-fill"></div></div>
                    <div class="progress-foot"><span>Failed</span><span id="failedFoot">0</span></div>
                </article>

                <article class="stat-card">
                    <div class="stat-top">
                        <div class="soft-icon purple">⚡</div>
                        <span class="growth positive">Realtime</span>
                    </div>
                    <div class="stat-value" id="latencyCard">0.00s</div>
                    <div class="stat-title">Avg. Response</div>
                    <div class="stat-sub">Prediction API latency</div>
                    <div class="progress-label">
                        <span>Target</span><span>1.00s</span>
                    </div>
                    <div class="progress"><div id="latencyProgress" class="progress-fill purple-fill"></div></div>
                    <div class="progress-foot"><span>Current</span><span id="latencyFoot">0.00s</span></div>
                </article>

            </section>


            <section class="two-column">

                <article class="panel-card analytics-card">
                    <div class="card-heading">
                        <div>
                            <h3>Prediction Analytics</h3>
                            <p>Runtime performance metrics</p>
                        </div>
                        <div class="segmented">
                            <button class="segment active" data-range="month">This Month</button>
                            <button class="segment" data-range="year">This Year</button>
                        </div>
                    </div>

                    <div class="analytics-rows">
                        <div class="analytics-row">
                            <div class="analytics-label">
                                <span>Requests</span>
                                <small id="requestsLabel">Live API traffic</small>
                            </div>
                            <div class="analytics-track">
                                <div id="requestsBar" class="analytics-fill blue-fill"></div>
                                <b id="requestsBarText">0%</b>
                            </div>
                            <div class="analytics-side" id="requestsSide">0</div>
                        </div>

                        <div class="analytics-row">
                            <div class="analytics-label">
                                <span>Successful</span>
                                <small>Completed predictions</small>
                            </div>
                            <div class="analytics-track">
                                <div id="successBar" class="analytics-fill teal-fill"></div>
                                <b id="successBarText">0%</b>
                            </div>
                            <div class="analytics-side" id="successSide">0</div>
                        </div>

                        <div class="analytics-row">
                            <div class="analytics-label">
                                <span>Failed</span>
                                <small>Requests requiring attention</small>
                            </div>
                            <div class="analytics-track">
                                <div id="failedBar" class="analytics-fill orange-fill"></div>
                                <b id="failedBarText">0%</b>
                            </div>
                            <div class="analytics-side" id="failedSide">0</div>
                        </div>
                    </div>

                    <div class="mini-stat-grid">
                        <div class="mini-stat light-blue">
                            <span>API Requests</span>
                            <strong id="miniRequests">0</strong>
                        </div>
                        <div class="mini-stat light-green">
                            <span>Success Rate</span>
                            <strong id="miniSuccessRate">0%</strong>
                        </div>
                        <div class="mini-stat light-purple">
                            <span>Avg Response</span>
                            <strong id="miniLatency">0.00s</strong>
                        </div>
                        <div class="mini-stat light-yellow">
                            <span>Model</span>
                            <strong id="miniModel">RF</strong>
                        </div>
                    </div>
                </article>


                <article class="panel-card activity-card">
                    <div class="card-heading">
                        <div>
                            <h3>Live Activity</h3>
                            <p>Real-time service updates</p>
                        </div>
                        <span class="live-badge"><i></i> Live</span>
                    </div>

                    <div id="activityList" class="activity-list"></div>

                    <button class="view-all" id="viewActivityBtn">
                        View All Activities
                    </button>
                </article>

            </section>


            <section class="two-column second-row">

                <article class="panel-card quick-card">
                    <div class="card-heading">
                        <div>
                            <h3>Quick Actions</h3>
                            <p>Frequently used API tasks</p>
                        </div>
                    </div>

                    <div class="quick-grid">
                        <button class="quick-action blue-action" data-go="predictions">
                            <span class="qa-icon">◇</span>
                            <strong>New Prediction</strong>
                            <small>Enter patient data</small>
                        </button>

                        <button class="quick-action green-action" data-go="model">
                            <span class="qa-icon">♙</span>
                            <strong>View Model</strong>
                            <small>Model information</small>
                        </button>

                        <button class="quick-action orange-action" data-go="monitoring">
                            <span class="qa-icon">▥</span>
                            <strong>View Analytics</strong>
                            <small>Runtime metrics</small>
                        </button>

                        <a class="quick-action purple-action" href="/swagger">
                            <span class="qa-icon">⚙</span>
                            <strong>Swagger</strong>
                            <small>API documentation</small>
                        </a>
                    </div>
                </article>


                <article class="panel-card recent-card">
                    <div class="card-heading">
                        <div>
                            <h3>Recent Predictions</h3>
                            <p>Latest model activity</p>
                        </div>
                        <button class="text-link" data-go="predictions">View All →</button>
                    </div>

                    <div id="recentPredictions" class="recent-list">
                        <div class="empty-state">No predictions yet.</div>
                    </div>
                </article>

            </section>


            <section class="panel-card pending-card">
                <div class="card-heading">
                    <div>
                        <h3>System Status</h3>
                        <p>Items requiring attention</p>
                    </div>
                    <span id="pendingBadge" class="pending-badge">0 Issues</span>
                </div>

                <div class="pending-grid">
                    <article class="pending-item warning">
                        <div class="pending-icon">♙</div>
                        <div>
                            <h4>Model Verification</h4>
                            <p>Check that the production model is loaded.</p>
                        </div>
                        <span id="modelIssueCount" class="pending-count">0</span>
                        <button data-go="model">Review Now →</button>
                    </article>

                    <article class="pending-item info">
                        <div class="pending-icon">◇</div>
                        <div>
                            <h4>API Readiness</h4>
                            <p>Verify service and model availability.</p>
                        </div>
                        <span id="readyIssueCount" class="pending-count">0</span>
                        <button data-go="monitoring">Check Now →</button>
                    </article>

                    <article class="pending-item danger">
                        <div class="pending-icon">⊗</div>
                        <div>
                            <h4>Failed Requests</h4>
                            <p>Prediction requests that returned errors.</p>
                        </div>
                        <span id="failedIssueCount" class="pending-count">0</span>
                        <button data-go="monitoring">View Issues →</button>
                    </article>
                </div>
            </section>

        </section>


        <!-- ==================================================
             PATIENTS
             ================================================== -->

        <section id="panel-patients" class="content-panel">
            <div class="page-card">
                <div class="large-page-icon blue">♙</div>
                <h2>Patient Data</h2>
                <p>
                    Patient information is entered through the prediction
                    form. No patient records are permanently stored by this
                    dashboard.
                </p>
                <button class="primary-btn" data-go="predictions">
                    Open Prediction Form
                </button>
            </div>
        </section>


        <!-- ==================================================
             PREDICTIONS
             ================================================== -->

        <section id="panel-predictions" class="content-panel">
            <div class="section-header">
                <div>
                    <h2>Heart Disease Prediction</h2>
                    <p>Enter the 13 model features and run a real-time prediction.</p>
                </div>
                <span class="status-chip" id="predictionApiStatus">Checking API...</span>
            </div>

            <div class="prediction-layout">
                <div class="page-card prediction-form-card">

                    <div class="form-section-title">Patient Information</div>

                    <div class="form-grid">

                        <label class="form-field">
                            <span>Age</span>
                            <input id="age" type="number" min="0" max="120" placeholder="55">
                        </label>

                        <label class="form-field">
                            <span>Sex</span>
                            <select id="sex">
                                <option value="1">Male (1)</option>
                                <option value="0">Female (0)</option>
                            </select>
                        </label>

                        <label class="form-field">
                            <span>Chest Pain Type</span>
                            <select id="cp">
                                <option value="0">Type 0</option>
                                <option value="1">Type 1</option>
                                <option value="2">Type 2</option>
                                <option value="3">Type 3</option>
                            </select>
                        </label>

                        <label class="form-field">
                            <span>Resting Blood Pressure</span>
                            <input id="trestbps" type="number" placeholder="140">
                        </label>

                        <label class="form-field">
                            <span>Cholesterol</span>
                            <input id="chol" type="number" placeholder="250">
                        </label>

                        <label class="form-field">
                            <span>Fasting Blood Sugar</span>
                            <select id="fbs">
                                <option value="0">Normal (0)</option>
                                <option value="1">High (1)</option>
                            </select>
                        </label>

                        <label class="form-field">
                            <span>Resting ECG</span>
                            <select id="restecg">
                                <option value="0">Normal (0)</option>
                                <option value="1">ST-T abnormality (1)</option>
                                <option value="2">LV hypertrophy (2)</option>
                            </select>
                        </label>

                        <label class="form-field">
                            <span>Maximum Heart Rate</span>
                            <input id="thalch" type="number" placeholder="150">
                        </label>

                        <label class="form-field">
                            <span>Exercise Induced Angina</span>
                            <select id="exang">
                                <option value="0">No (0)</option>
                                <option value="1">Yes (1)</option>
                            </select>
                        </label>

                        <label class="form-field">
                            <span>Oldpeak</span>
                            <input id="oldpeak" type="number" step="0.1" placeholder="1.0">
                        </label>

                        <label class="form-field">
                            <span>Slope</span>
                            <select id="slope">
                                <option value="0">0</option>
                                <option value="1">1</option>
                                <option value="2">2</option>
                            </select>
                        </label>

                        <label class="form-field">
                            <span>Major Vessels (CA)</span>
                            <select id="ca">
                                <option value="0">0</option>
                                <option value="1">1</option>
                                <option value="2">2</option>
                                <option value="3">3</option>
                            </select>
                        </label>

                        <label class="form-field">
                            <span>Thalassemia</span>
                            <select id="thal">
                                <option value="0">0</option>
                                <option value="1">1</option>
                                <option value="2">2</option>
                                <option value="3">3</option>
                            </select>
                        </label>

                    </div>

                    <div class="form-actions">
                        <button id="predictBtn" class="primary-btn wide">
                            ♥ Predict Heart Disease
                        </button>
                        <button id="sampleBtn" class="secondary-btn">
                            Fill Sample
                        </button>
                        <button id="clearBtn" class="secondary-btn">
                            Clear
                        </button>
                    </div>
                </div>


                <div class="page-card result-card">
                    <div class="result-heading">
                        <div>
                            <h3>Prediction Result</h3>
                            <p>Returned by POST /predict</p>
                        </div>
                        <span class="status-chip" id="resultStatus">Waiting</span>
                    </div>

                    <div id="predictionResultBox" class="prediction-result empty-result">
                        <div class="result-symbol">♥</div>
                        <h3>No prediction yet</h3>
                        <p>Submit patient information to see the model output.</p>
                    </div>

                    <div id="probabilityBox" class="probability-box"></div>
                </div>
            </div>
        </section>


        <!-- ==================================================
             MODEL
             ================================================== -->

        <section id="panel-model" class="content-panel">
            <div class="section-header">
                <div>
                    <h2>Model Information</h2>
                    <p>Details about the deployed machine learning model.</p>
                </div>
                <span class="status-chip" id="modelPageStatus">Loading...</span>
            </div>

            <div class="model-grid">
                <div class="page-card model-info-main">
                    <div class="model-badge">RF</div>
                    <h2 id="modelName">HeartDisease_4373_API_Model</h2>
                    <p id="modelAlgorithmText">Random Forest</p>

                    <div class="detail-grid">
                        <div><span>Version</span><strong id="modelVersion">—</strong></div>
                        <div><span>Features</span><strong id="modelFeatures">13</strong></div>
                        <div><span>Accuracy</span><strong id="modelAccuracy">—</strong></div>
                        <div><span>F1 Score</span><strong id="modelF1">—</strong></div>
                    </div>
                </div>

                <div class="page-card">
                    <h3>Deployment Status</h3>
                    <div class="status-list">
                        <div><span>Model file</span><strong id="deploymentModel">Checking...</strong></div>
                        <div><span>API health</span><strong id="deploymentHealth">Checking...</strong></div>
                        <div><span>Readiness</span><strong id="deploymentReady">Checking...</strong></div>
                        <div><span>Framework</span><strong>FastAPI</strong></div>
                    </div>
                </div>
            </div>
        </section>


        <!-- ==================================================
             MONITORING
             ================================================== -->

        <section id="panel-monitoring" class="content-panel">
            <div class="section-header">
                <div>
                    <h2>API Analytics</h2>
                    <p>Runtime request and prediction statistics.</p>
                </div>
                <button id="monitorRefreshBtn" class="secondary-btn">↻ Refresh</button>
            </div>

            <div class="monitor-cards">
                <div class="monitor-number">
                    <span>Total Requests</span>
                    <strong id="metricTotal">0</strong>
                </div>
                <div class="monitor-number">
                    <span>Successful Predictions</span>
                    <strong id="metricSuccess">0</strong>
                </div>
                <div class="monitor-number">
                    <span>Failed Requests</span>
                    <strong id="metricFailed">0</strong>
                </div>
                <div class="monitor-number">
                    <span>Average Response</span>
                    <strong id="metricResponse">0.00s</strong>
                </div>
            </div>

            <div class="page-card">
                <h3>Request Performance</h3>
                <div class="large-progress">
                    <div id="monitorSuccessBar" class="large-progress-fill"></div>
                </div>
                <div class="monitor-caption">
                    <span>Success rate</span>
                    <strong id="monitorSuccessRate">0%</strong>
                </div>
            </div>
        </section>


        <!-- ==================================================
             SETTINGS
             ================================================== -->

        <section id="panel-settings" class="content-panel">
            <div class="section-header">
                <div>
                    <h2>Settings</h2>
                    <p>Dashboard and API shortcuts.</p>
                </div>
            </div>

            <div class="settings-grid">
                <div class="page-card">
                    <h3>API Endpoints</h3>
                    <div class="endpoint-row"><code>GET /health</code><span>Health check</span></div>
                    <div class="endpoint-row"><code>GET /ready</code><span>Readiness check</span></div>
                    <div class="endpoint-row"><code>GET /model-info</code><span>Model metadata</span></div>
                    <div class="endpoint-row"><code>POST /predict</code><span>Prediction</span></div>
                    <div class="endpoint-row"><code>GET /metrics</code><span>Runtime metrics</span></div>
                </div>

                <div class="page-card">
                    <h3>Documentation</h3>
                    <p>Open the interactive API documentation.</p>
                    <a href="/swagger" class="primary-btn">Open Swagger</a>
                </div>
            </div>
        </section>

    </main>
</div>
"""


# ============================================================
# CSS
# ============================================================

DASHBOARD_CSS = r"""
:root {
    --blue: #1268a9;
    --blue-dark: #07548e;
    --blue-soft: #eaf4fc;
    --green: #14b887;
    --green-soft: #eafaf5;
    --orange: #ff8a00;
    --orange-soft: #fff4e5;
    --purple: #9148e8;
    --purple-soft: #f4edff;
    --red: #ef3d4f;
    --red-soft: #fff0f2;
    --yellow: #f5b942;
    --ink: #172033;
    --muted: #778197;
    --line: #e5eaf1;
    --page: #f6f8fb;
    --white: #ffffff;
    --shadow: 0 4px 18px rgba(24, 39, 75, .055);
}

* {
    box-sizing: border-box;
}

html {
    scroll-behavior: smooth;
}

body {
    margin: 0;
    min-height: 100vh;
    background: var(--page);
    color: var(--ink);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                 BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 13px;
}

button, input, select {
    font: inherit;
}

button, a {
    -webkit-tap-highlight-color: transparent;
}

button {
    cursor: pointer;
}

.app-shell {
    min-height: 100vh;
    display: flex;
}

.sidebar {
    width: 130px;
    min-height: 100vh;
    position: fixed;
    left: 0;
    top: 0;
    bottom: 0;
    z-index: 20;
    display: flex;
    flex-direction: column;
    padding: 14px 9px 10px;
    color: #fff;
    background: linear-gradient(180deg, #146eae 0%, #08629f 100%);
}

.side-brand {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 3px 5px 18px;
}

.brand-mark {
    width: 29px;
    height: 29px;
    display: grid;
    place-items: center;
    border-radius: 8px;
    background: rgba(255,255,255,.16);
    color: #fff;
    font-size: 15px;
    font-weight: 900;
}

.brand-title {
    font-size: 10px;
    font-weight: 800;
    line-height: 1.2;
}

.brand-subtitle {
    margin-top: 2px;
    font-size: 7px;
    opacity: .75;
}

.side-label {
    padding: 9px 8px 6px;
    font-size: 7px;
    letter-spacing: .9px;
    font-weight: 800;
    opacity: .52;
}

.side-label-space {
    margin-top: 8px;
}

.side-link {
    width: 100%;
    min-height: 31px;
    margin: 1px 0;
    padding: 7px 8px;
    display: flex;
    align-items: center;
    gap: 7px;
    border: 0;
    border-radius: 6px;
    background: transparent;
    color: rgba(255,255,255,.82);
    text-decoration: none;
    font-size: 8.5px;
    font-weight: 650;
    text-align: left;
}

.side-link:hover {
    background: rgba(255,255,255,.09);
    color: #fff;
}

.side-link.active {
    background: #2b8be0;
    color: #fff;
    box-shadow: 0 5px 13px rgba(0, 0, 0, .12);
}

.side-icon {
    width: 14px;
    text-align: center;
    font-size: 11px;
    flex: 0 0 14px;
}

.side-footer {
    margin-top: auto;
    padding: 9px 4px 2px;
    border-top: 1px solid rgba(255,255,255,.12);
    display: flex;
    align-items: center;
    gap: 6px;
}

.admin-avatar {
    width: 23px;
    height: 23px;
    display: grid;
    place-items: center;
    border-radius: 50%;
    background: #fff;
    color: var(--blue);
    font-size: 8px;
    font-weight: 900;
}

.admin-name {
    font-size: 7.5px;
    font-weight: 750;
}

.admin-email {
    margin-top: 1px;
    font-size: 6px;
    opacity: .62;
}

.main-content {
    width: calc(100% - 130px);
    margin-left: 130px;
    min-width: 0;
}

.topbar {
    height: 62px;
    padding: 10px 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #fff;
    border-bottom: 1px solid var(--line);
}

.topbar h1 {
    margin: 0;
    font-size: 17px;
    font-weight: 850;
    letter-spacing: -.4px;
}

.topbar p {
    margin: 3px 0 0;
    color: var(--muted);
    font-size: 8px;
}

.top-actions {
    display: flex;
    align-items: center;
    gap: 9px;
}

.search-box {
    width: 190px;
    height: 29px;
    padding: 0 9px;
    display: flex;
    align-items: center;
    gap: 6px;
    border: 1px solid var(--line);
    border-radius: 7px;
    background: #fbfcfe;
    color: #9ba4b3;
}

.search-box span {
    font-size: 14px;
}

.search-box input {
    width: 100%;
    border: 0;
    outline: 0;
    background: transparent;
    color: var(--ink);
    font-size: 8px;
}

.icon-button {
    width: 28px;
    height: 28px;
    position: relative;
    display: grid;
    place-items: center;
    border: 1px solid var(--line);
    border-radius: 7px;
    background: #fff;
    color: #6f7889;
    font-size: 12px;
}

.icon-button:hover {
    background: var(--blue-soft);
    color: var(--blue);
}

.notification-dot {
    width: 4px;
    height: 4px;
    position: absolute;
    top: 5px;
    right: 5px;
    border-radius: 50%;
    background: #ef3d4f;
}

.profile-circle {
    width: 27px;
    height: 27px;
    display: grid;
    place-items: center;
    border-radius: 50%;
    background: #216fd0;
    color: #fff;
    font-size: 8px;
    font-weight: 850;
}

.content-panel {
    display: none;
    padding: 17px 18px 28px;
    max-width: 1110px;
    margin: 0 auto;
}

.content-panel.active {
    display: block;
}

.hero-card {
    position: relative;
    min-height: 150px;
    overflow: hidden;
    padding: 18px 18px 0;
    border-radius: 13px;
    color: #fff;
    background:
        radial-gradient(circle at 90% 15%, rgba(70,166,235,.32), transparent 25%),
        linear-gradient(110deg, #116da9 0%, #0a568f 65%, #084e82 100%);
    box-shadow: 0 10px 22px rgba(15, 92, 145, .14);
}

.hero-copy {
    position: relative;
    z-index: 2;
}

.hero-kicker {
    font-size: 6.5px;
    letter-spacing: .8px;
    font-weight: 800;
    opacity: .7;
}

.hero-card h2 {
    margin: 8px 0 2px;
    font-size: 24px;
    line-height: 1.05;
    letter-spacing: -.8px;
}

.hero-card p {
    margin: 0;
    font-size: 8px;
    opacity: .78;
}

.hero-date {
    position: absolute;
    top: 18px;
    left: 315px;
    padding: 5px 8px;
    border-radius: 99px;
    background: rgba(255,255,255,.14);
    font-size: 7px;
    font-weight: 700;
}

.hero-actions {
    position: absolute;
    top: 17px;
    right: 17px;
    display: flex;
    gap: 6px;
}

.hero-btn {
    min-width: 27px;
    height: 27px;
    padding: 0 8px;
    border: 1px solid rgba(255,255,255,.13);
    border-radius: 6px;
    background: rgba(255,255,255,.1);
    color: #fff;
    font-size: 8px;
    font-weight: 700;
}

.hero-btn:hover {
    background: rgba(255,255,255,.18);
}

.hero-metrics {
    position: absolute;
    left: 18px;
    right: 18px;
    bottom: 15px;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
}

.hero-metric {
    min-height: 61px;
    padding: 8px 9px;
    border: 1px solid rgba(255,255,255,.13);
    border-radius: 9px;
    background: rgba(255,255,255,.08);
}

.metric-icon {
    width: 19px;
    height: 19px;
    display: grid;
    place-items: center;
    float: left;
    margin-right: 6px;
    border-radius: 5px;
    background: rgba(255,255,255,.12);
    font-size: 8px;
}

.metric-label {
    color: rgba(255,255,255,.68);
    font-size: 6.5px;
}

.metric-value {
    margin-top: 4px;
    font-size: 14px;
    font-weight: 850;
}

.metric-note {
    margin-top: 2px;
    color: rgba(255,255,255,.58);
    font-size: 6px;
}

.stat-grid {
    margin-top: 12px;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
}

.stat-card,
.panel-card,
.page-card {
    border: 1px solid var(--line);
    border-radius: 10px;
    background: #fff;
    box-shadow: var(--shadow);
}

.stat-card {
    min-height: 143px;
    padding: 12px;
}

.stat-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.soft-icon {
    width: 27px;
    height: 27px;
    display: grid;
    place-items: center;
    border-radius: 7px;
    font-size: 11px;
    font-weight: 800;
}

.soft-icon.blue { color: #1775b8; background: #edf6fd; }
.soft-icon.green { color: #0aa875; background: #eafaf5; }
.soft-icon.orange { color: #f08700; background: #fff5e7; }
.soft-icon.purple { color: #8d49df; background: #f4edff; }

.growth {
    padding: 4px 6px;
    border-radius: 99px;
    font-size: 6px;
    font-weight: 800;
}

.growth.positive {
    color: #13a978;
    background: #eafaf5;
}

.stat-value {
    margin-top: 9px;
    font-size: 17px;
    font-weight: 900;
    letter-spacing: -.4px;
}

.stat-title {
    margin-top: 2px;
    font-size: 8px;
    font-weight: 700;
}

.stat-sub {
    margin-top: 2px;
    color: var(--muted);
    font-size: 6.5px;
}

.progress-label,
.progress-foot {
    display: flex;
    justify-content: space-between;
    color: var(--muted);
    font-size: 5.8px;
}

.progress-label {
    margin-top: 10px;
    margin-bottom: 4px;
}

.progress-foot {
    margin-top: 4px;
}

.progress {
    height: 4px;
    overflow: hidden;
    border-radius: 99px;
    background: #edf0f4;
}

.progress-fill {
    width: 0%;
    height: 100%;
    border-radius: 99px;
    transition: width .35s ease;
}

.blue-fill { background: #1875b8; }
.green-fill { background: #10ae87; }
.orange-fill { background: #fb7b00; }
.purple-fill { background: #9349e5; }
.teal-fill { background: #08a88a; }

.two-column {
    margin-top: 12px;
    display: grid;
    grid-template-columns: 1.75fr .95fr;
    gap: 10px;
}

.second-row {
    grid-template-columns: 1.25fr 1fr;
}

.panel-card {
    padding: 12px;
}

.card-heading {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 10px;
}

.card-heading h3,
.result-heading h3,
.page-card h3 {
    margin: 0;
    font-size: 10px;
    font-weight: 850;
}

.card-heading p,
.result-heading p,
.page-card p {
    margin: 3px 0 0;
    color: var(--muted);
    font-size: 6.5px;
}

.segmented {
    display: flex;
    padding: 2px;
    border-radius: 6px;
    background: #f2f5f8;
}

.segment {
    border: 0;
    padding: 5px 8px;
    border-radius: 5px;
    background: transparent;
    color: #697487;
    font-size: 6.5px;
    font-weight: 700;
}

.segment.active {
    color: #fff;
    background: var(--blue);
}

.analytics-rows {
    margin-top: 12px;
}

.analytics-row {
    display: grid;
    grid-template-columns: 100px minmax(0, 1fr) 52px;
    align-items: center;
    gap: 9px;
    margin-bottom: 9px;
}

.analytics-label span {
    display: block;
    font-size: 7px;
    font-weight: 800;
}

.analytics-label small {
    display: block;
    margin-top: 2px;
    color: var(--muted);
    font-size: 5.8px;
}

.analytics-track {
    position: relative;
    height: 24px;
    overflow: hidden;
    border-radius: 6px;
    background: #f0f2f5;
}

.analytics-fill {
    width: 0%;
    height: 100%;
    border-radius: 6px;
    transition: width .5s ease;
}

.analytics-track b {
    position: absolute;
    right: 7px;
    top: 8px;
    color: #fff;
    font-size: 6px;
}

.analytics-side {
    text-align: right;
    font-size: 7px;
    font-weight: 800;
}

.mini-stat-grid {
    margin-top: 11px;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 7px;
}

.mini-stat {
    min-height: 44px;
    padding: 8px;
    border-radius: 7px;
    text-align: center;
}

.mini-stat span {
    display: block;
    color: var(--muted);
    font-size: 5.5px;
}

.mini-stat strong {
    display: block;
    margin-top: 4px;
    font-size: 9px;
    font-weight: 850;
}

.light-blue { background: #edf6fd; }
.light-green { background: #edf9f5; }
.light-purple { background: #f4effc; }
.light-yellow { background: #fff8e9; }

.live-badge {
    display: flex;
    align-items: center;
    gap: 4px;
    color: #11a976;
    font-size: 6px;
    font-weight: 800;
}

.live-badge i {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #11bd7f;
}

.activity-list {
    margin-top: 8px;
}

.activity-item {
    min-height: 41px;
    padding: 6px 0;
    display: grid;
    grid-template-columns: 22px 1fr;
    gap: 6px;
    border-bottom: 1px solid #f0f2f5;
}

.activity-item:last-child {
    border-bottom: 0;
}

.activity-icon {
    width: 21px;
    height: 21px;
    display: grid;
    place-items: center;
    border-radius: 6px;
    background: #edf6fd;
    color: #1a76b8;
    font-size: 8px;
}

.activity-name {
    font-size: 6.5px;
    font-weight: 800;
}

.activity-text {
    color: #687386;
    font-size: 6.3px;
}

.activity-time {
    margin-top: 2px;
    color: #a1a9b5;
    font-size: 5.5px;
}

.view-all,
.text-link {
    margin-top: 7px;
    border: 0;
    background: transparent;
    color: #1672b2;
    font-size: 6px;
    font-weight: 800;
}

.view-all {
    width: 100%;
    padding-top: 6px;
    border-top: 1px solid #f0f2f5;
}

.quick-grid {
    margin-top: 10px;
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 7px;
}

.quick-action {
    min-height: 62px;
    padding: 9px;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    justify-content: center;
    border: 0;
    border-radius: 8px;
    color: #fff;
    text-decoration: none;
    text-align: left;
}

.quick-action:hover {
    transform: translateY(-1px);
    filter: brightness(1.03);
}

.blue-action { background: #1472b2; }
.green-action { background: #0db489; }
.orange-action { background: #fa7c00; }
.purple-action { background: #8c42e3; }

.qa-icon {
    font-size: 10px;
    margin-bottom: 4px;
}

.quick-action strong {
    font-size: 7px;
}

.quick-action small {
    margin-top: 2px;
    font-size: 5.5px;
    opacity: .78;
}

.recent-list {
    margin-top: 8px;
}

.recent-item {
    min-height: 43px;
    padding: 7px 0;
    display: grid;
    grid-template-columns: 25px 1fr auto;
    align-items: center;
    gap: 7px;
    border-bottom: 1px solid #f0f2f5;
}

.recent-avatar {
    width: 25px;
    height: 25px;
    display: grid;
    place-items: center;
    border-radius: 50%;
    background: #edf6fd;
    color: #1672b2;
    font-size: 7px;
    font-weight: 900;
}

.recent-main strong {
    display: block;
    font-size: 6.5px;
}

.recent-main small {
    display: block;
    margin-top: 2px;
    color: var(--muted);
    font-size: 5.5px;
}

.recent-value {
    font-size: 7px;
    font-weight: 850;
}

.empty-state {
    padding: 23px 4px;
    text-align: center;
    color: var(--muted);
    font-size: 7px;
}

.pending-card {
    margin-top: 12px;
}

.pending-badge {
    padding: 5px 8px;
    border-radius: 99px;
    background: #fff0f2;
    color: #e8394d;
    font-size: 6px;
    font-weight: 800;
}

.pending-grid {
    margin-top: 10px;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 9px;
}

.pending-item {
    min-height: 94px;
    position: relative;
    padding: 10px 10px 9px 42px;
    border-radius: 8px;
    border: 1px solid transparent;
}

.pending-item.warning {
    background: #fff4e4;
    border-color: #ffe2bb;
}

.pending-item.info {
    background: #edf6fd;
    border-color: #d7ebfa;
}

.pending-item.danger {
    background: #fff0f2;
    border-color: #ffd7dd;
}

.pending-icon {
    width: 25px;
    height: 25px;
    position: absolute;
    left: 10px;
    top: 10px;
    display: grid;
    place-items: center;
    border-radius: 7px;
    background: rgba(255,255,255,.55);
    font-size: 10px;
}

.pending-item h4 {
    margin: 0;
    font-size: 7px;
}

.pending-item p {
    margin: 3px 0 8px;
    color: #727c8c;
    font-size: 5.5px;
    line-height: 1.45;
}

.pending-count {
    position: absolute;
    top: 9px;
    right: 9px;
    width: 20px;
    height: 20px;
    display: grid;
    place-items: center;
    border-radius: 50%;
    background: rgba(255,255,255,.75);
    font-size: 6px;
    font-weight: 900;
}

.pending-item button {
    width: 100%;
    height: 23px;
    border: 0;
    border-radius: 5px;
    color: #fff;
    font-size: 6px;
    font-weight: 800;
}

.warning button { background: #ff7900; }
.info button { background: #1472b2; }
.danger button { background: #ef3349; }

.section-header {
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.section-header h2 {
    margin: 0;
    font-size: 17px;
    font-weight: 900;
}

.section-header p {
    margin: 4px 0 0;
    color: var(--muted);
    font-size: 8px;
}

.status-chip {
    display: inline-flex;
    align-items: center;
    padding: 5px 8px;
    border-radius: 99px;
    background: var(--green-soft);
    color: #0c9d73;
    font-size: 6.5px;
    font-weight: 800;
}

.prediction-layout {
    display: grid;
    grid-template-columns: 1.25fr .75fr;
    gap: 11px;
}

.prediction-form-card,
.result-card {
    padding: 15px;
}

.form-section-title {
    margin-bottom: 11px;
    font-size: 8px;
    font-weight: 850;
    text-transform: uppercase;
    letter-spacing: .5px;
}

.form-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 9px;
}

.form-field span {
    display: block;
    margin-bottom: 4px;
    color: #556074;
    font-size: 6.5px;
    font-weight: 750;
}

.form-field input,
.form-field select {
    width: 100%;
    height: 31px;
    padding: 0 8px;
    border: 1px solid var(--line);
    border-radius: 6px;
    outline: 0;
    background: #fff;
    color: var(--ink);
    font-size: 7.5px;
}

.form-field input:focus,
.form-field select:focus {
    border-color: #1672b2;
    box-shadow: 0 0 0 2px rgba(22,114,178,.1);
}

.form-actions {
    margin-top: 13px;
    display: flex;
    gap: 6px;
}

.primary-btn,
.secondary-btn {
    min-height: 30px;
    padding: 0 12px;
    border-radius: 6px;
    font-size: 7px;
    font-weight: 800;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}

.primary-btn {
    border: 0;
    background: #1370af;
    color: #fff;
}

.primary-btn:hover {
    background: #0d5f98;
}

.secondary-btn {
    border: 1px solid var(--line);
    background: #fff;
    color: #526078;
}

.secondary-btn:hover {
    background: #f5f8fb;
}

.primary-btn.wide {
    flex: 1;
}

.result-heading {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
}

.prediction-result {
    min-height: 230px;
    margin-top: 13px;
    padding: 20px 12px;
    border: 1px dashed #dfe5ed;
    border-radius: 8px;
    text-align: center;
}

.empty-result {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.result-symbol {
    width: 43px;
    height: 43px;
    display: grid;
    place-items: center;
    margin-bottom: 9px;
    border-radius: 50%;
    background: #edf6fd;
    color: #1472b2;
    font-size: 17px;
}

.prediction-result h3 {
    margin: 0;
    font-size: 11px;
}

.prediction-result p {
    max-width: 230px;
    margin: 5px auto 0;
    color: var(--muted);
    font-size: 7px;
    line-height: 1.5;
}

.result-class {
    font-size: 32px;
    font-weight: 900;
    letter-spacing: -1px;
}

.result-meta {
    margin-top: 5px;
    color: var(--muted);
    font-size: 7px;
}

.probability-box {
    margin-top: 10px;
}

.probability-row {
    margin-top: 7px;
}

.probability-head {
    display: flex;
    justify-content: space-between;
    color: #6c7687;
    font-size: 6px;
}

.probability-track {
    height: 6px;
    margin-top: 3px;
    overflow: hidden;
    border-radius: 99px;
    background: #eef1f4;
}

.probability-fill {
    height: 100%;
    border-radius: 99px;
    background: #1472b2;
}

.model-grid {
    display: grid;
    grid-template-columns: 1.3fr .7fr;
    gap: 11px;
}

.model-info-main {
    position: relative;
    padding-left: 62px;
}

.model-badge {
    width: 39px;
    height: 39px;
    position: absolute;
    left: 15px;
    top: 15px;
    display: grid;
    place-items: center;
    border-radius: 10px;
    background: #edf6fd;
    color: #1472b2;
    font-size: 11px;
    font-weight: 900;
}

.model-info-main h2 {
    margin: 1px 0 2px;
    font-size: 13px;
}

.model-info-main > p {
    margin: 0;
    color: var(--muted);
}

.detail-grid {
    margin-top: 18px;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
}

.detail-grid div {
    padding: 10px;
    border-radius: 7px;
    background: #f7f9fb;
}

.detail-grid span {
    display: block;
    color: var(--muted);
    font-size: 5.5px;
}

.detail-grid strong {
    display: block;
    margin-top: 3px;
    font-size: 9px;
}

.status-list {
    margin-top: 9px;
}

.status-list > div,
.endpoint-row {
    padding: 8px 0;
    display: flex;
    justify-content: space-between;
    gap: 10px;
    border-bottom: 1px solid #eef1f4;
    font-size: 7px;
}

.status-list > div:last-child,
.endpoint-row:last-child {
    border-bottom: 0;
}

.status-list span {
    color: var(--muted);
}

.status-list strong {
    font-weight: 800;
}

.monitor-cards {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 11px;
}

.monitor-number {
    padding: 14px;
    border: 1px solid var(--line);
    border-radius: 9px;
    background: #fff;
    box-shadow: var(--shadow);
}

.monitor-number span {
    display: block;
    color: var(--muted);
    font-size: 7px;
}

.monitor-number strong {
    display: block;
    margin-top: 7px;
    font-size: 20px;
}

.large-progress {
    height: 12px;
    margin-top: 15px;
    overflow: hidden;
    border-radius: 99px;
    background: #eef1f4;
}

.large-progress-fill {
    width: 0%;
    height: 100%;
    background: #1472b2;
    transition: width .4s ease;
}

.monitor-caption {
    margin-top: 6px;
    display: flex;
    justify-content: space-between;
    color: var(--muted);
    font-size: 7px;
}

.settings-grid {
    display: grid;
    grid-template-columns: 1.2fr .8fr;
    gap: 11px;
}

.endpoint-row code {
    color: #1472b2;
    font-size: 7px;
    font-weight: 800;
}

.endpoint-row span {
    color: var(--muted);
}

.toast {
    position: fixed;
    right: 18px;
    top: 74px;
    z-index: 100;
    min-width: 170px;
    max-width: 280px;
    padding: 10px 12px;
    border-radius: 7px;
    background: #172033;
    color: #fff;
    box-shadow: 0 12px 30px rgba(0,0,0,.15);
    font-size: 7px;
    opacity: 0;
    transform: translateY(-8px);
    pointer-events: none;
    transition: .2s ease;
}

.toast.show {
    opacity: 1;
    transform: translateY(0);
}

.toast.error {
    background: #d92d43;
}

@media (max-width: 1050px) {
    .sidebar {
        width: 112px;
    }

    .main-content {
        width: calc(100% - 112px);
        margin-left: 112px;
    }

    .content-panel {
        padding-left: 12px;
        padding-right: 12px;
    }

    .hero-date {
        left: 275px;
    }
}

@media (max-width: 820px) {
    .sidebar {
        width: 62px;
        padding-left: 6px;
        padding-right: 6px;
    }

    .brand-title,
    .brand-subtitle,
    .side-label,
    .side-link span:not(.side-icon),
    .admin-name,
    .admin-email {
        display: none;
    }

    .side-brand {
        justify-content: center;
    }

    .side-link {
        justify-content: center;
    }

    .side-footer {
        justify-content: center;
    }

    .main-content {
        width: calc(100% - 62px);
        margin-left: 62px;
    }

    .topbar {
        padding-left: 13px;
        padding-right: 13px;
    }

    .search-box {
        width: 130px;
    }

    .hero-card {
        min-height: 245px;
    }

    .hero-date {
        left: 18px;
        top: 66px;
    }

    .hero-metrics {
        grid-template-columns: repeat(2, 1fr);
    }

    .stat-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .two-column,
    .second-row,
    .prediction-layout,
    .model-grid,
    .settings-grid {
        grid-template-columns: 1fr;
    }

    .monitor-cards {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 560px) {
    .top-actions .search-box {
        display: none;
    }

    .content-panel {
        padding: 11px 8px 20px;
    }

    .hero-card {
        min-height: 310px;
    }

    .hero-card h2 {
        font-size: 20px;
    }

    .hero-actions {
        top: 12px;
        right: 12px;
    }

    .hero-metrics {
        left: 10px;
        right: 10px;
        bottom: 10px;
    }

    .stat-grid,
    .pending-grid,
    .monitor-cards {
        grid-template-columns: 1fr;
    }

    .mini-stat-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .form-grid {
        grid-template-columns: 1fr;
    }

    .detail-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .topbar h1 {
        font-size: 14px;
    }
}
"""


# ============================================================
# JAVASCRIPT
# ============================================================

DASHBOARD_JS = r"""
const API_BASE = window.location.origin;
let lastMetrics = null;
let lastModel = null;
let activities = [];
let recentPredictions = [];


function $(id) {
    return document.getElementById(id);
}


function showToast(message, isError = false) {
    const toast = $("toast");
    toast.textContent = message;
    toast.classList.toggle("error", isError);
    toast.classList.add("show");

    clearTimeout(window.toastTimer);
    window.toastTimer = setTimeout(() => {
        toast.classList.remove("show");
    }, 2500);
}


async function fetchJSON(endpoint, options = {}) {
    const response = await fetch(
        `${API_BASE}${endpoint}`,
        {
            cache: "no-store",
            ...options
        }
    );

    const contentType = response.headers.get("content-type") || "";
    let data = null;

    if (contentType.includes("application/json")) {
        data = await response.json();
    } else {
        const text = await response.text();
        throw new Error(
            `Expected JSON from ${endpoint}; received ${response.status}: ${text.slice(0, 120)}`
        );
    }

    if (!response.ok) {
        throw new Error(
            data?.detail || data?.error || `HTTP ${response.status}`
        );
    }

    return data;
}


function goToPanel(name) {
    document.querySelectorAll(".content-panel").forEach(panel => {
        panel.classList.remove("active");
    });

    const target = $(`panel-${name}`);
    if (target) {
        target.classList.add("active");
    }

    document.querySelectorAll(".side-link[data-panel]").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.panel === name);
    });

    const titles = {
        dashboard: ["Dashboard", "Manage your prediction service here"],
        patients: ["Patients", "Enter patient information for prediction"],
        predictions: ["Predictions", "Generate heart disease predictions"],
        model: ["Model", "View deployed model information"],
        monitoring: ["Analytics", "Monitor API runtime performance"],
        settings: ["Settings", "Dashboard and API configuration"]
    };

    if (titles[name]) {
        $("pageTitle").textContent = titles[name][0];
        $("pageSubtitle").textContent = titles[name][1];
    }

    window.scrollTo({ top: 0, behavior: "smooth" });

    if (name === "model") {
        loadModel();
    }
    if (name === "monitoring") {
        loadMetrics();
    }
    if (name === "predictions") {
        loadHealth();
    }
}


function bindNavigation() {
    document.querySelectorAll(".side-link[data-panel]").forEach(btn => {
        btn.addEventListener("click", () => goToPanel(btn.dataset.panel));
    });

    document.querySelectorAll("[data-go]").forEach(btn => {
        btn.addEventListener("click", () => goToPanel(btn.dataset.go));
    });

    document.querySelectorAll(".segment").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".segment").forEach(x => x.classList.remove("active"));
            btn.classList.add("active");
            showToast(
                btn.dataset.range === "year"
                    ? "Year view selected."
                    : "Month view selected."
            );
        });
    });
}


function setDate() {
    const now = new Date();
    $("currentDate").textContent = now.toLocaleDateString(
        undefined,
        {
            weekday: "long",
            month: "long",
            day: "numeric",
            year: "numeric"
        }
    );
}


function setText(id, value) {
    const el = $(id);
    if (el) el.textContent = value;
}


function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}


function renderActivities() {
    const box = $("activityList");

    if (!activities.length) {
        box.innerHTML = `
            <div class="empty-state">
                No live activity yet. Make a prediction to create an activity.
            </div>
        `;
        return;
    }

    box.innerHTML = activities.slice(0, 5).map(item => `
        <div class="activity-item">
            <div class="activity-icon">${item.icon}</div>
            <div>
                <div class="activity-name">${item.title}</div>
                <div class="activity-text">${item.text}</div>
                <div class="activity-time">${item.time}</div>
            </div>
        </div>
    `).join("");
}


function addActivity(title, text, icon = "✓") {
    activities.unshift({
        title,
        text,
        icon,
        time: "just now"
    });

    activities = activities.slice(0, 8);
    renderActivities();
}


function renderRecentPredictions() {
    const box = $("recentPredictions");

    if (!recentPredictions.length) {
        box.innerHTML = `<div class="empty-state">No predictions yet.</div>`;
        return;
    }

    box.innerHTML = recentPredictions.slice(0, 4).map((item, index) => `
        <div class="recent-item">
            <div class="recent-avatar">${item.prediction}</div>
            <div class="recent-main">
                <strong>Prediction #${recentPredictions.length - index}</strong>
                <small>Class ${item.prediction} · Version ${item.version}</small>
            </div>
            <div class="recent-value">${item.time}</div>
        </div>
    `).join("");
}


function renderMetrics(data) {
    lastMetrics = data;

    const total = Number(data.total_requests || 0);
    const success = Number(data.successful_predictions || 0);
    const failed = Number(data.failed_requests || 0);
    const response = Number(data.average_response_time_seconds || 0);

    const successRate = total > 0 ? (success / total) * 100 : 0;
    const failedRate = total > 0 ? (failed / total) * 100 : 0;
    const latencyScore = response > 0
        ? clamp((response / 1) * 100, 0, 100)
        : 0;

    setText("heroRequests", total);
    setText("heroSuccess", success);
    setText("heroLatency", response.toFixed(2) + "s");

    setText("totalRequestsCard", total);
    setText("successPercent", Math.round(successRate) + "%");
    setText("successFoot", success);
    $("successProgress").style.width = successRate + "%";

    setText("predictionCountCard", success);
    setText("predictionPercent", Math.round(successRate) + "%");
    setText("failedFoot", failed);
    $("predictionProgress").style.width = successRate + "%";

    setText("latencyCard", response.toFixed(2) + "s");
    setText("latencyFoot", response.toFixed(2) + "s");
    $("latencyProgress").style.width = latencyScore + "%";

    setText("requestsBarText", Math.min(total, 100) + "%");
    $("requestsBar").style.width = Math.min(total, 100) + "%";
    setText("requestsSide", total);

    setText("successBarText", Math.round(successRate) + "%");
    $("successBar").style.width = successRate + "%";
    setText("successSide", success);

    setText("failedBarText", Math.round(failedRate) + "%");
    $("failedBar").style.width = failedRate + "%";
    setText("failedSide", failed);

    setText("miniRequests", total);
    setText("miniSuccessRate", Math.round(successRate) + "%");
    setText("miniLatency", response.toFixed(2) + "s");

    setText("metricTotal", total);
    setText("metricSuccess", success);
    setText("metricFailed", failed);
    setText("metricResponse", response.toFixed(2) + "s");

    setText("monitorSuccessRate", Math.round(successRate) + "%");
    $("monitorSuccessBar").style.width = successRate + "%";

    setText("failedIssueCount", failed);
    setText("pendingBadge", failed + " Issues");

    setText("failedIssueCount", failed);
}


async function loadMetrics() {
    try {
        const data = await fetchJSON("/metrics");
        renderMetrics(data);
    } catch (error) {
        showToast("Unable to load API metrics: " + error.message, true);
    }
}


function renderModel(data) {
    lastModel = data;

    const algorithm = data.algorithm || "Random Forest";
    const version = data.model_version || "1.0";
    const features = data.number_of_features || 13;
    const accuracy = Number(data.accuracy);

    setText("modelName", data.model_name || "HeartDisease_4373_API_Model");
    setText("modelAlgorithmText", algorithm);
    setText("modelVersion", version);
    setText("modelFeatures", features);
    setText(
        "modelAccuracy",
        Number.isFinite(accuracy) ? (accuracy * 100).toFixed(2) + "%" : "—"
    );
    setText(
        "modelF1",
        Number.isFinite(Number(data.f1_score))
            ? Number(data.f1_score).toFixed(4)
            : "—"
    );

    setText("modelFeaturesCard", features);
    setText(
        "heroAccuracy",
        Number.isFinite(accuracy) ? (accuracy * 100).toFixed(2) + "%" : "—"
    );

    setText(
        "miniModel",
        algorithm.toLowerCase().includes("forest") ? "RF" : algorithm.slice(0, 2).toUpperCase()
    );

    setText("deploymentModel", "Loaded");
    setText("modelPageStatus", "Loaded");
}


async function loadModel() {
    try {
        const data = await fetchJSON("/model-info");
        renderModel(data);
    } catch (error) {
        setText("deploymentModel", "Unavailable");
        setText("modelPageStatus", "Unavailable");
        showToast("Unable to load model information.", true);
    }
}


async function loadHealth() {
    try {
        const health = await fetchJSON("/health");
        const ready = await fetchJSON("/ready");

        const healthy = health.status === "healthy" && health.model_loaded;
        const isReady = ready.status === "ready" && ready.model_loaded;

        setText("predictionApiStatus", healthy ? "API Online" : "API Offline");
        setText("deploymentHealth", healthy ? "Healthy" : "Unavailable");
        setText("deploymentReady", isReady ? "Ready" : "Not Ready");

        setText("modelIssueCount", healthy ? 0 : 1);
        setText("readyIssueCount", isReady ? 0 : 1);

        $("modelProgress").style.width = healthy ? "100%" : "35%";
        setText("modelStatusPercent", healthy ? "100%" : "35%");
    } catch (error) {
        setText("predictionApiStatus", "API Offline");
        setText("deploymentHealth", "Unavailable");
        setText("deploymentReady", "Unavailable");
        setText("modelIssueCount", 1);
        setText("readyIssueCount", 1);
        $("modelProgress").style.width = "0%";
        setText("modelStatusPercent", "0%");
    }
}


async function predictHeartDisease() {
    const fields = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalch", "exang", "oldpeak", "slope",
        "ca", "thal"
    ];

    const payload = {};

    for (const id of fields) {
        const value = $(id).value;

        if (value === "") {
            showToast("Please fill all required patient fields.", true);
            $(id).focus();
            return;
        }

        payload[id] = Number(value);

        if (!Number.isFinite(payload[id])) {
            showToast(`Invalid value for ${id}.`, true);
            $(id).focus();
            return;
        }
    }

    const button = $("predictBtn");
    const oldText = button.textContent;

    button.disabled = true;
    button.textContent = "Predicting...";

    try {
        const data = await fetchJSON("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify(payload)
        });

        renderPrediction(data);

        recentPredictions.unshift({
            prediction: data.prediction,
            version: data.model_version,
            time: "just now"
        });

        recentPredictions = recentPredictions.slice(0, 6);
        renderRecentPredictions();

        addActivity(
            "Prediction completed",
            `Model returned class ${data.prediction}`,
            "✓"
        );

        await loadMetrics();

        showToast("Prediction completed successfully.");
    } catch (error) {
        $("predictionResultBox").className = "prediction-result";
        $("predictionResultBox").innerHTML = `
            <div class="result-symbol" style="background:#fff0f2;color:#df3046">!</div>
            <h3>Prediction Failed</h3>
            <p>${escapeHTML(error.message)}</p>
        `;
        $("probabilityBox").innerHTML = "";
        setText("resultStatus", "Error");
        showToast(error.message, true);
        await loadMetrics();
    } finally {
        button.disabled = false;
        button.textContent = oldText;
    }
}


function renderPrediction(data) {
    $("predictionResultBox").className = "prediction-result";
    $("predictionResultBox").innerHTML = `
        <div class="result-symbol">♥</div>
        <div class="result-class">Class ${escapeHTML(data.prediction)}</div>
        <div class="result-meta">
            Model Version ${escapeHTML(data.model_version || "1.0")}
        </div>
    `;

    setText("resultStatus", "Completed");

    const probabilityBox = $("probabilityBox");
    probabilityBox.innerHTML = "";

    if (Array.isArray(data.probabilities)) {
        data.probabilities.forEach((value, index) => {
            const numeric = Number(value);
            if (!Number.isFinite(numeric)) return;

            const percentage = Math.max(0, Math.min(100, numeric * 100));

            probabilityBox.insertAdjacentHTML(
                "beforeend",
                `
                <div class="probability-row">
                    <div class="probability-head">
                        <span>Class ${index}</span>
                        <span>${percentage.toFixed(1)}%</span>
                    </div>
                    <div class="probability-track">
                        <div class="probability-fill"
                             style="width:${percentage}%"></div>
                    </div>
                </div>
                `
            );
        });
    }
}


function escapeHTML(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


function fillSample() {
    const sample = {
        age: 55,
        sex: 1,
        cp: 2,
        trestbps: 140,
        chol: 250,
        fbs: 0,
        restecg: 1,
        thalch: 150,
        exang: 0,
        oldpeak: 1.0,
        slope: 1,
        ca: 0,
        thal: 2
    };

    Object.entries(sample).forEach(([key, value]) => {
        $(key).value = value;
    });

    showToast("Sample patient values filled.");
}


function clearForm() {
    [
        "age", "trestbps", "chol", "thalch", "oldpeak"
    ].forEach(id => $(id).value = "");

    [
        "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
    ].forEach(id => $(id).selectedIndex = 0);

    $("predictionResultBox").className = "prediction-result empty-result";
    $("predictionResultBox").innerHTML = `
        <div class="result-symbol">♥</div>
        <h3>No prediction yet</h3>
        <p>Submit patient information to see the model output.</p>
    `;
    $("probabilityBox").innerHTML = "";
    setText("resultStatus", "Waiting");

    showToast("Prediction form cleared.");
}


function exportDashboard() {
    const exportData = {
        project: "HeartDisease_4373",
        exported_at: new Date().toISOString(),
        model: lastModel,
        metrics: lastMetrics,
        recent_predictions: recentPredictions
    };

    const blob = new Blob(
        [JSON.stringify(exportData, null, 2)],
        { type: "application/json" }
    );

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "heart_disease_api_dashboard.json";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);

    showToast("Dashboard data exported.");
}


function refreshDashboard() {
    Promise.all([
        loadHealth(),
        loadModel(),
        loadMetrics()
    ]).then(() => {
        showToast("Dashboard refreshed.");
    });
}


function initialize() {
    setDate();
    bindNavigation();
    renderActivities();
    renderRecentPredictions();

    $("predictBtn").addEventListener("click", predictHeartDisease);
    $("sampleBtn").addEventListener("click", fillSample);
    $("clearBtn").addEventListener("click", clearForm);

    $("refreshBtn").addEventListener("click", refreshDashboard);
    $("heroRefreshBtn").addEventListener("click", refreshDashboard);
    $("monitorRefreshBtn").addEventListener("click", loadMetrics);
    $("exportBtn").addEventListener("click", exportDashboard);

    $("notifyBtn").addEventListener("click", () => {
        showToast("No new system notifications.");
    });

    $("viewActivityBtn").addEventListener("click", () => {
        goToPanel("monitoring");
    });

    $("globalSearch").addEventListener("keydown", event => {
        if (event.key !== "Enter") return;

        const query = event.target.value.trim().toLowerCase();

        if (!query) {
            goToPanel("dashboard");
            return;
        }

        if (query.includes("predict") || query.includes("patient")) {
            goToPanel("predictions");
        } else if (query.includes("model")) {
            goToPanel("model");
        } else if (
            query.includes("metric") ||
            query.includes("analytic") ||
            query.includes("request")
        ) {
            goToPanel("monitoring");
        } else if (query.includes("setting")) {
            goToPanel("settings");
        } else {
            showToast("No matching dashboard section.");
        }
    });

    loadHealth();
    loadModel();
    loadMetrics();

    setInterval(loadMetrics, 5000);
    setInterval(loadHealth, 10000);
}

document.addEventListener("DOMContentLoaded", initialize);
"""


# ============================================================
# CUSTOM DOCS / DASHBOARD
# ============================================================

@app.get("/docs", include_in_schema=False)
def custom_docs():
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">
    <title>HeartDisease_4373 Dashboard</title>
    <style>{DASHBOARD_CSS}</style>
</head>
<body>
    {DASHBOARD_HTML}
    <script>{DASHBOARD_JS}</script>
</body>
</html>
"""
    return HTMLResponse(content=html)


# ============================================================
# OPTIONAL: serve the dashboard from /dashboard as well
# ============================================================

@app.get("/dashboard", include_in_schema=False)
def dashboard():
    return custom_docs()
