// GET HTML ELEMENTS
const homePage = document.getElementById("homePage");
const dashboardPage = document.getElementById("dashboardPage");

// Logistic-regression model trained from cardio_train.csv.
// Feature order: age_years, gender, height, weight, systolic BP,
// diastolic BP, cholesterol, glucose, smoke, alcohol, active.
const MODEL_MEAN = [53.2759374, 1.34909270, 164.4154, 74.1383, 126.6374, 81.2763, 1.3658541, 1.2278458, 0.08837997, 0.05383690, 0.80299874];
const MODEL_SCALE = [6.77469811, 0.47668332, 7.89914859, 14.33945948, 16.7001274, 9.43865344, 0.67949172, 0.57403621, 0.28384671, 0.22569557, 0.39773321];
const MODEL_COEF = [0.34813541, -0.00925664, -0.02491153, 0.15271779, 0.9298108, 0.11197161, 0.34122681, -0.06626927, -0.04269353, -0.05249961, -0.09039738];
const MODEL_INTERCEPT = 0.02771095;

let currentPatient = null;

// CAMERA / FOLDER OCR SCANNER
const cameraInput = document.getElementById("cameraInput");
const folderInput = document.getElementById("folderInput");
const mediaAccessStatus = document.getElementById("mediaAccessStatus");
let tesseractLoadPromise = null;

function loadTesseract() {
    if (window.Tesseract) return Promise.resolve(window.Tesseract);
    if (tesseractLoadPromise) return tesseractLoadPromise;

    tesseractLoadPromise = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = "https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js";
        script.onload = () => window.Tesseract ? resolve(window.Tesseract) : reject(new Error("OCR library did not load."));
        script.onerror = () => reject(new Error("Could not load the OCR scanner. Check your internet connection and try again."));
        document.head.appendChild(script);
    });
    return tesseractLoadPromise;
}

function setScannedField(id, value) {
    if (value === undefined || value === null || value === "") return false;
    const field = document.getElementById(id);
    if (!field) return false;
    const option = field.tagName === "SELECT"
        ? Array.from(field.options).find(item => item.value === String(value))
        : null;
    if (field.tagName === "SELECT") {
        if (!option) return false;
        field.value = String(value);
    } else {
        field.value = String(value);
    }
    field.dispatchEvent(new Event("input", { bubbles: true }));
    field.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
}

function extractPatientDetails(ocrText) {
    const text = ocrText.replace(/\r/g, "\n");
    const lines = text.split(/\n+/).map(line => line.trim()).filter(Boolean);
    const result = {};

    function matchValue(pattern) {
        for (const line of lines) {
            const found = line.match(pattern);
            if (found && found[1]) return found[1].trim().replace(/[|]+/g, "").trim();
        }
        return null;
    }

    const name = matchValue(/(?:patient\s*name|full\s*name|name)\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{1,60})/i);
    if (name) result.name = name;

    const age = matchValue(/(?:age)\s*[:\-]?\s*(\d{1,3})\s*(?:years?|yrs?)?/i);
    if (age && Number(age) >= 30 && Number(age) <= 65) result.age = age;

    const gender = matchValue(/(?:gender|sex)\s*[:\-]?\s*(male|female|other)/i);
    if (gender) result.gender = /^male$/i.test(gender) ? "2" : /^female$/i.test(gender) ? "1" : "1.5";

    const height = matchValue(/height\s*[:\-]?\s*(\d{2,3}(?:\.\d+)?)\s*(cm|m)?/i);
    if (height) {
        const line = lines.find(item => /height/i.test(item));
        const unit = line && line.match(/(\d{2,3}(?:\.\d+)?)\s*(cm|m)\b/i);
        let value = Number(height);
        if (unit && unit[2].toLowerCase() === "m") value *= 100;
        if (value >= 120 && value <= 220) result.height = Math.round(value);
    }

    const weight = matchValue(/weight\s*[:\-]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:kg|kgs)?/i);
    if (weight && Number(weight) >= 30 && Number(weight) <= 200) result.weight = weight;

    let bp = matchValue(/(?:blood\s*pressure|bp)\s*[:\-]?\s*(\d{2,3})\s*[\/\-]\s*(\d{2,3})/i);
    if (bp) {
        const parts = bp.match(/(\d{2,3})\s*[\/\-]\s*(\d{2,3})/);
        if (parts) {
            if (Number(parts[1]) >= 60 && Number(parts[1]) <= 250) result.ap_hi = parts[1];
            if (Number(parts[2]) >= 40 && Number(parts[2]) <= 150) result.ap_lo = parts[2];
        }
    } else {
        const systolic = matchValue(/systolic(?:\s*(?:bp|blood pressure))?\s*[:\-]?\s*(\d{2,3})/i);
        const diastolic = matchValue(/diastolic(?:\s*(?:bp|blood pressure))?\s*[:\-]?\s*(\d{2,3})/i);
        if (systolic && Number(systolic) >= 60 && Number(systolic) <= 250) result.ap_hi = systolic;
        if (diastolic && Number(diastolic) >= 40 && Number(diastolic) <= 150) result.ap_lo = diastolic;
    }

    const cholesterol = matchValue(/cholesterol\s*[:\-]?\s*(normal|above\s*normal|well\s*above\s*normal|high|elevated)/i);
    if (cholesterol) {
        const v = cholesterol.toLowerCase().replace(/\s+/g, " ");
        result.cholesterol = v === "normal" ? "1" : v === "above normal" ? "2" : "3";
    }
    const glucose = matchValue(/(?:glucose|blood\s*sugar)\s*[:\-]?\s*(normal|above\s*normal|well\s*above\s*normal|high|elevated)/i);
    if (glucose) {
        const v = glucose.toLowerCase().replace(/\s+/g, " ");
        result.gluc = v === "normal" ? "1" : v === "above normal" ? "2" : "3";
    }

    const smoke = matchValue(/(?:smoking|smoker|smoke)\s*[:\-]?\s*(yes|no|positive|negative)/i);
    if (smoke) result.smoke = /^(yes|positive)$/i.test(smoke) ? "1" : "0";
    const alcohol = matchValue(/(?:alcohol(?:\s*consumption)?|drinks\s*alcohol)\s*[:\-]?\s*(yes|no|positive|negative)/i);
    if (alcohol) result.alco = /^(yes|positive)$/i.test(alcohol) ? "1" : "0";
    const active = matchValue(/(?:physically\s*active|physical\s*activity|activity\s*level)\s*[:\-]?\s*(yes|no|active|inactive)/i);
    if (active) result.active = /^(yes|active)$/i.test(active) ? "1" : "0";

    return result;
}

// LIVE CAMERA STREAM CONTROLLER
let liveCameraStream = null;
let currentCameraFacing = "environment";

async function openLiveCamera() {
    const modal = document.getElementById("liveCameraModal");
    const video = document.getElementById("webcamVideo");
    const flipBtn = document.getElementById("btnFlipCamera");

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        mediaAccessStatus.textContent = "Live webcam stream is not supported in this browser environment. Opening file selector instead.";
        document.getElementById("cameraInput").click();
        return;
    }

    if (modal) modal.style.display = "flex";
    mediaAccessStatus.textContent = "Starting camera...";

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { ideal: currentCameraFacing },
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });

        liveCameraStream = stream;
        if (video) {
            video.srcObject = stream;
            await video.play().catch(() => {});
        }

        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(d => d.kind === "videoinput");
            if (flipBtn && videoDevices.length > 1) {
                flipBtn.style.display = "inline-block";
            }
        } catch (_) {}

        mediaAccessStatus.textContent = "Camera active. Hold your patient record in front of the lens and click 'Capture & Scan'.";
    } catch (cameraErr) {
        console.warn("Camera access error:", cameraErr);
        closeLiveCamera();
        mediaAccessStatus.textContent = "Camera access denied or unavailable. Opening file selector to choose an image of your record.";
        document.getElementById("cameraInput").click();
    }
}

function closeLiveCamera() {
    const modal = document.getElementById("liveCameraModal");
    const video = document.getElementById("webcamVideo");
    if (liveCameraStream) {
        liveCameraStream.getTracks().forEach(track => track.stop());
        liveCameraStream = null;
    }
    if (video) {
        video.srcObject = null;
    }
    if (modal) modal.style.display = "none";
}

async function flipLiveCamera() {
    currentCameraFacing = (currentCameraFacing === "environment") ? "user" : "environment";
    closeLiveCamera();
    openLiveCamera();
}

async function captureLiveSnapshot() {
    const video = document.getElementById("webcamVideo");
    const canvas = document.getElementById("cameraCanvas");
    if (!video || !canvas || !liveCameraStream) return;

    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Stop camera stream & close viewfinder modal
    closeLiveCamera();

    mediaAccessStatus.textContent = "Photo captured! Reading patient record with OCR...";

    canvas.toBlob(async (blob) => {
        if (!blob) {
            mediaAccessStatus.textContent = "Capture failed. Please try again or select an image file.";
            return;
        }
        const photoFile = new File([blob], "patient_record_capture.jpg", { type: "image/jpeg" });
        await scanPatientFiles([photoFile]);
    }, "image/jpeg", 0.95);
}

async function scanPatientFiles(fileList) {
    const files = Array.from(fileList || []);
    const imageFiles = files.filter(file => file.type.startsWith("image/"));
    if (!imageFiles.length) {
        mediaAccessStatus.textContent = "No image found. Please choose a clear JPG, PNG, or other image file of the patient record.";
        return;
    }

    mediaAccessStatus.textContent = "Loading scanner…";
    try {
        const Tesseract = await loadTesseract();
        const combinedText = [];
        for (let i = 0; i < imageFiles.length; i++) {
            const file = imageFiles[i];
            mediaAccessStatus.textContent = `Scanning ${i + 1} of ${imageFiles.length}: ${file.name}`;
            const result = await Tesseract.recognize(file, "eng", {
                logger: message => {
                    if (message.status === "recognizing text") {
                        mediaAccessStatus.textContent = `Reading ${file.name}: ${Math.round((message.progress || 0) * 100)}%`;
                    }
                }
            });
            combinedText.push(result.data.text || "");
        }

        const extracted = extractPatientDetails(combinedText.join("\n"));
        const filled = [];
        Object.entries(extracted).forEach(([id, value]) => {
            if (setScannedField(id, value)) filled.push(id);
        });

        if (filled.length) {
            mediaAccessStatus.textContent =
                `Scan complete. Filled ${filled.length} field(s): ${filled.join(", ")}. Please check every value and complete any missing fields before clicking Get Started.`;
        } else {
            mediaAccessStatus.textContent =
                "The scanner could not confidently match patient details. Try a clearer, well-lit image with labels such as Name, Age, Height, Weight and BP.";
        }
    } catch (error) {
        console.error("Patient record scan failed:", error);
        mediaAccessStatus.textContent = error.message || "Scanning failed. Please try another image.";
    }
}

cameraInput.addEventListener("change", function () {
    if (this.files && this.files.length) scanPatientFiles(this.files);
});

folderInput.addEventListener("change", function () {
    if (this.files && this.files.length) scanPatientFiles(this.files);
});


function calculateModelProbability(patient) {
    const values = [
        Number(patient.age), Number(patient.gender), Number(patient.height),
        Number(patient.weight), Number(patient.ap_hi), Number(patient.ap_lo),
        Number(patient.cholesterol), Number(patient.gluc), Number(patient.smoke),
        Number(patient.alco), Number(patient.active)
    ];

    let z = MODEL_INTERCEPT;
    for (let i = 0; i < values.length; i++) {
        z += ((values[i] - MODEL_MEAN[i]) / MODEL_SCALE[i]) * MODEL_COEF[i];
    }
    return 1 / (1 + Math.exp(-z));
}

function getPatientFromForm() {
    const ids = ["name", "age", "gender", "height", "weight", "ap_hi",
                 "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"];
    const patient = {};
    ids.forEach(id => {
        const element = document.getElementById(id);
        patient[id] = element.value.trim();
    });
    return patient;
}

function renderPatient(patient) {
    document.getElementById("userDisplay").textContent = patient.name;
    updateBiomarkers(patient);
    const details = document.getElementById("patientDetails");
    details.innerHTML = "";

    const rows = [
        ["Name", patient.name],
        ["Age", patient.age + " years"],
        ["Gender", patient.gender === "2" ? "Male" : patient.gender === "1" ? "Female" : "Other / not specified"],
        ["Height", patient.height + " cm"],
        ["Weight", patient.weight + " kg"],
        ["Blood pressure", patient.ap_hi + "/" + patient.ap_lo + " mmHg"],
        ["Cholesterol", ["", "Normal", "Above normal", "Well above normal"][Number(patient.cholesterol)]],
        ["Glucose", ["", "Normal", "Above normal", "Well above normal"][Number(patient.gluc)]],
        ["Smoking", patient.smoke === "1" ? "Yes" : "No"],
        ["Alcohol", patient.alco === "1" ? "Yes" : "No"],
        ["Physically active", patient.active === "1" ? "Yes" : "No"]
    ];

    rows.forEach(item => {
        const paragraph = document.createElement("p");
        const label = document.createElement("strong");
        label.textContent = item[0] + ": ";
        paragraph.appendChild(label);
        paragraph.appendChild(document.createTextNode(item[1]));
        details.appendChild(paragraph);
    });
}

let currentModelChoice = "primary";

// BACKEND API CONNECTIVITY & RUNTIME SETTINGS
function updateApiStatusIndicators(connected, label) {
    const dots = document.querySelectorAll(".api-dot");
    const texts = document.querySelectorAll("#apiStatusText, .apiStatusTextMirror");
    dots.forEach(dot => {
        dot.className = "api-dot " + (connected ? "connected" : "offline");
    });
    texts.forEach(txt => {
        txt.textContent = label || (connected ? "API: Online" : "API: Local/Offline");
    });
}

async function checkApiConnection() {
    const healthUrl = window.CARDIOGUARD_CONFIG ? window.CARDIOGUARD_CONFIG.getApiUrl("/health") : "/health";
    try {
        const res = await fetch(healthUrl, { method: "GET", cache: "no-store" });
        if (res.ok) {
            const data = await res.json();
            updateApiStatusIndicators(true, "API: Online");
            return true;
        } else {
            updateApiStatusIndicators(false, "API: Standalone");
            return false;
        }
    } catch (e) {
        updateApiStatusIndicators(false, "API: Standalone");
        return false;
    }
}

function openApiSettings() {
    const modal = document.getElementById("apiSettingsModal");
    const input = document.getElementById("apiUrlInput");
    const status = document.getElementById("apiTestStatus");
    if (modal) modal.style.display = "flex";
    if (input) {
        input.value = localStorage.getItem("cardioguard_api_url") || (window.CARDIOGUARD_CONFIG ? window.CARDIOGUARD_CONFIG.API_BASE_URL : "");
    }
    if (status) status.style.display = "none";
}

function closeApiSettings() {
    const modal = document.getElementById("apiSettingsModal");
    if (modal) modal.style.display = "none";
}

async function testApiConnectionUI() {
    const input = document.getElementById("apiUrlInput");
    const status = document.getElementById("apiTestStatus");
    if (!input || !status) return;

    let testBase = input.value.trim().replace(/\/+$/, "");
    let testUrl = testBase ? (testBase + "/health") : (window.CARDIOGUARD_CONFIG ? window.CARDIOGUARD_CONFIG.getApiUrl("/health") : "/health");

    status.style.display = "block";
    status.style.background = "#eff6ff";
    status.style.color = "#1e40af";
    status.textContent = "Connecting to " + testUrl + "...";

    try {
        const res = await fetch(testUrl, { method: "GET", cache: "no-store" });
        if (res.ok) {
            const data = await res.json();
            status.style.background = "#ecfdf5";
            status.style.color = "#065f46";
            status.innerHTML = `✓ <strong>Connected successfully!</strong> (${data.service || "CardioGuard API"})`;
        } else {
            status.style.background = "#fef2f2";
            status.style.color = "#991b1b";
            status.innerHTML = `✗ Server responded with HTTP ${res.status}. Check endpoint URL.`;
        }
    } catch (err) {
        status.style.background = "#fffbeb";
        status.style.color = "#92400e";
        status.innerHTML = `✗ Could not connect. Ensure backend is running and CORS is allowed.`;
    }
}

function saveApiUrlFromUI() {
    const input = document.getElementById("apiUrlInput");
    if (!input) return;
    const url = input.value.trim().replace(/\/+$/, "");
    if (url) {
        localStorage.setItem("cardioguard_api_url", url);
    } else {
        localStorage.removeItem("cardioguard_api_url");
    }
    closeApiSettings();
    checkApiConnection();
    if (currentPatient) {
        renderRisk(currentPatient, currentModelChoice);
    }
}

// Check API connection on startup
window.addEventListener("DOMContentLoaded", () => {
    checkApiConnection();
});

function switchModel(modelKey) {
    currentModelChoice = modelKey;
    const btnPrimary = document.getElementById("btnModelPrimary");
    const btnSVC = document.getElementById("btnModelSVC");
    const btnKNN = document.getElementById("btnModelKNN");
    const activeTag = document.getElementById("activeModelTag");

    if (btnPrimary) btnPrimary.classList.toggle("active", modelKey === "primary");
    if (btnSVC) btnSVC.classList.toggle("active", modelKey === "svc");
    if (btnKNN) btnKNN.classList.toggle("active", modelKey === "knn");

    if (activeTag) {
        if (modelKey === "primary") activeTag.textContent = "Active: Primary Ensemble (80.1% AUC)";
        else if (modelKey === "svc") activeTag.textContent = "Active: Support Vector Classifier (SVC)";
        else if (modelKey === "knn") activeTag.textContent = "Active: K-Nearest Neighbors (KNN)";
    }

    if (currentPatient) {
        renderRisk(currentPatient, modelKey);
    }
}

async function renderRisk(patient, modelChoice = currentModelChoice) {
    if (!patient) return;
    document.getElementById("riskPatientName").textContent = "Computing model estimate for " + patient.name + "...";

    const payload = {
        patient: {
            name: String(patient.name),
            age: Number(patient.age),
            gender: Number(patient.gender),
            height_cm: Number(patient.height),
            weight_kg: Number(patient.weight)
        },
        health_parameters: {
            systolic_bp_mmhg: Number(patient.ap_hi),
            diastolic_bp_mmhg: Number(patient.ap_lo),
            cholesterol_level: Number(patient.cholesterol),
            glucose_level: Number(patient.gluc),
            smoking: Number(patient.smoke),
            alcohol: Number(patient.alco),
            physically_active: Number(patient.active)
        },
        model_choice: modelChoice
    };

    let percent = 0;
    let label = "";
    let explanation = "";
    let modelName = "";
    let recommendations = [];

    try {
        const targetUrl = window.CARDIOGUARD_CONFIG ? window.CARDIOGUARD_CONFIG.getApiUrl("/predict") : "/predict";
        const response = await fetch(targetUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error("HTTP " + response.status);
        }

        const result = await response.json();
        percent = Math.round(result.risk_percentage);
        label = result.risk_level + " Risk (" + percent + "%)";
        explanation = result.message;
        modelName = result.model_name;
        recommendations = result.recommendations || [];

    } catch (apiError) {
        console.warn("Backend prediction unavailable, using offline mathematical baseline:", apiError);
        const probability = calculateModelProbability(patient);
        percent = Math.round(probability * 100);
        if (probability < 0.5) {
            label = "Lower model score (Local mode)";
            explanation = "The baseline model estimates lower probability of cardiovascular indicators for the values entered.";
        } else {
            label = "Higher model score (Local mode)";
            explanation = "The baseline model estimates an elevated probability of cardiovascular indicators for the values entered.";
        }
        modelName = "Logistic Regression Baseline (Offline)";
    }

    document.getElementById("riskPatientName").textContent = "Model estimate for " + patient.name;
    document.getElementById("riskScore").textContent = percent + "%";

    const meter = document.getElementById("riskMeter");
    meter.style.width = Math.min(100, Math.max(0, percent)) + "%";
    if (percent >= 60) {
        meter.style.backgroundColor = "#e11d48"; // Crimson High
    } else if (percent >= 40) {
        meter.style.backgroundColor = "#f59e0b"; // Amber Moderate
    } else {
        meter.style.backgroundColor = "#10b981"; // Green Low
    }

    document.getElementById("riskLabel").textContent = label + " – " + modelName;
    document.getElementById("riskExplanation").textContent = explanation;

    renderPatientReport(patient, percent, label, explanation, recommendations, modelName);
}


// DAY 3: REPORT GENERATION AND FRONTEND/BACKEND DATA CONTRACT
function escapeReportText(value) {
    return String(value ?? "").replace(/[&<>"']/g, character => ({
        "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"
    }[character]));
}

function renderPatientReport(patient, riskPercent, riskLabel, riskExplanation, backendRecommendations = [], modelUsed = "") {
    if (!patient) return;
    const heightM = Number(patient.height) / 100;
    const bmi = heightM > 0 ? Number(patient.weight) / (heightM * heightM) : NaN;
    const bmiCategory = Number.isFinite(bmi) ? getBMICategory(bmi).label : "Not available";
    const bpCategory = getBPCategory(Number(patient.ap_hi), Number(patient.ap_lo));
    const cholesterol = ["", "Normal", "Above normal", "Well above normal"][Number(patient.cholesterol)] || "Not entered";
    const glucose = ["", "Normal", "Above normal", "Well above normal"][Number(patient.gluc)] || "Not entered";
    const riskText = Number.isFinite(riskPercent) ? riskPercent + "%" : "Not available";
    
    let prevention = [];
    if (backendRecommendations && backendRecommendations.length > 0) {
        prevention = backendRecommendations;
    } else {
        if (Number(patient.ap_hi) >= 130 || Number(patient.ap_lo) >= 80) prevention.push("Arrange a blood-pressure review with a qualified healthcare professional and keep a log of properly taken readings.");
        else prevention.push("Continue periodic blood-pressure checks as advised by your healthcare professional.");
        if (Number.isFinite(bmi) && bmi >= 25) prevention.push("Discuss a sustainable nutrition and activity plan with a healthcare professional; BMI is only a screening measure.");
        if (Number(patient.cholesterol) > 1 || Number(patient.gluc) > 1) prevention.push("The selected cholesterol/glucose category is not a laboratory result. Consider appropriate testing and clinical review.");
        if (patient.smoke === "1") prevention.push("Seek support to stop tobacco use.");
        if (patient.active !== "1") prevention.push("If safe for you, gradually build regular physical activity with advice suited to your health.");
        prevention.push("Choose a balanced eating pattern, limit excess salt, prioritize sleep, and attend routine health check-ups.");
    }

    const report = document.getElementById("printableReport");
    if (!report) return;
    const today = new Date().toLocaleDateString();
    report.innerHTML = `
      <div class="report-header">
        <div><h2>CardioGuard – Patient Risk Report</h2><p class="report-muted">Personal health summary · Generated ${escapeReportText(today)}</p></div>
        <strong>CONFIDENTIAL</strong>
      </div>
      <h3>Patient information</h3>
      <div class="report-grid">
        <div class="report-item"><strong>Patient name</strong>${escapeReportText(patient.name)}</div>
        <div class="report-item"><strong>Age / Gender</strong>${escapeReportText(patient.age)} years / ${escapeReportText(patient.gender === "2" ? "Male" : patient.gender === "1" ? "Female" : "Other / not specified")}</div>
        <div class="report-item"><strong>Height / Weight</strong>${escapeReportText(patient.height)} cm / ${escapeReportText(patient.weight)} kg</div>
        <div class="report-item"><strong>Smoking / Alcohol</strong>${patient.smoke === "1" ? "Yes" : "No"} / ${patient.alco === "1" ? "Yes" : "No"}</div>
        <div class="report-item"><strong>Physical activity</strong>${patient.active === "1" ? "Reported active" : "Not reported active"}</div>
        <div class="report-item"><strong>Blood pressure</strong>${escapeReportText(patient.ap_hi)}/${escapeReportText(patient.ap_lo)} mmHg – ${escapeReportText(bpCategory.label)}</div>
        <div class="report-item"><strong>BMI</strong>${Number.isFinite(bmi) ? bmi.toFixed(2) + " kg/m² – " + escapeReportText(bmiCategory) : "Not available"}</div>
        <div class="report-item"><strong>Cholesterol / Glucose</strong>${escapeReportText(cholesterol)} / ${escapeReportText(glucose)}</div>
      </div>
      <h3>Machine-learning prediction</h3>
      <div class="report-item">
        <strong>Model Output: ${escapeReportText(riskText)} (${escapeReportText(riskLabel || "Assessment")})</strong>
        <p><strong>Algorithm:</strong> ${escapeReportText(modelUsed || "Trained Pipeline")}</p>
        <p>${escapeReportText(riskExplanation || "")}</p>
      </div>
      <p class="report-muted">Trained on the Cardiovascular Disease dataset (68,586 cleaned records) per PRD Section 3 & 8 (Group C). This is an educational screening estimate, not a clinically validated personal risk score.</p>
      <h3>Personalized preventive health recommendations</h3>
      <ul>${prevention.map(item => `<li>${escapeReportText(item)}</li>`).join("")}</ul>
      <div class="report-disclaimer">
        <strong>Academic & Non-Diagnostic Disclaimer</strong>
        <p>This report is generated for educational and awareness purposes as part of the BCA End-Semester Project (Group C). It does not diagnose disease, replace a physician's clinical assessment, or provide treatment advice. Do not modify medication or delay medical care based on this report. Seek urgent medical attention if experiencing severe chest pain, shortness of breath, or cardiac symptoms.</p>
      </div>
    `;
}


function getBackendPayload(patient = currentPatient) {
    if (!patient) return null;
    // JSON contract for the backend: numeric measurements and integer-coded categories.
    return {
        patient: {
            name: String(patient.name),
            age: Number(patient.age),
            gender: Number(patient.gender),
            height_cm: Number(patient.height),
            weight_kg: Number(patient.weight)
        },
        health_parameters: {
            systolic_bp_mmhg: Number(patient.ap_hi),
            diastolic_bp_mmhg: Number(patient.ap_lo),
            cholesterol_level: Number(patient.cholesterol),
            glucose_level: Number(patient.gluc),
            smoking: Number(patient.smoke),
            alcohol: Number(patient.alco),
            physically_active: Number(patient.active)
        }
    };
}

// Call this when the real backend URL and endpoint contract are supplied.
// No request is sent automatically to an invented endpoint.
async function sendPatientDataToBackend(apiUrl) {
    const payload = getBackendPayload();
    if (!payload) throw new Error("Register a patient before sending data.");
    if (!apiUrl) throw new Error("Configure the backend API URL before sending data.");
    const response = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });
    if (!response.ok) throw new Error("Backend request failed: HTTP " + response.status);
    return response.json();
}

function buildReportPlainText() {
    const report = document.getElementById("printableReport");
    if (!report) return "";
    return report.innerText || report.textContent || "";
}

function printPatientReport() {
    if (!currentPatient) {
        alert("Please register a patient before printing the report.");
        return;
    }
    window.print();
}

function downloadPatientReport() {
    if (!currentPatient) {
        alert("Please register a patient before downloading the report.");
        return;
    }
    if (!window.jspdf || !window.jspdf.jsPDF) {
        alert("The PDF library could not be loaded. Use Print Report and choose 'Save as PDF' in your print dialog.");
        return;
    }
    const doc = new window.jspdf.jsPDF({ unit: "mm", format: "a4" });
    const margin = 15;
    const pageWidth = doc.internal.pageSize.getWidth();
    const usableWidth = pageWidth - margin * 2;
    const lines = doc.splitTextToSize(buildReportPlainText(), usableWidth);
    let y = 18;
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    lines.forEach(line => {
        if (y > 280) {
            doc.addPage();
            y = 18;
        }
        doc.text(String(line), margin, y);
        y += 5.2;
    });
    const safeName = String(currentPatient.name || "patient").replace(/[^a-z0-9_-]/gi, "_");
    doc.save("CardioGuard_Risk_Report_" + safeName + ".pdf");
}

// DAY 2: BLOOD PRESSURE AND BMI CALCULATIONS
function getBPCategory(systolic, diastolic) {
    if (systolic > 180 || diastolic > 120) {
        return {label:"Severely high reading", cls:"status-high",
            note:"Repeat the measurement after at least one minute. If it remains this high, contact a healthcare professional. With concerning symptoms, seek emergency help."};
    }
    if (systolic >= 140 || diastolic >= 90) {
        return {label:"Stage 2 hypertension range", cls:"status-high",
            note:"This reading is in the Stage 2 range. Repeat it correctly and discuss the result with a healthcare professional."};
    }
    if (systolic >= 130 || diastolic >= 80) {
        return {label:"Stage 1 hypertension range", cls:"status-warn",
            note:"This reading is in the Stage 1 range. Recheck and discuss repeated readings with a healthcare professional."};
    }
    if (systolic >= 120 && systolic <= 129 && diastolic < 80) {
        return {label:"Elevated range", cls:"status-warn",
            note:"This reading is in the elevated range. Heart-healthy habits and follow-up readings may be helpful."};
    }
    if (systolic < 120 && diastolic < 80) {
        return {label:"Normal range", cls:"status-good",
            note:"This reading is in the normal adult range."};
    }
    return {label:"Outside standard categories", cls:"status-info",
        note:"This tool does not classify this reading. Discuss it with a healthcare professional, especially if you feel unwell."};
}

function calculateBloodPressure() {
    const sys = Number(document.getElementById("calcSystolic").value);
    const dia = Number(document.getElementById("calcDiastolic").value);
    const result = document.getElementById("bpResult");
    if (!sys || !dia || sys < 50 || sys > 300 || dia < 30 || dia > 200 || sys <= dia) {
        result.className = "calc-result visible";
        result.innerHTML = "<strong>Check your entries.</strong><br>Enter valid readings and make sure systolic is higher than diastolic.";
        return;
    }
    const category = getBPCategory(sys, dia);
    result.className = "calc-result visible";
    result.innerHTML = "<strong>Your reading:</strong> " + sys + "/" + dia + " mmHg<br><strong>Category:</strong> " + category.label + "<br>" + category.note;
    updateBPCar(sys, dia);
}

function getBMICategory(bmi) {
    if (bmi < 18.5) return {label:"Underweight", cls:"status-warn"};
    if (bmi < 25) return {label:"Healthy weight range", cls:"status-good"};
    if (bmi < 30) return {label:"Overweight", cls:"status-warn"};
    return {label:"Obesity range", cls:"status-high"};
}

function calculateBMI(updateCard) {
    const heightCm = Number(document.getElementById("calcHeight").value);
    const weightKg = Number(document.getElementById("calcWeight").value);
    const result = document.getElementById("bmiResult");
    if (!heightCm || !weightKg) {
        result.textContent = "Enter your height and weight to calculate BMI.";
        result.className = "calc-result visible";
        return null;
    }
    if (heightCm < 50 || heightCm > 250 || weightKg < 10 || weightKg > 500) {
        result.textContent = "Please enter a valid height (50–250 cm) and weight (10–500 kg).";
        result.className = "calc-result visible";
        return null;
    }
    const bmi = weightKg / Math.pow(heightCm / 100, 2);
    const category = getBMICategory(bmi);
    result.className = "calc-result visible";
    result.innerHTML = "<strong>Your BMI:</strong> " + bmi.toFixed(2) + " kg/m²<br><strong>Category:</strong> " + category.label +
        "<br><small>BMI is a screening measure, not a diagnosis.</small>";
    if (updateCard !== false) updateBMICard(bmi, category);
    return bmi;
}

function setBiomarker(id, value, status, cls) {
    const valueEl = document.getElementById(id);
    const statusEl = document.getElementById(id + "Status");
    if (!valueEl || !statusEl) return;
    valueEl.textContent = value;
    statusEl.textContent = status;
    statusEl.className = "biomarker-status " + (cls || "status-info");
}

function updateBPCar(sys, dia) {
    const category = getBPCategory(sys, dia);
    setBiomarker("bioBP", sys + "/" + dia, category.label, category.cls);
}
function updateBMICard(bmi, category) {
    setBiomarker("bioBMI", bmi.toFixed(2), category.label, category.cls);
}

function updateBiomarkers(patient) {
    if (!patient) return;
    const age = Number(patient.age);
    setBiomarker("bioAge", age + " yrs", "Age entered", "status-info");

    const sys = Number(patient.ap_hi), dia = Number(patient.ap_lo);
    if (sys > 0 && dia > 0) updateBPCar(sys, dia);

    const height = Number(patient.height), weight = Number(patient.weight);
    if (height > 0 && weight > 0) {
        const bmi = weight / Math.pow(height / 100, 2);
        updateBMICard(bmi, getBMICategory(bmi));
        document.getElementById("calcHeight").value = patient.height;
        document.getElementById("calcWeight").value = patient.weight;
        document.getElementById("bmiResult").innerHTML = "<strong>Your BMI:</strong> " + bmi.toFixed(2) + " kg/m²<br><strong>Category:</strong> " + getBMICategory(bmi).label;
    }

    const levels = ["", "Normal", "Above normal", "Well above normal"];
    const chol = levels[Number(patient.cholesterol)] || "Not entered";
    const gluc = levels[Number(patient.gluc)] || "Not entered";
    setBiomarker("bioCholesterol", chol, chol === "Normal" ? "Selected level: normal" : "Selected category only", chol === "Normal" ? "status-good" : "status-info");
    setBiomarker("bioGlucose", gluc, gluc === "Normal" ? "Selected level: normal" : "Selected category only", gluc === "Normal" ? "status-good" : "status-info");
    const smoking = patient.smoke === "1";
    setBiomarker("bioSmoking", smoking ? "Yes" : "No", smoking ? "Tobacco use reported" : "No tobacco use reported", smoking ? "status-warn" : "status-good");
    const active = patient.active === "1";
    setBiomarker("bioActivity", active ? "Yes" : "No", active ? "Physical activity reported" : "No activity reported", active ? "status-good" : "status-warn");

    document.getElementById("calcSystolic").value = patient.ap_hi;
    document.getElementById("calcDiastolic").value = patient.ap_lo;
    const bpCategory = getBPCategory(sys, dia);
    document.getElementById("bpResult").className = "calc-result visible";
    document.getElementById("bpResult").innerHTML = "<strong>Your reading:</strong> " + sys + "/" + dia + " mmHg<br><strong>Category:</strong> " + bpCategory.label + "<br>" + bpCategory.note;
}

// BMI recalculates whenever either value changes.
document.getElementById("calcHeight").addEventListener("input", function() { calculateBMI(true); });
document.getElementById("calcWeight").addEventListener("input", function() { calculateBMI(true); });

// REGISTRATION FORM — require every field, then show the dashboard with this patient's details.
const registrationForm = document.getElementById("registrationForm");

registrationForm.addEventListener("submit", async function(event) {
    event.preventDefault();

    // Do not open the dashboard until every required field is valid.
    if (!registrationForm.checkValidity()) {
        registrationForm.reportValidity();
        return;
    }

    const patient = getPatientFromForm();
    const requiredValues = [
        patient.name, patient.age, patient.gender, patient.height, patient.weight,
        patient.ap_hi, patient.ap_lo, patient.cholesterol, patient.gluc,
        patient.smoke, patient.alco, patient.active
    ];

    if (requiredValues.some(value => value === "" || value === null || value === undefined)) {
        alert("Please complete all patient details before continuing.");
        return;
    }

    const systolic = Number(patient.ap_hi);
    const diastolic = Number(patient.ap_lo);
    if (systolic <= diastolic) {
        alert("Systolic blood pressure must be higher than diastolic blood pressure. Please check both values.");
        document.getElementById("ap_hi").focus();
        return;
    }

    // Store the newly entered patient and populate the dashboard before displaying it.
    currentPatient = patient;
    try {
        localStorage.setItem("patientData", JSON.stringify(patient));
    } catch (storageError) {
        console.warn("Patient details could not be saved in this browser:", storageError);
    }

    try {
        renderPatient(patient);
        await renderRisk(patient);
        homePage.style.display = "none";
        dashboardPage.style.display = "flex";
        showSection("dashboardContent");
        window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (dashboardError) {
        console.error("Dashboard could not be opened:", dashboardError);
        homePage.style.display = "block";
        dashboardPage.style.display = "none";
        alert("The patient details were accepted, but the dashboard could not load. Please refresh the page and try again.");
    }
});

// Open About or Health Tips as a separate full page from the home navigation.
function openHomeSection(event, sectionId) {
    event.preventDefault();
    const sourceSection = document.getElementById(sectionId);
    if (!sourceSection) return;

    let standalonePage = document.getElementById("standaloneInfoPage");
    if (!standalonePage) {
        standalonePage = document.createElement("section");
        standalonePage.id = "standaloneInfoPage";
        standalonePage.innerHTML = `
            <header class="standalone-info-header">
                <h1 id="standaloneInfoTitle">CardioGuard</h1>
                <button type="button" class="standalone-back-btn" onclick="closeStandaloneInfoPage()">← Back to Home</button>
            </header>
            <main id="standaloneInfoContent"></main>`;
        document.body.appendChild(standalonePage);
    }

    const title = sectionId === "aboutContent" ? "About CardioGuard" : "Heart Health Tips";
    document.getElementById("standaloneInfoTitle").textContent = title;
    const originalCard = sourceSection.querySelector(".content-card");
    const contentArea = document.getElementById("standaloneInfoContent");
    contentArea.innerHTML = "";
    if (originalCard) {
        const copiedCard = originalCard.cloneNode(true);
        copiedCard.querySelectorAll("[id]").forEach(element => element.removeAttribute("id"));
        contentArea.appendChild(copiedCard);
    }

    homePage.style.display = "none";
    dashboardPage.style.display = "none";
    standalonePage.style.display = "block";
    window.scrollTo({ top: 0, behavior: "smooth" });
}

// Return from the separate About / Health Tips page to the home page.
function closeStandaloneInfoPage() {
    const standalonePage = document.getElementById("standaloneInfoPage");
    if (standalonePage) standalonePage.style.display = "none";
    homePage.style.display = "block";
    dashboardPage.style.display = "none";
    window.scrollTo({ top: 0, behavior: "smooth" });
}

// Toggle the dropdown beside the patient's name.
function toggleUserDropdown(event) {
    event.stopPropagation();
    const dropdown = document.getElementById("userDropdown");
    const toggle = document.getElementById("userMenuToggle");
    const isOpen = dropdown.classList.toggle("show");
    toggle.setAttribute("aria-expanded", String(isOpen));
}

function closeUserDropdown() {
    const dropdown = document.getElementById("userDropdown");
    const toggle = document.getElementById("userMenuToggle");
    if (dropdown && toggle) {
        dropdown.classList.remove("show");
        toggle.setAttribute("aria-expanded", "false");
    }
}

function openPatientDetails() {
    closeUserDropdown();
    showSection("patientContent");
}

document.addEventListener("click", function(event) {
    const menu = document.getElementById("userMenu");
    if (menu && !menu.contains(event.target)) closeUserDropdown();
});

// Return from the dashboard to the patient registration page.
function backToRegistration() {
    dashboardPage.style.display = "none";
    homePage.style.display = "block";
    window.scrollTo({ top: 0, behavior: "smooth" });
}

// OPEN DASHBOARD
function showDashboard(initialSection = "dashboardContent") {
    homePage.style.display = "none";
    dashboardPage.style.display = "flex";

    if (!currentPatient) {
        const savedData = localStorage.getItem("patientData");
        if (savedData) {
            try { currentPatient = JSON.parse(savedData); }
            catch (error) { currentPatient = null; }
        }
    }

    if (currentPatient) {
        renderPatient(currentPatient);
        renderRisk(currentPatient);
    }
    showSection(initialSection);
}

// SIDEBAR NAVIGATION

function showSection(sectionId, button) {

    const sections =
    document.querySelectorAll(".page-section");

    // HIDE ALL SECTIONS

    sections.forEach(function(section) {

        section.style.display = "none";

    });


    // SHOW SELECTED SECTION

    document.getElementById(sectionId)
    .style.display = "block";


    // REMOVE ACTIVE CLASS

    const buttons =
    document.querySelectorAll(".menu button");

    buttons.forEach(function(btn) {

        btn.classList.remove("active");

    });


    // ACTIVATE SELECTED SIDEBAR BUTTON

    if (button) {

        button.classList.add("active");

    } else {

        const sectionButtons = {
            dashboardContent: 0,
            patientContent: 1,
            riskContent: 2,
            calculatorContent: 3,
            healthContent: 4,
            aboutContent: 5
        };

        const index = sectionButtons[sectionId];

        if (index !== undefined) {
            buttons[index].classList.add("active");
        }

    }

}

/* ===== Extracted inline script block ===== */

(function () {
  const numericFieldIds = [
    "height", "weight", "ap_hi", "ap_lo",
    "calcHeight", "calcWeight", "calcSystolic", "calcDiastolic"
  ];

  numericFieldIds.forEach(function (id) {
    const field = document.getElementById(id);
    if (!field) return;

    field.setAttribute("inputmode", "decimal");
    field.addEventListener("keydown", function (event) {
      if (["e", "E", "+", "-"].includes(event.key)) {
        event.preventDefault();
      }
    });
  });
})();

/* ===== Extracted inline script block ===== */

function playNotificationBell(event) {
    if (event) event.stopPropagation();

    const bell = document.getElementById('notificationBell');
    if (bell) {
        bell.classList.remove('is-ringing');
        void bell.offsetWidth;
        bell.classList.add('is-ringing');
        window.setTimeout(() => bell.classList.remove('is-ringing'), 1800);
    }

    // Short, metallic notification chime. Audio starts from the bell click,
    // which satisfies browser autoplay restrictions.
    try {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) return;

        const audio = new AudioContextClass();

        const playChime = () => {
            const startAt = audio.currentTime + 0.025;

            // Layered bell partials create a bright, resonant "ding-ding".
            const strikes = [
                { time: 0.00, frequency: 880, duration: 1.25 },
                { time: 0.34, frequency: 1174.66, duration: 1.35 }
            ];

            strikes.forEach((strike) => {
                [1, 2.01, 2.76, 3.92].forEach((partial, index) => {
                    const oscillator = audio.createOscillator();
                    const gain = audio.createGain();
                    const at = startAt + strike.time;
                    const level = [0.105, 0.043, 0.022, 0.012][index];

                    oscillator.type = index === 2 ? 'triangle' : 'sine';
                    oscillator.frequency.setValueAtTime(
                        strike.frequency * partial, at
                    );

                    gain.gain.setValueAtTime(0.0001, at);
                    gain.gain.exponentialRampToValueAtTime(level, at + 0.012);
                    gain.gain.exponentialRampToValueAtTime(
                        0.0001, at + strike.duration
                    );

                    oscillator.connect(gain);
                    gain.connect(audio.destination);
                    oscillator.start(at);
                    oscillator.stop(at + strike.duration + 0.04);
                });
            });

            window.setTimeout(() => {
                audio.close().catch(() => {});
            }, 1900);
        };

        if (audio.state === 'suspended') {
            audio.resume().then(playChime).catch(() => {
                audio.close().catch(() => {});
            });
        } else {
            playChime();
        }
    } catch (error) {
        console.info('Notification sound is unavailable in this browser.');
    }
}

/* ===== Extracted inline script block ===== */

function toggleNotificationPanel(event) {
    if (event) event.stopPropagation();
    playNotificationBell();
    const panel = document.getElementById('notificationPanel');
    const bell = document.getElementById('notificationBell');
    if (!panel || !bell) return;
    const opening = !panel.classList.contains('is-open');
    panel.classList.toggle('is-open', opening);
    panel.setAttribute('aria-hidden', String(!opening));
    bell.setAttribute('aria-expanded', String(opening));
    if (opening) {
        closeUserDropdown();
    }
}

function markNotificationRead(item) {
    if (!item) return;
    item.classList.remove('unread');
    item.dataset.unread = 'false';
    const dot = item.querySelector('.notification-unread-dot');
    if (dot) dot.remove();
    updateNotificationCount();
}

function markAllNotificationsRead() {
    document.querySelectorAll('#notificationList .notification-item.unread').forEach(markNotificationRead);
}

function clearNotifications() {
    const list = document.getElementById('notificationList');
    if (list) list.innerHTML = '<div class="notification-empty">You’re all caught up. No notifications.</div>';
    updateNotificationCount();
}

function updateNotificationCount() {
    const count = document.querySelectorAll('#notificationList .notification-item.unread').length;
    const badge = document.getElementById('notificationCount');
    const dot = document.querySelector('#notificationBell .bell-dot');
    if (badge) badge.textContent = count === 1 ? '1 new' : count + ' new';
    if (dot) dot.style.display = count ? 'block' : 'none';
}

function closeNotificationPanel() {
    const panel = document.getElementById('notificationPanel');
    const bell = document.getElementById('notificationBell');
    if (!panel || !bell) return;
    panel.classList.remove('is-open');
    panel.setAttribute('aria-hidden', 'true');
    bell.setAttribute('aria-expanded', 'false');
}

document.addEventListener('click', function(event) {
    const menu = document.getElementById('userMenu');
    if (menu && !menu.contains(event.target)) closeNotificationPanel();
});
