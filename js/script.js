// ==========================================
// CardioGuard - Health Portal Logic & Analytics
// ==========================================

// Calculate BMI automatically and update badge
function calculateBMI() {
    const heightInput = document.getElementById("height");
    const weightInput = document.getElementById("weight");
    const bmiInput = document.getElementById("bmi");
    const bmiBadge = document.getElementById("bmiBadge");

    if (heightInput && weightInput && bmiInput && heightInput.value && weightInput.value) {
        const heightM = parseFloat(heightInput.value) / 100;
        const weight = parseFloat(weightInput.value);
        if (heightM > 0 && weight > 0) {
            const bmi = (weight / (heightM * heightM)).toFixed(1);
            bmiInput.value = bmi;

            if (bmiBadge) {
                if (bmi < 18.5) {
                    bmiBadge.innerText = "Underweight";
                    bmiBadge.style.backgroundColor = "#e0f2fe";
                    bmiBadge.style.color = "#0284c7";
                } else if (bmi <= 24.9) {
                    bmiBadge.innerText = "Healthy Weight";
                    bmiBadge.style.backgroundColor = "#dcfce7";
                    bmiBadge.style.color = "#15803d";
                } else if (bmi <= 29.9) {
                    bmiBadge.innerText = "Overweight";
                    bmiBadge.style.backgroundColor = "#fef3c7";
                    bmiBadge.style.color = "#b45309";
                } else {
                    bmiBadge.innerText = "Obese";
                    bmiBadge.style.backgroundColor = "#fee2e2";
                    bmiBadge.style.color = "#b91c1c";
                }
            }
        }
    }
}

// Handle Patient Form Submission & AI Prediction
document.getElementById("registrationForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const payload = {
        General_Health: document.getElementById("General_Health").value,
        Checkup: document.getElementById("Checkup").value,
        Exercise: document.getElementById("Exercise").value,
        Diabetes: document.getElementById("Diabetes").value,
        Depression: document.getElementById("Depression").value,
        Age_Category: document.getElementById("Age_Category").value,
        Sex: document.getElementById("Sex").value,
        Smoking_History: document.getElementById("Smoking_History").value,
        Skin_Cancer: document.getElementById("Skin_Cancer").value,
        Other_Cancer: document.getElementById("Other_Cancer").value,
        Arthritis: document.getElementById("Arthritis").value,
        BMI: parseFloat(document.getElementById("bmi").value),
        Alcohol_Consumption: parseFloat(document.getElementById("Alcohol_Consumption").value),
        Fruit_Consumption: parseFloat(document.getElementById("Fruit_Consumption").value),
        Green_Vegetables_Consumption: parseFloat(document.getElementById("Green_Vegetables_Consumption").value),
        FriedPotato_Consumption: parseFloat(document.getElementById("FriedPotato_Consumption").value),
        Height_cm: parseFloat(document.getElementById("height").value),
        Weight_kg: parseFloat(document.getElementById("weight").value)
    };

    const name = document.getElementById("name").value.trim() || "Patient";
    const btn = document.getElementById("getStartedButton");
    const originalBtnHtml = btn.innerHTML;

    try {
        btn.innerHTML = `<span>Analyzing Clinical Signals...</span>`;
        btn.disabled = true;

        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error("API Error");

        const data = await response.json();
        btn.innerHTML = originalBtnHtml;
        btn.disabled = false;

        document.getElementById("homePage").style.display = "none";
        document.getElementById("dashboardPage").style.display = "flex";
        
        // Update user display & avatar badge
        const userDisplay = document.getElementById("userDisplay");
        if (userDisplay) userDisplay.innerText = name;
        
        const userInitial = document.getElementById("userInitial");
        if (userInitial) userInitial.innerText = name.charAt(0).toUpperCase();

        document.getElementById("riskPatientName").innerText = `Clinical Assessment for: ${name}`;

        // Format Patient Intake Dossier
        document.getElementById("patientDetails").innerHTML = `
            <div class="patient-summary-grid">
                <div class="patient-summary-item">
                    <span class="label">Patient Name</span>
                    <span class="value">${name}</span>
                </div>
                <div class="patient-summary-item">
                    <span class="label">Age Cohort</span>
                    <span class="value">${payload.Age_Category}</span>
                </div>
                <div class="patient-summary-item">
                    <span class="label">Biological Sex</span>
                    <span class="value">${payload.Sex}</span>
                </div>
                <div class="patient-summary-item">
                    <span class="label">General Health</span>
                    <span class="value">${payload.General_Health}</span>
                </div>
                <div class="patient-summary-item">
                    <span class="label">Calculated BMI</span>
                    <span class="value">${payload.BMI}</span>
                </div>
                <div class="patient-summary-item">
                    <span class="label">Smoking History</span>
                    <span class="value">${payload.Smoking_History}</span>
                </div>
                <div class="patient-summary-item">
                    <span class="label">Exercise Routine</span>
                    <span class="value">${payload.Exercise}</span>
                </div>
                <div class="patient-summary-item">
                    <span class="label">Diabetes Status</span>
                    <span class="value">${payload.Diabetes}</span>
                </div>
            </div>
        `;

        // Update Risk Score and Animated Meter
        const riskScore = (data.probability * 100).toFixed(1);
        document.getElementById("riskScore").innerText = riskScore + "%";
        document.getElementById("riskLabel").innerText = data.risk_level + " Cardiovascular Risk";
        
        const meter = document.getElementById("riskMeter");
        meter.style.width = "0%";
        setTimeout(() => {
            meter.style.width = riskScore + "%";
        }, 150);

        if (data.risk_level === "High") {
            meter.style.backgroundColor = "#ef4444";
            document.getElementById("riskLabel").style.color = "#ef4444";
        } else if (data.risk_level === "Moderate") {
            meter.style.backgroundColor = "#f59e0b";
            document.getElementById("riskLabel").style.color = "#f59e0b";
        } else {
            meter.style.backgroundColor = "#10b981";
            document.getElementById("riskLabel").style.color = "#10b981";
        }

        document.getElementById("riskExplanation").innerText = data.message;

        showSection('riskContent', document.querySelector('.menu button:nth-child(3)'));

    } catch (error) {
        alert("Error connecting to backend. Make sure your FastAPI server is running with 'python -m uvicorn main:app --reload'!");
        console.error(error);
        btn.innerHTML = originalBtnHtml;
        btn.disabled = false;
    }
});

// ==========================================
// 1. SCANNER SYSTEM (OCR)
// ==========================================
async function scanDocument(event) {
    const file = event.target.files[0];
    if (!file) return;

    const statusText = document.getElementById("scanStatus");
    statusText.innerText = "Scanning document slip with OCR... Please wait ~5 seconds.";
    statusText.style.color = "#0284c7";

    try {
        const result = await Tesseract.recognize(file, 'eng');
        const text = result.data.text.toLowerCase();
        
        let fieldsFound = 0;

        if (text.includes("male") && !text.includes("female")) { 
            document.getElementById("Sex").value = "Male"; 
            fieldsFound++; 
        } else if (text.includes("female")) { 
            document.getElementById("Sex").value = "Female"; 
            fieldsFound++; 
        }

        if (text.includes("diabetes") || text.includes("diabetic")) { 
            document.getElementById("Diabetes").value = "Yes"; 
            fieldsFound++; 
        }
        if (text.includes("smoker") || text.includes("smoke")) { 
            document.getElementById("Smoking_History").value = "Yes"; 
            fieldsFound++; 
        }
        
        const ageMatch = text.match(/age[^\d]*(\d{2})/);
        if (ageMatch) {
            const age = parseInt(ageMatch[1]);
            if (age >= 18 && age <= 24) document.getElementById("Age_Category").value = "18-24";
            else if (age >= 25 && age <= 29) document.getElementById("Age_Category").value = "25-29";
            else if (age >= 30 && age <= 34) document.getElementById("Age_Category").value = "30-34";
            else if (age >= 35 && age <= 39) document.getElementById("Age_Category").value = "35-39";
            else if (age >= 40 && age <= 44) document.getElementById("Age_Category").value = "40-44";
            else if (age >= 45 && age <= 49) document.getElementById("Age_Category").value = "45-49";
            else if (age >= 50 && age <= 54) document.getElementById("Age_Category").value = "50-54";
            else if (age >= 55) document.getElementById("Age_Category").value = "55-59";
            fieldsFound++;
        }

        // Height & Weight regex check
        const heightMatch = text.match(/(?:height|ht)[\s:]*(\d{3})/i);
        if (heightMatch) {
            document.getElementById("height").value = heightMatch[1];
            fieldsFound++;
        }

        const weightMatch = text.match(/(?:weight|wt)[\s:]*(\d{2,3})/i);
        if (weightMatch) {
            document.getElementById("weight").value = weightMatch[1];
            fieldsFound++;
        }

        calculateBMI();

        if (fieldsFound > 0) {
            statusText.innerText = `Scan complete! Successfully extracted and auto-filled ${fieldsFound} fields.`;
            statusText.style.color = "#10b981";
        } else {
            statusText.innerText = "Notice: Could not parse specific clinical fields. Please input manually.";
            statusText.style.color = "#f59e0b";
        }

    } catch (error) {
        console.error(error);
        statusText.innerText = "Error scanning image. Please ensure image is well-lit and legible.";
        statusText.style.color = "#ef4444";
    }
}

// ==========================================
// 2. DOWNLOAD PDF REPORT
// ==========================================
function downloadPDF() {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();

    const patientName = document.getElementById("userDisplay").innerText || "Patient";
    const riskScore = document.getElementById("riskScore").innerText || "--%";
    const riskLabel = document.getElementById("riskLabel").innerText || "Unestimated";
    const riskExplanation = document.getElementById("riskExplanation").innerText || "";
    
    const age = document.getElementById("Age_Category").value || "N/A";
    const sex = document.getElementById("Sex").value || "N/A";
    const bmi = document.getElementById("bmi").value || "N/A";
    const smoking = document.getElementById("Smoking_History").value || "N/A";
    const diabetes = document.getElementById("Diabetes").value || "N/A";
    const health = document.getElementById("General_Health").value || "N/A";

    // Header styling
    doc.setFillColor(9, 18, 34);
    doc.rect(0, 0, 210, 32, "F");

    doc.setFont("helvetica", "bold");
    doc.setFontSize(20);
    doc.setTextColor(255, 255, 255);
    doc.text("CardioGuard Clinical AI Report", 105, 18, null, null, "center");

    doc.setFontSize(9);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(56, 189, 248);
    doc.text("BRFSS Machine Learning Stratification Platform", 105, 26, null, null, "center");
    
    // Patient Information Card
    doc.setDrawColor(226, 232, 240);
    doc.setFillColor(248, 250, 252);
    doc.roundedRect(15, 42, 180, 52, 3, 3, "FD");

    doc.setFont("helvetica", "bold");
    doc.setFontSize(13);
    doc.setTextColor(11, 21, 40);
    doc.text("Patient Intake Summary", 22, 52);
    
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10.5);
    doc.setTextColor(71, 85, 105);
    doc.text(`Patient Name: ${patientName}`, 22, 62);
    doc.text(`Age Cohort: ${age}`, 22, 70);
    doc.text(`Biological Sex: ${sex}`, 22, 78);
    doc.text(`General Health: ${health}`, 22, 86);

    doc.text(`Body Mass Index (BMI): ${bmi}`, 110, 62);
    doc.text(`Smoking History: ${smoking}`, 110, 70);
    doc.text(`Diabetes Diagnosis: ${diabetes}`, 110, 78);

    // AI Assessment Card
    doc.setFillColor(255, 255, 255);
    doc.roundedRect(15, 104, 180, 60, 3, 3, "FD");

    doc.setFont("helvetica", "bold");
    doc.setFontSize(13);
    doc.setTextColor(11, 21, 40);
    doc.text("AI Risk Evaluation", 22, 116);
    
    doc.setFontSize(15);
    if (riskLabel.includes("High")) {
        doc.setTextColor(239, 68, 68);
    } else if (riskLabel.includes("Moderate")) {
        doc.setTextColor(245, 158, 11);
    } else {
        doc.setTextColor(16, 185, 129);
    }

    doc.text(`Classification: ${riskLabel}`, 22, 128);
    
    doc.setTextColor(15, 23, 42);
    doc.setFontSize(11);
    doc.setFont("helvetica", "bold");
    doc.text(`Calculated Model Probability: ${riskScore}`, 22, 138);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor(71, 85, 105);
    doc.text(`Clinical Indication: ${riskExplanation}`, 22, 148);

    // Footer Disclaimer
    doc.setFont("helvetica", "italic");
    doc.setFontSize(8.5);
    doc.setTextColor(148, 163, 184);
    doc.text("Disclaimer: CardioGuard is an automated AI educational assessment and does not constitute a formal diagnosis.", 105, 185, null, null, "center");
    doc.text("Always consult licensed medical practitioners for cardiovascular evaluation.", 105, 191, null, null, "center");

    doc.save(`${patientName.replace(/\s+/g, '_')}_CardioGuard_Report.pdf`);
}

// ==========================================
// 3. UI NAVIGATION
// ==========================================
function showSection(sectionId, btn) {
    document.querySelectorAll('.page-section').forEach(sec => sec.style.display = 'none');
    
    const target = document.getElementById(sectionId);
    if (target) {
        target.style.display = 'block';
    }
    
    const menuButtons = document.querySelectorAll('.menu button');
    menuButtons.forEach(b => b.classList.remove('active'));

    if (btn) {
        btn.classList.add('active');
    } else {
        // Automatically find and highlight the button that corresponds to sectionId
        menuButtons.forEach(b => {
            const clickAttr = b.getAttribute('onclick') || '';
            if (clickAttr.includes(sectionId)) {
                b.classList.add('active');
            }
        });
    }
}

function backToRegistration() {
    document.getElementById("dashboardPage").style.display = "none";
    document.getElementById("homePage").style.display = "block";
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function openPatientDetails() {
    showSection('patientContent', document.querySelector('.menu button:nth-child(2)'));
    const dropdown = document.getElementById("userDropdown");
    if (dropdown) dropdown.style.display = "none";
}

function toggleUserDropdown(event) {
    event.stopPropagation();
    const dropdown = document.getElementById("userDropdown");
    if (dropdown) {
        dropdown.style.display = dropdown.style.display === "block" ? "none" : "block";
    }
}

function openHomeSection(event, sectionId) {
    if (event) event.preventDefault();
    document.getElementById("homePage").style.display = "none";
    document.getElementById("dashboardPage").style.display = "flex";
    showSection(sectionId, null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

window.onclick = function(event) {
    if (!event.target.closest('.user-menu-toggle')) {
        const dropdown = document.getElementById("userDropdown");
        if (dropdown && dropdown.style.display === 'block') {
            dropdown.style.display = 'none';
        }
    }
};

// Initial BMI check on load if fields have default values
window.addEventListener("DOMContentLoaded", () => {
    calculateBMI();
});
