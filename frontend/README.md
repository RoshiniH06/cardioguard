# CardioGuard - Standalone Frontend

This folder contains the complete, self-contained **CardioGuard** browser frontend. It is completely decoupled from Python and can be published independently on any static web hosting platform (GitHub Pages, Netlify, Vercel, Cloudflare Pages, Firebase Hosting, AWS S3, etc.).

---

## 📁 Folder Structure

```text
frontend/
├── index.html          <- Main application HTML (Intake, Dashboard, Calculators, Reports)
├── css/
│   └── styles.css      <- Complete stylesheet with embedded visual assets (Zero external image dependencies)
├── js/
│   ├── config.js       <- Configurable backend API URL setting
│   └── script.js       <- Application controller (API client, PDF generator, OCR scanner)
└── README.md           <- Deployment instructions (this file)
```

---

## ⚡ How to Run Locally (No Backend Required)

1. **Directly in Browser:** Double-click `index.html` to open it in Chrome, Edge, or Firefox.
2. **Using VS Code Live Server:** Right-click `index.html` and choose **"Open with Live Server"**.
3. **Using Python Simple Server:**
   ```powershell
   cd frontend
   python -m http.server 3000
   ```
   Open [http://127.0.0.1:3000](http://127.0.0.1:3000).

> **Offline / Standalone Mode:**  
> If the Python FastAPI backend is not running, the portal **automatically falls back** to a client-side mathematical logistic regression model. You will still receive immediate risk scores, visual meter updates, biomarker evaluations, and printable PDF reports!

---

## 🌐 How to Connect to the Backend

When running with the Python FastAPI backend, CardioGuard supports dynamic configuration:

### Method 1: Interactive UI (No Code Changes)
1. Open the portal in your browser.
2. Click the **"API: Auto"** pill in the top navigation bar.
3. Enter your deployed backend URL (e.g., `https://your-backend.onrender.com` or `http://127.0.0.1:8000`).
4. Click **Test Connection**, then click **Save & Apply**.  
   *The URL is saved in your browser's local storage and used for all future predictions.*

### Method 2: Configure in `js/config.js`
Open `js/config.js` and edit the `API_BASE_URL` property:
```javascript
window.CARDIOGUARD_CONFIG = {
    API_BASE_URL: "https://your-cardioguard-backend.onrender.com",
    // ...
};
```

---

## 🚀 How to Publish Separately

### 1. Deploy on Netlify (Easiest - 1 Minute)
1. Go to [app.netlify.com](https://app.netlify.com).
2. Drag and drop this `frontend/` folder into Netlify Drop.
3. Your frontend is live instantly with an HTTPS link (e.g. `https://cardioguard.netlify.app`)!

### 2. Deploy on GitHub Pages
1. Push this `frontend/` folder to a GitHub repository (e.g. in `gh-pages` branch or root).
2. Go to **Settings > Pages**.
3. Under **Branch**, select `main` (or `gh-pages`) and root `/`, then click **Save**.
4. Your site will be published at `https://<your-username>.github.io/<repo-name>/`.

### 3. Deploy on Vercel
1. Install Vercel CLI: `npm i -g vercel` or link your GitHub repo at [vercel.com](https://vercel.com).
2. Run `vercel` inside this `frontend/` folder.
3. Follow the prompts to deploy.

---

## 🩺 Features Included

- **Patient Intake Form:** Validated inputs (age, gender, height, weight, BP, cholesterol, glucose, smoking, alcohol, physical activity).
- **OCR Patient Card Scanner:** Client-side camera and document scanning via Tesseract.js.
- **Dynamic Dashboard:** Real-time biomarker status cards, average health statistics, and weekly trend graphs.
- **Interactive Calculators:** AHA Blood Pressure stage calculator and CDC Body Mass Index (BMI) calculator.
- **Multi-Model Inference:** Real-time prediction with Primary Serving Ensemble, Support Vector Classifier (SVC), and K-Nearest Neighbors (KNN).
- **Clinical Health Report:** Generates formatted, printable, and downloadable PDF risk reports via jsPDF.
- **Academic Medical Disclaimer:** Clear non-diagnostic warnings per PRD specifications.
