/**
 * CardioGuard - Frontend Configuration
 * ====================================
 * Use this file to configure the backend API endpoint when deploying the frontend
 * independently (e.g. on GitHub Pages, Netlify, Vercel, or Cloudflare Pages).
 *
 * HOW IT WORKS:
 * 1. If you leave API_BASE_URL as "", CardioGuard automatically detects the URL:
 *    - If served by FastAPI on port 8000: uses relative paths ("/predict", "/health")
 *    - If opened locally or hosted statically: defaults to "http://127.0.0.1:8000"
 * 2. If you deploy your backend to the cloud (e.g. Render, Railway, AWS, Fly.io),
 *    set API_BASE_URL to your cloud URL, e.g.:
 *    API_BASE_URL: "https://your-cardioguard-backend.onrender.com"
 * 3. You can also change the API URL directly in the UI at runtime using the
 *    "⚙️ API Settings" button without modifying any code!
 */

window.CARDIOGUARD_CONFIG = {
    // Set to your backend URL (e.g., "http://127.0.0.1:8000" or "https://api.yourdomain.com").
    // Leave empty "" for automatic detection.
    API_BASE_URL: "",

    // System info
    APP_NAME: "CardioGuard",
    APP_VERSION: "1.0.0",
    DEFAULT_MODEL: "primary",

    /**
     * Resolves the full URL for any API endpoint.
     * Priority:
     * 1. User runtime setting in localStorage ('cardioguard_api_url')
     * 2. Configured API_BASE_URL above
     * 3. Same-origin if running under FastAPI
     * 4. Fallback to http://127.0.0.1:8000 for static hosting / file://
     */
    getApiUrl: function(endpoint) {
        if (!endpoint.startsWith("/")) {
            endpoint = "/" + endpoint;
        }

        // 1. Check runtime localStorage override
        const storedUrl = localStorage.getItem("cardioguard_api_url");
        if (storedUrl && storedUrl.trim() !== "") {
            return storedUrl.trim().replace(/\/+$/, "") + endpoint;
        }

        // 2. Check config object
        if (this.API_BASE_URL && this.API_BASE_URL.trim() !== "") {
            return this.API_BASE_URL.trim().replace(/\/+$/, "") + endpoint;
        }

        // 3. Auto-detection: if hosted together on same origin or port 8000
        if (window.location.port === "8000" || (window.location.origin && window.location.origin !== "null" && window.location.protocol.startsWith("http") && !window.location.hostname.includes("github.io") && !window.location.hostname.includes("netlify.app") && !window.location.hostname.includes("vercel.app") && !window.location.hostname.includes("pages.dev"))) {
            return endpoint;
        }

        // 4. Fallback for static hosting (GitHub Pages, Netlify) or file://
        return "http://127.0.0.1:8000" + endpoint;
    }
};
