/**
 * CardioGuard - Frontend Configuration
 * ====================================
 * Use this file to configure the backend API endpoint when deploying the frontend
 * independently (e.g. on GitHub Pages, Netlify, Vercel, or Cloudflare Pages).
 *
 * HOW IT WORKS:
 * 1. If you leave API_BASE_URL as "", CardioGuard automatically detects the URL:
 *    - If running over HTTP/HTTPS (e.g. deployed on Vercel, Render, or localhost:8000),
 *      it uses relative paths ("/predict", "/health") which work seamlessly with same-origin APIs.
 *    - If opened locally via file://, it defaults to "http://127.0.0.1:8000".
 * 2. If you deploy your backend to an external cloud URL, you can set API_BASE_URL here.
 * 3. You can also change the API URL directly in the UI at runtime using the
 *    "⚙️ API Settings" button without modifying any code!
 */

window.CARDIOGUARD_CONFIG = {
    // Set to your backend URL (e.g., "http://127.0.0.1:8000" or "https://api.yourdomain.com").
    // Leave empty "" for automatic same-origin detection.
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
     * 3. Same-origin relative path if running online via HTTP/HTTPS
     * 4. Fallback to http://127.0.0.1:8000 for local file:// testing
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

        // 3. Auto-detection: if hosted online on HTTP/HTTPS or local dev server
        if (window.location.protocol.startsWith("http")) {
            return endpoint;
        }

        // 4. Fallback for static file:// testing
        return "http://127.0.0.1:8000" + endpoint;
    }
};
