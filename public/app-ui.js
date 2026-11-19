// Modern 3D Reconstruction Landing Page
// Provides dedicated ingestion options: ZIP Dataset, Video, and Raw Photos
// Includes sleek WebGPU loading transition so the user never sees the old egui screen.

(function() {
  if (typeof window === "undefined") return;

  // Global interceptor for WASM fetch & Request to support blob URLs
  const OriginalRequest = window.Request;
  window.Request = class extends OriginalRequest {
    constructor(input, init) {
      if (typeof input === "string" && /^https?:\/\/blob:/.test(input)) {
        input = input.replace(/^https?:\/\/blob:/, "blob:");
      }
      super(input, init);
    }
  };

  const originalFetch = window.fetch;
  window.fetch = function(input, init) {
    if (typeof input === "string" && /^https?:\/\/blob:/.test(input)) {
      input = input.replace(/^https?:\/\/blob:/, "blob:");
    } else if (input instanceof Request && /^https?:\/\/blob:/.test(input.url)) {
      const cleanUrl = input.url.replace(/^https?:\/\/blob:/, "blob:");
      input = new OriginalRequest(cleanUrl, input);
    }
    return originalFetch.call(this, input, init);
  };

  const STYLES = `
    @import url("https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap");

    #reconstruction-landing {
      position: fixed;
      inset: 0;
      z-index: 9990;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      background: rgba(10, 12, 16, 0.94);
      backdrop-filter: blur(32px);
      -webkit-backdrop-filter: blur(32px);
      font-family: "Plus Jakarta Sans", system-ui, -apple-system, sans-serif;
      color: #f8fafc;
      padding: 24px;
      box-sizing: border-box;
      transition: opacity 0.5s cubic-bezier(0.16, 1, 0.3, 1), visibility 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    }

    #reconstruction-landing.hidden {
      opacity: 0;
      visibility: hidden;
      pointer-events: none;
    }

    .landing-content {
      width: 100%;
      max-width: 820px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 24px;
      transition: opacity 0.3s ease, transform 0.3s ease;
    }

    .landing-content.hidden-fade {
      opacity: 0;
      transform: scale(0.96);
      pointer-events: none;
      display: none;
    }

    /* Header */
    .landing-header {
      text-align: center;
    }

    .landing-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 12px;
      background: rgba(56, 189, 248, 0.1);
      border: 1px solid rgba(56, 189, 248, 0.25);
      border-radius: 20px;
      font-size: 11px;
      font-weight: 600;
      color: #38bdf8;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      margin-bottom: 12px;
    }

    .landing-title {
      font-size: 32px;
      font-weight: 800;
      letter-spacing: -0.03em;
      margin: 0 0 8px 0;
      background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .landing-subtitle {
      font-size: 14px;
      color: #94a3b8;
      margin: 0;
      font-weight: 400;
      max-width: 480px;
      line-height: 1.5;
    }

    /* 3 Main Ingestion Options Cards */
    .ingestion-cards-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
      width: 100%;
    }

    @media (max-width: 720px) {
      .ingestion-cards-grid {
        grid-template-columns: 1fr;
      }
    }

    .ingestion-card {
      background: rgba(22, 27, 38, 0.75);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 16px;
      padding: 24px 20px;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      text-align: left;
      transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
      position: relative;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }

    .ingestion-card:hover {
      transform: translateY(-4px);
      border-color: rgba(56, 189, 248, 0.4);
      background: rgba(30, 41, 59, 0.85);
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.6), 0 0 20px rgba(56, 189, 248, 0.15);
    }

    .card-icon-wrapper {
      width: 44px;
      height: 44px;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 16px;
      transition: transform 0.25s ease;
    }

    .ingestion-card:hover .card-icon-wrapper {
      transform: scale(1.08);
    }

    /* Card 1: ZIP */
    .card-zip .card-icon-wrapper {
      background: rgba(56, 189, 248, 0.15);
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.3);
    }

    /* Card 2: Video */
    .card-video .card-icon-wrapper {
      background: rgba(168, 85, 247, 0.15);
      color: #c084fc;
      border: 1px solid rgba(168, 85, 247, 0.3);
    }

    /* Card 3: Photos */
    .card-photos .card-icon-wrapper {
      background: rgba(52, 211, 153, 0.15);
      color: #34d399;
      border: 1px solid rgba(52, 211, 153, 0.3);
    }

    .card-title {
      font-size: 16px;
      font-weight: 700;
      color: #ffffff;
      margin: 0 0 6px 0;
      letter-spacing: -0.01em;
    }

    .card-desc {
      font-size: 12px;
      color: #94a3b8;
      line-height: 1.45;
      margin: 0 0 16px 0;
      flex-grow: 1;
    }

    .card-action-tag {
      font-size: 11px;
      font-weight: 600;
      color: #cbd5e1;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      transition: gap 0.2s;
    }

    .ingestion-card:hover .card-action-tag {
      color: #38bdf8;
      gap: 7px;
    }

    /* Secondary actions bar (Directory, URL) */
    .secondary-row {
      display: flex;
      align-items: center;
      gap: 16px;
      font-size: 12px;
      color: #64748b;
    }

    .secondary-btn {
      background: transparent;
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: #94a3b8;
      padding: 6px 14px;
      border-radius: 8px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      font-family: inherit;
      transition: all 0.2s;
    }

    .secondary-btn:hover {
      background: rgba(255, 255, 255, 0.05);
      color: #ffffff;
      border-color: rgba(255, 255, 255, 0.25);
    }

    /* Controls Preview Card */
    .landing-controls {
      background: rgba(18, 22, 32, 0.5);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 12px;
      padding: 12px 20px;
      width: 100%;
      box-sizing: border-box;
      display: flex;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
    }

    .control-item-chip {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
    }

    .chip-key {
      color: #38bdf8;
      font-weight: 600;
      font-family: "JetBrains Mono", monospace;
    }

    .chip-action {
      color: #94a3b8;
    }

    /* Loading Transition Screen (Prevents flashing drab egui screen) */
    #landing-loading-screen {
      display: none;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      gap: 16px;
      animation: fadeIn 0.3s ease;
    }

    #landing-loading-screen.active {
      display: flex;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: scale(0.95); }
      to { opacity: 1; transform: scale(1); }
    }

    .spinner-ring {
      width: 56px;
      height: 56px;
      border-radius: 50%;
      border: 3px solid rgba(56, 189, 248, 0.15);
      border-top-color: #38bdf8;
      animation: spin 0.8s cubic-bezier(0.5, 0.1, 0.4, 0.9) infinite;
      box-shadow: 0 0 20px rgba(56, 189, 248, 0.3);
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .loading-title {
      font-size: 20px;
      font-weight: 700;
      color: #ffffff;
      margin: 0;
      letter-spacing: -0.01em;
    }

    .loading-status-text {
      font-size: 13px;
      color: #94a3b8;
      margin: 0;
      font-family: "JetBrains Mono", monospace;
    }

    .loading-progress-bar-container {
      width: 240px;
      height: 4px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 4px;
      overflow: hidden;
      margin-top: 4px;
    }

    .loading-progress-bar {
      width: 0%;
      height: 100%;
      background: linear-gradient(90deg, #38bdf8, #818cf8);
      border-radius: 4px;
      transition: width 6s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .loading-progress-bar.animate {
      width: 100%;
    }

    /* COLMAP Data Preparation Modal */
    #colmap-modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(4, 7, 13, 0.85);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      z-index: 100000;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 20px;
      animation: fadeIn 0.2s ease;
    }

    #colmap-modal-backdrop.open {
      display: flex;
    }

    #colmap-modal {
      width: 660px;
      max-width: 95vw;
      max-height: 90vh;
      background: rgba(15, 23, 42, 0.96);
      border: 1px solid rgba(56, 189, 248, 0.3);
      border-radius: 20px;
      padding: 28px;
      box-shadow: 0 25px 60px -15px rgba(0, 0, 0, 0.8), 0 0 40px rgba(56, 189, 248, 0.15);
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      color: #e2e8f0;
      box-sizing: border-box;
    }

    .colmap-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 18px;
    }

    .colmap-title-row {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .colmap-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(56, 189, 248, 0.12);
      border: 1px solid rgba(56, 189, 248, 0.3);
      color: #38bdf8;
      font-size: 11px;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 9999px;
      width: fit-content;
    }

    .colmap-title {
      font-size: 20px;
      font-weight: 700;
      color: #ffffff;
      margin: 0;
      letter-spacing: -0.01em;
    }

    .colmap-subtitle {
      font-size: 12px;
      color: #94a3b8;
      margin: 0;
      line-height: 1.45;
    }

    .colmap-close-btn {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: #94a3b8;
      width: 32px;
      height: 32px;
      border-radius: 50%;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 18px;
      transition: all 0.2s;
    }

    .colmap-close-btn:hover {
      background: rgba(255, 255, 255, 0.15);
      color: #ffffff;
    }

    .colmap-photos-summary {
      background: rgba(2, 6, 23, 0.6);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 12px;
      padding: 12px 16px;
      margin-bottom: 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .photos-count-tag {
      font-size: 13px;
      font-weight: 600;
      color: #38bdf8;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .photos-list-pill {
      font-size: 11px;
      color: #94a3b8;
      font-family: monospace;
    }

    .colmap-options-group {
      display: flex;
      flex-direction: column;
      gap: 16px;
      margin-bottom: 24px;
    }

    .colmap-field {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .colmap-label {
      font-size: 12px;
      font-weight: 600;
      color: #cbd5e1;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .colmap-label-hint {
      font-size: 11px;
      color: #64748b;
      font-weight: 400;
    }

    .colmap-pills-row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }

    .colmap-pill {
      background: rgba(30, 41, 59, 0.5);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 10px;
      padding: 10px 12px;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      gap: 3px;
      transition: all 0.2s;
    }

    .colmap-pill:hover {
      border-color: rgba(56, 189, 248, 0.4);
      background: rgba(30, 41, 59, 0.8);
    }

    .colmap-pill.selected {
      border-color: #38bdf8;
      background: rgba(56, 189, 248, 0.14);
      box-shadow: 0 0 12px rgba(56, 189, 248, 0.2);
    }

    .colmap-pill-name {
      font-size: 12px;
      font-weight: 600;
      color: #f8fafc;
    }

    .colmap-pill-desc {
      font-size: 10px;
      color: #94a3b8;
    }

    /* Live Pipeline Status & Terminal */
    .colmap-pipeline-view {
      display: none;
      flex-direction: column;
      gap: 14px;
    }

    .colmap-pipeline-view.active {
      display: flex;
    }

    .colmap-progress-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 13px;
      font-weight: 600;
      color: #38bdf8;
    }

    .colmap-progress-track {
      width: 100%;
      height: 6px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 6px;
      overflow: hidden;
    }

    .colmap-progress-bar {
      width: 0%;
      height: 100%;
      background: linear-gradient(90deg, #38bdf8, #818cf8);
      border-radius: 6px;
      transition: width 0.3s ease;
    }

    .colmap-terminal {
      background: #090d16;
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 10px;
      padding: 12px 14px;
      height: 180px;
      overflow-y: auto;
      font-family: "JetBrains Mono", monospace;
      font-size: 11px;
      color: #94a3b8;
      line-height: 1.5;
      display: flex;
      flex-direction: column;
      gap: 3px;
    }

    .colmap-terminal .term-line-info {
      color: #38bdf8;
    }

    .colmap-terminal .term-line-success {
      color: #34d399;
      font-weight: 600;
    }

    .colmap-terminal .term-line-error {
      color: #f87171;
    }

    .colmap-actions-row {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 12px;
      margin-top: 10px;
    }

    .colmap-btn-secondary {
      background: transparent;
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: #94a3b8;
      padding: 9px 18px;
      border-radius: 10px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }

    .colmap-btn-secondary:hover {
      background: rgba(255, 255, 255, 0.05);
      color: #ffffff;
    }

    .colmap-btn-primary {
      background: linear-gradient(135deg, #0284c7, #0369a1);
      border: 1px solid rgba(56, 189, 248, 0.5);
      color: #ffffff;
      padding: 9px 22px;
      border-radius: 10px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      box-shadow: 0 0 16px rgba(56, 189, 248, 0.3);
      transition: all 0.2s;
    }

    .colmap-btn-primary:hover:not(:disabled) {
      box-shadow: 0 0 24px rgba(56, 189, 248, 0.5);
      transform: translateY(-1px);
    }

    .colmap-btn-primary:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    /* Highlight badge for ZIP Archive card after COLMAP download */
    .card-zip.zip-highlight {
      border-color: #38bdf8 !important;
      box-shadow: 0 0 35px rgba(56, 189, 248, 0.5) !important;
      animation: pulseZip 1.5s infinite alternate ease-in-out;
    }

    @keyframes pulseZip {
      from { transform: scale(1); }
      to { transform: scale(1.03); }
    }

    .zip-hint-badge {
      display: none;
      position: absolute;
      top: -12px;
      right: 14px;
      background: #0284c7;
      color: #ffffff;
      font-size: 10px;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 9999px;
      box-shadow: 0 0 12px rgba(56, 189, 248, 0.6);
      animation: bounceBadge 1s infinite alternate ease-in-out;
    }

    @keyframes bounceBadge {
      from { transform: translateY(0); }
      to { transform: translateY(-3px); }
    }

    .card-zip.zip-highlight .zip-hint-badge {
      display: block;
    }
  `;

  function initLandingPage() {
    const styleTag = document.createElement("style");
    styleTag.textContent = STYLES;
    document.head.appendChild(styleTag);

    // Hidden inputs
    const zipInput = document.createElement("input");
    zipInput.type = "file";
    zipInput.accept = ".zip,.ply,.compressed.ply";
    zipInput.style.display = "none";
    document.body.appendChild(zipInput);

    const videoInput = document.createElement("input");
    videoInput.type = "file";
    videoInput.accept = "video/*,.mp4,.mov,.webm,.avi";
    videoInput.style.display = "none";
    document.body.appendChild(videoInput);

    const photosInput = document.createElement("input");
    photosInput.type = "file";
    photosInput.accept = "image/*,.jpg,.jpeg,.png";
    photosInput.multiple = true;
    photosInput.style.display = "none";
    document.body.appendChild(photosInput);

    const dirInput = document.createElement("input");
    dirInput.type = "file";
    dirInput.webkitdirectory = true;
    dirInput.directory = true;
    dirInput.multiple = true;
    dirInput.style.display = "none";
    document.body.appendChild(dirInput);

    // Container
    const landing = document.createElement("div");
    landing.id = "reconstruction-landing";
    landing.innerHTML = `
      <div class="landing-content" id="landing-main-content">
        <!-- Header -->
        <div class="landing-header">
          <div class="landing-badge">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
            WebGPU Reconstruction Studio
          </div>
          <h1 class="landing-title">3D Reconstruction</h1>
          <p class="landing-subtitle">Choose an input format to begin 3D Gaussian Splatting scene reconstruction or view a trained model.</p>
        </div>

        <!-- 3 Primary Options -->
        <div class="ingestion-cards-grid">
          <!-- Card 1: ZIP Dataset -->
          <div class="ingestion-card card-zip" id="btn-select-zip">
            <span class="zip-hint-badge" id="zip-hint-badge">Dataset Ready! Click to Load</span>
            <div class="card-icon-wrapper">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 8v13H3V8"></path>
                <path d="M1 3h22v5H1z"></path>
                <path d="M10 12h4"></path>
              </svg>
            </div>
            <div class="card-title">ZIP Archive</div>
            <div class="card-desc">COLMAP dataset, NeRF transforms.json folder, or pre-trained .PLY model.</div>
            <div class="card-action-tag">
              Upload Dataset
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"></polyline></svg>
            </div>
          </div>

          <!-- Card 2: Video -->
          <div class="ingestion-card card-video" id="btn-select-video">
            <div class="card-icon-wrapper">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="23 7 16 12 23 17 23 7"></polygon>
                <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
              </svg>
            </div>
            <div class="card-title">Video Recording</div>
            <div class="card-desc">Single moving video sequence (.MP4, .MOV) for keyframe trajectory extraction.</div>
            <div class="card-action-tag">
              Upload Video
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"></polyline></svg>
            </div>
          </div>

          <!-- Card 3: Raw Photos -->
          <div class="ingestion-card card-photos" id="btn-select-photos">
            <div class="card-icon-wrapper">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                <circle cx="8.5" cy="8.5" r="1.5"></circle>
                <polyline points="21 15 16 10 5 21"></polyline>
              </svg>
            </div>
            <div class="card-title">Raw Photos</div>
            <div class="card-desc">Batch of individual photos (.JPG, .PNG) taken from multiple viewpoints.</div>
            <div class="card-action-tag">
              Select Photos
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"></polyline></svg>
            </div>
          </div>
        </div>

        <!-- Secondary options row -->
        <div class="secondary-row">
          <span>Or load from:</span>
          <button class="secondary-btn" id="btn-select-dir">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>
            Local Folder
          </button>
          <button class="secondary-btn" id="btn-enter-url">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path></svg>
            Remote URL
          </button>
        </div>

        <!-- Controls Cheat Sheet -->
        <div class="landing-controls">
          <div class="control-item-chip">
            <span class="chip-key">Left Drag</span>
            <span class="chip-action">Orbit</span>
          </div>
          <div class="control-item-chip">
            <span class="chip-key">Right Drag</span>
            <span class="chip-action">Look</span>
          </div>
          <div class="control-item-chip">
            <span class="chip-key">Middle Drag</span>
            <span class="chip-action">Pan</span>
          </div>
          <div class="control-item-chip">
            <span class="chip-key">Scroll</span>
            <span class="chip-action">Zoom</span>
          </div>
          <div class="control-item-chip">
            <span class="chip-key">WASD/QE</span>
            <span class="chip-action">Fly</span>
          </div>
          <div class="control-item-chip">
            <span class="chip-key">Shift</span>
            <span class="chip-action">Fast</span>
          </div>
        </div>
      </div>

      <!-- Loading Transition Screen -->
      <div id="landing-loading-screen">
        <div class="spinner-ring"></div>
        <div class="loading-title">Launching 3D Scene</div>
        <div class="loading-status-text" id="loading-status">Unpacking dataset & configuring WebGPU... (6s)</div>
        <div class="loading-progress-bar-container">
          <div class="loading-progress-bar" id="loading-progress-fill"></div>
        </div>
      </div>

      <!-- COLMAP Data Preparation Modal -->
      <div id="colmap-modal-backdrop">
        <div id="colmap-modal">
          <div class="colmap-header">
            <div class="colmap-title-row">
              <div class="colmap-badge">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
                <span id="colmap-modal-badge-text">COLMAP Structure-from-Motion</span>
              </div>
              <h2 class="colmap-title" id="colmap-modal-title">Raw Photos Data Preparation</h2>
              <p class="colmap-subtitle" id="colmap-modal-subtitle">Calibrate camera poses and build sparse 3D point cloud for Gaussian Splatting.</p>
            </div>
            <button class="colmap-close-btn" id="colmap-close-btn" title="Close">&times;</button>
          </div>

          <!-- Configuration View -->
          <div id="colmap-config-view">
            <div class="colmap-photos-summary">
              <div class="photos-count-tag">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
                <span id="colmap-photo-count">0 Photos Selected</span>
              </div>
              <span class="photos-list-pill" id="colmap-photo-names">ready for processing</span>
            </div>

            <div class="colmap-options-group">
              <!-- Video Frame Rate (FPS) Option -->
              <div class="colmap-field" id="fps-field" style="display: none;">
                <div class="colmap-label">
                  <span>Frame Extraction Rate (FFmpeg)</span>
                  <span class="colmap-label-hint">Keyframe density along video sequence</span>
                </div>
                <div class="colmap-pills-row" id="fps-options">
                  <div class="colmap-pill selected" data-value="5.0">
                    <span class="colmap-pill-name">5 FPS (Recommended)</span>
                    <span class="colmap-pill-desc">~90 frames (18s video)</span>
                  </div>
                  <div class="colmap-pill" data-value="10.0">
                    <span class="colmap-pill-name">10 FPS (High Density)</span>
                    <span class="colmap-pill-desc">~180 frames (Fine details)</span>
                  </div>
                  <div class="colmap-pill" data-value="2.0">
                    <span class="colmap-pill-name">2 FPS (Fast)</span>
                    <span class="colmap-pill-desc">~36 frames (Quick test)</span>
                  </div>
                </div>
              </div>

              <!-- Scaling Option -->
              <div class="colmap-field">
                <div class="colmap-label">
                  <span>Image Scaling Option</span>
                  <span class="colmap-label-hint">Dynamic downscaling for training speed</span>
                </div>
                <div class="colmap-pills-row" id="scaling-options">
                  <div class="colmap-pill selected" data-value="1600k">
                    <span class="colmap-pill-name">1600px (Optimal)</span>
                    <span class="colmap-pill-desc">Best balance for 3DGS</span>
                  </div>
                  <div class="colmap-pill" data-value="Half">
                    <span class="colmap-pill-name">Half (50%)</span>
                    <span class="colmap-pill-desc">Faster processing</span>
                  </div>
                  <div class="colmap-pill" data-value="No Scaling">
                    <span class="colmap-pill-name">Original</span>
                    <span class="colmap-pill-desc">Full image resolution</span>
                  </div>
                </div>
              </div>

              <!-- Matching Strategy -->
              <div class="colmap-field" id="matching-field">
                <div class="colmap-label">
                  <span>COLMAP Feature Matching Type</span>
                  <span class="colmap-label-hint">SIFT feature correspondences</span>
                </div>
                <div class="colmap-pills-row" id="matching-options">
                  <div class="colmap-pill selected" data-value="Sequential">
                    <span class="colmap-pill-name">Sequential</span>
                    <span class="colmap-pill-desc">Ordered loop / video</span>
                  </div>
                  <div class="colmap-pill" data-value="Exhaustive">
                    <span class="colmap-pill-name">Exhaustive</span>
                    <span class="colmap-pill-desc">All pairwise matches</span>
                  </div>
                  <div class="colmap-pill" data-value="Spatial">
                    <span class="colmap-pill-name">Spatial</span>
                    <span class="colmap-pill-desc">Geotagged positions</span>
                  </div>
                </div>
              </div>
            </div>

            <div class="colmap-actions-row">
              <button class="colmap-btn-secondary" id="colmap-btn-cancel">Cancel</button>
              <button class="colmap-btn-primary" id="colmap-btn-run">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                <span id="colmap-btn-run-text">Run COLMAP Pipeline</span>
              </button>
            </div>
          </div>

          <!-- Live Pipeline View (shows during execution) -->
          <div class="colmap-pipeline-view" id="colmap-pipeline-view">
            <div class="colmap-progress-header">
              <span id="colmap-pipeline-step">Initializing COLMAP...</span>
              <span id="colmap-pipeline-pct">0%</span>
            </div>
            <div class="colmap-progress-track">
              <div class="colmap-progress-bar" id="colmap-progress-bar"></div>
            </div>
            <div class="colmap-terminal" id="colmap-terminal">
              <div class="term-line-info">[COLMAP Engine] Initializing workspace & camera model...</div>
            </div>
            <div class="colmap-actions-row" id="colmap-completion-row" style="display: none;">
              <span style="color: #34d399; font-size: 12px; font-weight: 600; margin-right: auto;" id="colmap-completion-msg">
                ✓ Dataset ZIP downloaded automatically!
              </span>
              <button class="colmap-btn-primary" id="colmap-btn-done">
                Return to Landing Page
              </button>
            </div>
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(landing);

    const mainContent = document.getElementById("landing-main-content");
    const loadingScreen = document.getElementById("landing-loading-screen");
    const loadingStatus = document.getElementById("loading-status");
    const progressFill = document.getElementById("loading-progress-fill");

    function showLoadingTransition(statusText = "Loading 3D Gaussian Splats...") {
      mainContent.classList.add("hidden-fade");
      loadingScreen.classList.add("active");
      if (loadingStatus) loadingStatus.textContent = statusText;
      if (progressFill) {
        progressFill.classList.remove("animate");
        void progressFill.offsetWidth; // trigger reflow
        progressFill.classList.add("animate");
      }
    }

    function hideLanding() {
      landing.classList.add("hidden");
      // Reset back after fade animation finishes
      setTimeout(() => {
        mainContent.classList.remove("hidden-fade");
        loadingScreen.classList.remove("active");
        if (progressFill) progressFill.classList.remove("animate");
      }, 600);
    }

    function showLanding() {
      mainContent.classList.remove("hidden-fade");
      loadingScreen.classList.remove("active");
      landing.classList.remove("hidden");
    }

    function loadInto3DGSEngine(url, isZip = false) {
      showLoadingTransition(isZip ? "Unpacking dataset & configuring WebGPU... (6s)" : "Loading 3D Gaussian splats... (6s)");

      // Countdown updater over the 6 seconds
      let remaining = 6;
      const countdownInterval = setInterval(() => {
        remaining--;
        if (remaining > 0 && loadingStatus) {
          loadingStatus.textContent = isZip 
            ? `Allocating WebGPU compute buffers... (${remaining}s)` 
            : `Rendering Gaussian splat fields... (${remaining}s)`;
        } else {
          clearInterval(countdownInterval);
        }
      }, 1000);

      const executeLoad = () => {
        if (window.brushApp) {
          window.brushApp.load_url(url);
          // Exactly 6.0 second timer requested by the user
          setTimeout(() => {
            clearInterval(countdownInterval);
            hideLanding();
          }, 6000);
        }
      };

      if (window.brushApp) {
        executeLoad();
      } else {
        const interval = setInterval(() => {
          if (window.brushApp) {
            clearInterval(interval);
            executeLoad();
          }
        }, 80);
      }
    }

    // Handlers
    document.getElementById("btn-select-zip")?.addEventListener("click", () => {
      document.getElementById("btn-select-zip")?.classList.remove("zip-highlight");
      zipInput.click();
    });

    zipInput.addEventListener("change", (e) => {
      const file = e.target.files?.[0];
      if (file) {
        const isZip = file.name.toLowerCase().endsWith(".zip");
        loadInto3DGSEngine(URL.createObjectURL(file), isZip);
      }
    });

    document.getElementById("btn-select-video")?.addEventListener("click", () => {
      videoInput.click();
    });

    videoInput.addEventListener("change", (e) => {
      const file = e.target.files?.[0];
      if (file) {
        console.log("[Reconstruction] Video selected -> Opening Video Data Preparation:", file.name);
        openModalForVideo(file);
      }
    });

    // COLMAP Data Preparation Controller & Video Frame Extraction
    let currentMode = "photos";
    let pendingPhotos = [];
    let pendingVideo = null;
    let selectedScaling = "1600k";
    let selectedMatching = "Sequential";
    let selectedFps = "5.0";

    const colmapModal = document.getElementById("colmap-modal-backdrop");
    const colmapCloseBtn = document.getElementById("colmap-close-btn");
    const colmapCancelBtn = document.getElementById("colmap-btn-cancel");
    const colmapRunBtn = document.getElementById("colmap-btn-run");
    const colmapDoneBtn = document.getElementById("colmap-btn-done");
    const colmapConfigView = document.getElementById("colmap-config-view");
    const colmapPipelineView = document.getElementById("colmap-pipeline-view");
    const colmapProgressBar = document.getElementById("colmap-progress-bar");
    const colmapPipelineStep = document.getElementById("colmap-pipeline-step");
    const colmapPipelinePct = document.getElementById("colmap-pipeline-pct");
    const colmapTerminal = document.getElementById("colmap-terminal");
    const colmapCompletionRow = document.getElementById("colmap-completion-row");
    const colmapCompletionMsg = document.getElementById("colmap-completion-msg");
    const zipCard = document.getElementById("btn-select-zip");

    const badgeText = document.getElementById("colmap-modal-badge-text");
    const modalTitle = document.getElementById("colmap-modal-title");
    const modalSubtitle = document.getElementById("colmap-modal-subtitle");
    const fpsField = document.getElementById("fps-field");
    const matchingField = document.getElementById("matching-field");
    const runBtnText = document.getElementById("colmap-btn-run-text");

    function openModalForPhotos(files) {
      currentMode = "photos";
      pendingPhotos = Array.from(files);
      pendingVideo = null;

      if (badgeText) badgeText.textContent = "COLMAP Structure-from-Motion";
      if (modalTitle) modalTitle.textContent = "Raw Photos Data Preparation";
      if (modalSubtitle) modalSubtitle.textContent = "Calibrate camera poses and build sparse 3D point cloud for Gaussian Splatting.";
      if (fpsField) fpsField.style.display = "none";
      if (matchingField) matchingField.style.display = "flex";
      if (runBtnText) runBtnText.textContent = "Run COLMAP Pipeline";

      const photoCountEl = document.getElementById("colmap-photo-count");
      const photoNamesEl = document.getElementById("colmap-photo-names");
      if (photoCountEl) photoCountEl.textContent = `${pendingPhotos.length} Photos Selected`;
      if (photoNamesEl) {
        const preview = pendingPhotos.slice(0, 3).map(f => f.name).join(", ");
        photoNamesEl.textContent = preview + (pendingPhotos.length > 3 ? ` (+${pendingPhotos.length - 3} more)` : "");
      }

      showConfigModal();
    }

    function openModalForVideo(file) {
      currentMode = "video";
      pendingVideo = file;
      pendingPhotos = [];

      if (badgeText) badgeText.textContent = "Video Keyframe Trajectory Extraction";
      if (modalTitle) modalTitle.textContent = "Video Recording Data Preparation";
      if (modalSubtitle) modalSubtitle.textContent = "Extract sequential frames with FFmpeg and build calibrated COLMAP dataset for Gaussian Splatting.";
      if (fpsField) fpsField.style.display = "flex";
      if (matchingField) matchingField.style.display = "none";
      if (runBtnText) runBtnText.textContent = "Extract Frames & Run COLMAP Pipeline";

      const photoCountEl = document.getElementById("colmap-photo-count");
      const photoNamesEl = document.getElementById("colmap-photo-names");
      const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
      if (photoCountEl) photoCountEl.textContent = `${file.name} (${sizeMb} MB)`;
      if (photoNamesEl) photoNamesEl.textContent = `Type: ${file.type || 'video'} | Ready for FFmpeg extraction`;

      showConfigModal();
    }

    function showConfigModal() {
      if (colmapConfigView) colmapConfigView.style.display = "block";
      if (colmapPipelineView) colmapPipelineView.classList.remove("active");
      if (colmapCompletionRow) colmapCompletionRow.style.display = "none";
      if (colmapRunBtn) colmapRunBtn.disabled = false;
      if (colmapModal) colmapModal.classList.add("open");
    }

    function closeColmapModal() {
      if (colmapModal) colmapModal.classList.remove("open");
    }

    colmapCloseBtn?.addEventListener("click", closeColmapModal);
    colmapCancelBtn?.addEventListener("click", closeColmapModal);

    // FPS pill selection
    document.querySelectorAll("#fps-options .colmap-pill").forEach(pill => {
      pill.addEventListener("click", () => {
        document.querySelectorAll("#fps-options .colmap-pill").forEach(p => p.classList.remove("selected"));
        pill.classList.add("selected");
        selectedFps = pill.getAttribute("data-value") || "5.0";
      });
    });

    // Scaling pill selection
    document.querySelectorAll("#scaling-options .colmap-pill").forEach(pill => {
      pill.addEventListener("click", () => {
        document.querySelectorAll("#scaling-options .colmap-pill").forEach(p => p.classList.remove("selected"));
        pill.classList.add("selected");
        selectedScaling = pill.getAttribute("data-value") || "1600k";
      });
    });

    // Matching pill selection
    document.querySelectorAll("#matching-options .colmap-pill").forEach(pill => {
      pill.addEventListener("click", () => {
        document.querySelectorAll("#matching-options .colmap-pill").forEach(p => p.classList.remove("selected"));
        pill.classList.add("selected");
        selectedMatching = pill.getAttribute("data-value") || "Sequential";
      });
    });

    function addTerminalLine(text, type = "normal") {
      if (!colmapTerminal) return;
      const line = document.createElement("div");
      if (type === "info") line.className = "term-line-info";
      else if (type === "success") line.className = "term-line-success";
      else if (type === "error") line.className = "term-line-error";
      line.textContent = text;
      colmapTerminal.appendChild(line);
      colmapTerminal.scrollTop = colmapTerminal.scrollHeight;
    }

    async function executeColmapPipeline() {
      const formData = new FormData();

      if (colmapConfigView) colmapConfigView.style.display = "none";
      if (colmapPipelineView) colmapPipelineView.classList.add("active");
      if (colmapTerminal) colmapTerminal.innerHTML = "";
      if (colmapProgressBar) colmapProgressBar.style.width = "4%";
      if (colmapPipelineStep) colmapPipelineStep.textContent = "Initializing pipeline...";
      if (colmapPipelinePct) colmapPipelinePct.textContent = "4%";

      if (currentMode === "video") {
        if (!pendingVideo) return;
        formData.append("video", pendingVideo);
        formData.append("fps", selectedFps);
        formData.append("scaling", selectedScaling);
        formData.append("matching", "Sequential");
        formData.append("camera_model", "SIMPLE_RADIAL");
        addTerminalLine(`[Pipeline] Initializing with video: ${pendingVideo.name}`, "info");
        addTerminalLine(`[Config] FFmpeg: ${selectedFps} FPS | Scaling: ${selectedScaling} | Matching: Sequential`, "info");
      } else {
        if (!pendingPhotos || pendingPhotos.length === 0) return;
        for (const file of pendingPhotos) {
          formData.append("photos", file);
        }
        formData.append("scaling", selectedScaling);
        formData.append("matching", selectedMatching);
        formData.append("camera_model", "SIMPLE_RADIAL");
        addTerminalLine(`[Pipeline] Initializing with ${pendingPhotos.length} images`, "info");
        addTerminalLine(`[Config] Scaling: ${selectedScaling} | Matching: ${selectedMatching} | Model: SIMPLE_RADIAL`, "info");
      }

      try {
        const response = await fetch("/api/colmap/process", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Server returned HTTP ${response.status}`);
        }

        const reader = response.body?.getReader();
        const decoder = new TextDecoder("utf-8");
        let streamBuffer = "";
        let finalDownloadUrl = null;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          streamBuffer += decoder.decode(value, { stream: true });
          const parts = streamBuffer.split("\n\n");
          streamBuffer = parts.pop() || "";

          for (const part of parts) {
            const raw = part.replace(/^data:\s*/, "").trim();
            if (!raw) continue;
            try {
              const data = JSON.parse(raw);
              if (data.type === "log" && data.message) {
                addTerminalLine(data.message);
              } else if (data.type === "progress") {
                if (colmapProgressBar) colmapProgressBar.style.width = `${data.percent}%`;
                if (colmapPipelinePct) colmapPipelinePct.textContent = `${data.percent}%`;
                if (colmapPipelineStep && data.message) colmapPipelineStep.textContent = data.message;
              } else if (data.type === "done" || data.type === "complete") {
                if (data.downloadUrl) {
                  finalDownloadUrl = data.downloadUrl;
                }
                if (colmapProgressBar) colmapProgressBar.style.width = "100%";
                if (colmapPipelinePct) colmapPipelinePct.textContent = "100%";
                if (colmapPipelineStep) colmapPipelineStep.textContent = "Reconstruction complete!";
                addTerminalLine("✓ COLMAP reconstruction completed successfully!", "success");
                if (finalDownloadUrl) {
                  addTerminalLine(`✓ Generated package: ${finalDownloadUrl}`, "success");
                }
              } else if (data.type === "error") {
                addTerminalLine(`[Error] ${data.message}`, "error");
              }
            } catch (err) {
              addTerminalLine(raw);
            }
          }
        }

        if (finalDownloadUrl) {
          // 1. Trigger browser automatic download of generated ZIP
          const a = document.createElement("a");
          a.href = finalDownloadUrl;
          a.download = finalDownloadUrl.split("/").pop() || "colmap_dataset.zip";
          document.body.appendChild(a);
          a.click();
          a.remove();

          if (colmapCompletionRow) colmapCompletionRow.style.display = "flex";
          if (colmapCompletionMsg) {
            colmapCompletionMsg.textContent = "✓ Dataset ZIP downloaded! Redirecting to landing page in 3s...";
          }

          // 2. Redirect to landing page after 3 seconds as requested
          setTimeout(() => {
            closeColmapModal();
            showLanding();
            // Highlight the ZIP Archive button so the user uses it to load the new ZIP
            if (zipCard) {
              zipCard.classList.add("zip-highlight");
              zipCard.scrollIntoView({ behavior: "smooth", block: "center" });
            }
          }, 3000);
        } else {
          addTerminalLine("Notice: Pipeline finished. You may review the logs above or return to the landing page.", "info");
          if (colmapCompletionRow) colmapCompletionRow.style.display = "flex";
        }
      } catch (err) {
        console.error("[ColmapPipeline] Error:", err);
        addTerminalLine(`[Pipeline Failure] ${err.message || String(err)}`, "error");
        if (colmapCompletionRow) colmapCompletionRow.style.display = "flex";
        if (colmapCompletionMsg) colmapCompletionMsg.textContent = "Error during pipeline execution.";
      }
    }

    colmapRunBtn?.addEventListener("click", executeColmapPipeline);
    colmapDoneBtn?.addEventListener("click", () => {
      closeColmapModal();
      showLanding();
      if (zipCard) zipCard.classList.add("zip-highlight");
    });

    document.getElementById("btn-select-photos")?.addEventListener("click", () => {
      photosInput.click();
    });

    photosInput.addEventListener("change", (e) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        console.log(`[Reconstruction] ${files.length} photos selected -> Opening COLMAP Data Preparation`);
        openModalForPhotos(files);
      }
    });

    document.getElementById("btn-select-dir")?.addEventListener("click", () => {
      dirInput.click();
    });

    dirInput.addEventListener("change", (e) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        for (let i = 0; i < files.length; i++) {
          const l = files[i].name.toLowerCase();
          if (l.endsWith(".ply") || l.endsWith(".zip")) {
            loadInto3DGSEngine(URL.createObjectURL(files[i]), l.endsWith(".zip"));
            return;
          }
        }
        hideLanding();
      }
    });

    document.getElementById("btn-enter-url")?.addEventListener("click", () => {
      const url = prompt("Enter model (.ply) or dataset (.zip) URL:", "http://127.0.0.1:3000/sample_splat.ply");
      if (url) {
        loadInto3DGSEngine(url, url.toLowerCase().endsWith(".zip"));
      }
    });

    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        if (landing.classList.contains("hidden")) {
          showLanding();
        } else {
          hideLanding();
        }
      }
    });

    window.hideLandingPage = hideLanding;
    window.showLandingPage = showLanding;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initLandingPage);
  } else {
    initLandingPage();
  }
})();
