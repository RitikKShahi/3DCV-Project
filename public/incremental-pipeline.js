// Incremental 3D Reconstruction Pipeline for WebGPU 3DGS Engine
// Offloads heavy ZIP reconstruction to local server endpoint so browser never freezes.

(function() {
  if (typeof window === "undefined") return;

  // Global cache of latest trained model splats
  window.lastTrainedModelPly = null;

  // Intercept anchor clicks when exporting a 3D model
  const originalAnchorClick = HTMLAnchorElement.prototype.click;
  HTMLAnchorElement.prototype.click = function() {
    if (this.download && this.download.endsWith(".ply") && this.href) {
      fetch(this.href)
        .then(res => res.blob())
        .then(blob => {
          window.lastTrainedModelPly = blob;
          showToast("3D Model Checkpoint Saved! Ready for incremental photos.");
          console.log("[IncrementalPipeline] Automatically cached latest trained model checkpoint:", blob.size, "bytes");
        })
        .catch(err => console.warn("[IncrementalPipeline] Failed to cache PLY export:", err));
    }
    return originalAnchorClick.call(this);
  };

  /**
   * Floating toast notification
   */
  function showToast(message, isError = false, persistent = false) {
    let toast = document.getElementById("incremental-toast");
    if (!toast) {
      toast = document.createElement("div");
      toast.id = "incremental-toast";
      toast.style.position = "fixed";
      toast.style.bottom = "24px";
      toast.style.left = "50%";
      toast.style.transform = "translateX(-50%)";
      toast.style.padding = "10px 22px";
      toast.style.borderRadius = "8px";
      toast.style.fontSize = "13px";
      toast.style.fontFamily = "system-ui, -apple-system, sans-serif";
      toast.style.fontWeight = "600";
      toast.style.zIndex = "999999";
      toast.style.transition = "all 0.25s ease";
      toast.style.boxShadow = "0 10px 30px rgba(0,0,0,0.6)";
      document.body.appendChild(toast);
    }

    toast.style.background = isError ? "#ef4444" : "#0284c7";
    toast.style.color = "#ffffff";
    toast.textContent = message;
    toast.style.opacity = "1";
    toast.style.display = "block";

    if (toast._timer) clearTimeout(toast._timer);
    if (!persistent) {
      toast._timer = setTimeout(() => {
        toast.style.opacity = "0";
        setTimeout(() => { toast.style.display = "none"; }, 300);
      }, 4000);
    }
  }

  /**
   * Public API: Load an updated dataset with additional photos without starting from zero.
   */
  window.loadIncrementalDataset = async function(newDatasetZipFile, customBasePlyBlob) {
    showToast("Processing new photos with trained 3D model...", false, true);

    try {
      const formData = new FormData();
      formData.append("zip", newDatasetZipFile);

      const basePly = customBasePlyBlob || window.lastTrainedModelPly;
      if (basePly) {
        formData.append("ply", basePly);
      }

      const res = await fetch("/api/incremental", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({ error: "Server error" }));
        throw new Error(errorData.error || ("Server returned status " + res.status));
      }

      const data = await res.json();
      if (data.success && data.url) {
        if (window.brushApp) {
          window.brushApp.load_url(data.url);
          showToast("Success! Reconstruction resumed with new photos from current 3D model.");
        }
      } else {
        throw new Error(data.error || "Failed to generate warm dataset");
      }
    } catch (err) {
      showToast("Error: " + err.message, true);
      console.error("[IncrementalPipeline] Error:", err);
    }
  };

  /**
   * Sphere Toggle UI attached cleanly to screen edge
   */
  function setupSphereUI() {
    const STYLES = `
      #inc-sphere-btn {
        position: fixed;
        left: 20px;
        bottom: 24px;
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 35%, #38bdf8 0%, #0369a1 70%, #082f49 100%);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5), 0 0 16px rgba(56, 189, 248, 0.45);
        border: 2px solid rgba(255, 255, 255, 0.25);
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 99990;
        transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.3s;
        outline: none;
      }

      #inc-sphere-btn:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.6), 0 0 24px rgba(56, 189, 248, 0.7);
      }

      #inc-sphere-btn.hidden {
        opacity: 0;
        transform: scale(0.4);
        pointer-events: none;
      }

      #inc-modal-card {
        position: fixed;
        left: 20px;
        bottom: 24px;
        width: 320px;
        max-width: 90vw;
        background: rgba(15, 23, 42, 0.92);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 16px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.7), 0 0 1px rgba(255, 255, 255, 0.15) inset;
        padding: 18px 20px;
        z-index: 99995;
        font-family: system-ui, -apple-system, sans-serif;
        color: #f1f5f9;
        transform-origin: bottom left;
        transition: opacity 0.25s cubic-bezier(0.16, 1, 0.3, 1), transform 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        box-sizing: border-box;
      }

      #inc-modal-card.hidden {
        opacity: 0;
        transform: scale(0.6);
        pointer-events: none;
        visibility: hidden;
      }

      .inc-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
      }

      .inc-card-title {
        font-size: 14px;
        font-weight: 700;
        color: #38bdf8;
        display: flex;
        align-items: center;
        gap: 6px;
      }

      .inc-close-btn {
        background: transparent;
        border: none;
        color: #94a3b8;
        font-size: 18px;
        line-height: 1;
        cursor: pointer;
        padding: 2px 6px;
        border-radius: 4px;
        transition: color 0.15s;
      }

      .inc-close-btn:hover {
        color: #ffffff;
      }

      .inc-card-desc {
        font-size: 12px;
        color: #94a3b8;
        line-height: 1.4;
        margin-bottom: 14px;
      }

      .inc-upload-zone {
        border: 2px dashed rgba(56, 189, 248, 0.35);
        border-radius: 10px;
        padding: 16px 12px;
        text-align: center;
        background: rgba(30, 41, 59, 0.4);
        cursor: pointer;
        transition: all 0.2s;
      }

      .inc-upload-zone:hover {
        border-color: #38bdf8;
        background: rgba(56, 189, 248, 0.1);
      }

      .inc-upload-text {
        font-size: 12px;
        font-weight: 600;
        color: #e2e8f0;
        margin-top: 6px;
      }

      .inc-upload-subtext {
        font-size: 10px;
        color: #64748b;
        margin-top: 2px;
      }
    `;

    const styleTag = document.createElement("style");
    styleTag.textContent = STYLES;
    document.head.appendChild(styleTag);

    // Hidden file input
    const hiddenFileInput = document.createElement("input");
    hiddenFileInput.type = "file";
    hiddenFileInput.accept = ".zip";
    hiddenFileInput.style.display = "none";
    document.body.appendChild(hiddenFileInput);

    // The Sphere Button
    const sphereBtn = document.createElement("button");
    sphereBtn.id = "inc-sphere-btn";
    sphereBtn.title = "Add Photos (.zip)";
    sphereBtn.innerHTML = `
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
        <polyline points="17 8 12 3 7 8"></polyline>
        <line x1="12" y1="3" x2="12" y2="15"></line>
      </svg>
    `;
    document.body.appendChild(sphereBtn);

    // The Popup Modal Card
    const modalCard = document.createElement("div");
    modalCard.id = "inc-modal-card";
    modalCard.classList.add("hidden");
    modalCard.innerHTML = `
      <div class="inc-card-header">
        <div class="inc-card-title">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <line x1="12" y1="5" x2="12" y2="19"></line>
            <line x1="5" y1="12" x2="19" y2="12"></line>
          </svg>
          Add Photos (.zip)
        </div>
        <button class="inc-close-btn" id="inc-close-card" title="Minimize to sphere">&times;</button>
      </div>
      <div class="inc-card-desc">
        Select an updated dataset archive to incorporate new photo angles without restarting from step 0.
      </div>
      <div class="inc-upload-zone" id="inc-drop-target">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="17 8 12 3 7 8"></polyline>
          <line x1="12" y1="3" x2="12" y2="15"></line>
        </svg>
        <div class="inc-upload-text">Choose .ZIP Dataset or Drop Here</div>
        <div class="inc-upload-subtext">Retains your current 3D reconstruction</div>
      </div>
    `;
    document.body.appendChild(modalCard);

    function openCard() {
      sphereBtn.classList.add("hidden");
      modalCard.classList.remove("hidden");
    }

    function closeCard() {
      modalCard.classList.add("hidden");
      sphereBtn.classList.remove("hidden");
    }

    sphereBtn.addEventListener("click", openCard);
    document.getElementById("inc-close-card")?.addEventListener("click", closeCard);

    const dropTarget = document.getElementById("inc-drop-target");
    dropTarget?.addEventListener("click", () => {
      hiddenFileInput.click();
    });

    hiddenFileInput.addEventListener("change", (e) => {
      const file = e.target.files?.[0];
      if (file) {
        closeCard();
        window.loadIncrementalDataset(file);
      }
    });

    // Window drag-and-drop
    window.addEventListener("dragover", (e) => e.preventDefault());
    window.addEventListener("drop", (e) => {
      e.preventDefault();
      const files = e.dataTransfer?.files;
      if (files && files.length > 0 && files[0].name.toLowerCase().endsWith(".zip")) {
        closeCard();
        window.loadIncrementalDataset(files[0]);
      }
    });

    // Close on Escape key
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !modalCard.classList.contains("hidden")) {
        closeCard();
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setupSphereUI);
  } else {
    setupSphereUI();
  }
})();
