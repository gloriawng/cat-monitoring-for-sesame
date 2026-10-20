class CatMonitor extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this.pollTimer = null;
    this.previewUrl = null;
  }

  connectedCallback() {
    this.shadowRoot.innerHTML = `
      <link rel="stylesheet" href="/static/cat-monitor.css">
      <main class="page">
        <header class="topbar">
          <a class="brand" href="/" aria-label="Sesame Watch home">
            <span class="brand-mark" aria-hidden="true">s.</span>
            <span>sesame watch</span>
          </a>
          <span class="local-note"><span class="tiny-dot"></span> private, local dashboard</span>
        </header>

        <section class="intro">
          <div class="eyebrow"><span>AT-HOME CHECK-IN</span><span class="line"></span></div>
          <h1>A little closer<br>to <em>Sesame.</em></h1>
          <p>Upload a photo for a quick check, or let Blink clips keep an eye on the bowl.</p>
        </section>

        <section class="grid">
          <article class="card upload-card">
            <div class="card-head">
              <div>
                <span class="step">01 / PHOTO CHECK</span>
                <h2>Is there a cat?</h2>
              </div>
              <span class="round-icon" aria-hidden="true">↗</span>
            </div>
            <label class="dropzone" for="photo">
              <input id="photo" type="file" accept="image/*">
              <span class="drop-art" aria-hidden="true">
                <svg viewBox="0 0 48 48" fill="none"><rect x="6" y="9" width="36" height="30" rx="5"></rect><circle cx="18" cy="20" r="4"></circle><path d="m9 34 10-9 7 6 5-4 8 7"></path></svg>
              </span>
              <span class="drop-title">Drop a photo here</span>
              <span class="drop-subtitle">or <span class="browse">browse files</span> · JPG, PNG, WEBP</span>
              <img class="preview" alt="Selected photo preview" hidden>
            </label>
            <div class="upload-actions">
              <span class="file-name">No photo selected</span>
              <button class="button button-dark" id="analyze" disabled>Check photo <span aria-hidden="true">→</span></button>
            </div>
            <p class="notice" role="status" aria-live="polite"></p>
            <div class="result" hidden aria-live="polite"></div>
            <p class="fine-print">This model detects cat presence; it cannot tell Sesame apart from another cat.</p>
          </article>

          <article class="card blink-card">
            <div class="card-head">
              <div>
                <span class="step">02 / CAMERA SYNC</span>
                <h2>Blink, in the loop.</h2>
              </div>
              <span class="camera-icon" aria-hidden="true">◉</span>
            </div>
            <p class="body-copy">Check for new motion clips and scan a few frames for a cat. Sync runs locally against your saved Blink session.</p>
            <div class="connection">
              <span class="status-light" id="blink-light"></span>
              <span id="blink-state">Checking Blink setup…</span>
            </div>
            <div class="camera-controls">
              <button class="button button-light" id="connect">Connect Blink</button>
              <label class="select-wrap" for="camera">
                <span class="visually-hidden">Choose a Blink camera</span>
                <select id="camera" disabled><option value="">Choose a camera</option></select>
              </label>
              <button class="button button-dark" id="toggle-monitor" disabled>Start monitoring <span aria-hidden="true">→</span></button>
            </div>
            <p class="sync-meta" id="sync-meta">New clips are checked about once a minute.</p>
            <div class="events-heading"><span>RECENT ACTIVITY</span><span class="event-count" id="event-count">0</span></div>
            <div class="events" id="events"><p class="empty">No clips checked yet. Connect Blink to get started.</p></div>
            <p class="fine-print">BlinkPy polls for saved motion clips; it does not provide a live video stream. Keep blink_config.json private.</p>
          </article>
        </section>
        <footer>Made with care for Sesame <span aria-hidden="true">♥</span></footer>
      </main>`;

    this.photoInput = this.shadowRoot.querySelector("#photo");
    this.photoInput.addEventListener("change", () => this.selectPhoto(this.photoInput.files[0]));
    this.shadowRoot.querySelector(".dropzone").addEventListener("dragover", (event) => {
      event.preventDefault();
      event.currentTarget.classList.add("dragover");
    });
    this.shadowRoot.querySelector(".dropzone").addEventListener("dragleave", (event) => {
      event.currentTarget.classList.remove("dragover");
    });
    this.shadowRoot.querySelector(".dropzone").addEventListener("drop", (event) => {
      event.preventDefault();
      event.currentTarget.classList.remove("dragover");
      const [file] = event.dataTransfer.files;
      if (file) this.setPhoto(file);
    });
    this.shadowRoot.querySelector("#analyze").addEventListener("click", () => this.analyzePhoto());
    this.shadowRoot.querySelector("#connect").addEventListener("click", () => this.connectBlink());
    this.shadowRoot.querySelector("#toggle-monitor").addEventListener("click", () => this.toggleMonitoring());
    this.shadowRoot.querySelector("#camera").addEventListener("change", (event) => {
      const button = this.shadowRoot.querySelector("#toggle-monitor");
      if (button.dataset.running !== "true") button.disabled = !event.currentTarget.value;
    });
    this.refreshStatus();
    this.pollTimer = window.setInterval(() => this.refreshStatus(), 5000);
  }

  disconnectedCallback() {
    window.clearInterval(this.pollTimer);
    if (this.previewUrl) URL.revokeObjectURL(this.previewUrl);
  }

  async request(path, options = {}) {
    let response;
    try {
      response = await fetch(path, options);
    } catch {
      throw new Error(
        "Can't reach the Cat Monitor API. Start `python web_app.py` and open http://127.0.0.1:8000."
      );
    }

    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error(
        "This page isn't connected to the Cat Monitor API. Open http://127.0.0.1:8000 instead of using a static preview."
      );
    }
    if (!response.ok) throw new Error(data.error || "Something went wrong.");
    return data;
  }

  setPhoto(file) {
    if (!file.type.startsWith("image/")) {
      this.setNotice("Choose an image file to continue.", true);
      return;
    }
    if (file.size > 12 * 1024 * 1024) {
      this.setNotice("Images must be smaller than 12 MB.", true);
      return;
    }
    this.selectPhoto(file);
  }

  selectPhoto(file) {
    if (!file) return;
    this.photoFile = file;
    const preview = this.shadowRoot.querySelector(".preview");
    const zone = this.shadowRoot.querySelector(".dropzone");
    const button = this.shadowRoot.querySelector("#analyze");
    const name = this.shadowRoot.querySelector(".file-name");
    if (!file) return;
    if (this.previewUrl) URL.revokeObjectURL(this.previewUrl);
    this.previewUrl = URL.createObjectURL(file);
    preview.src = this.previewUrl;
    preview.hidden = false;
    zone.classList.add("has-preview");
    name.textContent = file.name;
    button.disabled = false;
    this.shadowRoot.querySelector(".result").hidden = true;
    this.setNotice("");
  }

  async analyzePhoto() {
    const file = this.photoFile;
    if (!file) return;
    const button = this.shadowRoot.querySelector("#analyze");
    button.disabled = true;
    button.textContent = "Checking…";
    this.setNotice("Looking for a cat in your photo…");
    try {
      const image = await this.readAsBase64(file);
      const result = await this.request("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image, filename: file.name }),
      });
      this.renderResult(result);
      this.setNotice("");
    } catch (error) {
      this.setNotice(error.message, true);
    } finally {
      button.disabled = false;
      button.innerHTML = 'Check photo <span aria-hidden="true">→</span>';
    }
  }

  readAsBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error("Could not read this photo."));
      reader.onload = () => resolve(String(reader.result).split(",", 2)[1]);
      reader.readAsDataURL(file);
    });
  }

  renderResult(result) {
    const target = this.shadowRoot.querySelector(".result");
    const detected = result.class === "cat-present";
    const percentage = Math.round(result.confidence * 100);
    target.innerHTML = `
      <span class="result-icon">${detected ? "✳" : "○"}</span>
      <span><strong>${detected ? "Cat detected" : "No cat detected"}</strong><small>${percentage}% confidence</small></span>
      <span class="result-tag">${detected ? "CAT PRESENT" : "NO CAT"}</span>`;
    target.hidden = false;
  }

  setNotice(message, isError = false) {
    const notice = this.shadowRoot.querySelector(".notice");
    notice.textContent = message;
    notice.classList.toggle("error", isError);
  }

  async connectBlink() {
    const button = this.shadowRoot.querySelector("#connect");
    const state = this.shadowRoot.querySelector("#blink-state");
    button.disabled = true;
    state.textContent = "Connecting to Blink…";
    try {
      const data = await this.request("/api/blink/connect", { method: "POST" });
      const select = this.shadowRoot.querySelector("#camera");
      select.replaceChildren(new Option("Choose a camera", ""));
      for (const camera of data.cameras) select.add(new Option(camera, camera));
      select.disabled = false;
      this.shadowRoot.querySelector("#toggle-monitor").disabled = true;
      state.textContent = data.cameras.length
        ? `${data.cameras.length} camera${data.cameras.length === 1 ? "" : "s"} ready`
        : "No Blink cameras found.";
      this.shadowRoot.querySelector("#blink-light").classList.toggle("online", data.cameras.length > 0);
    } catch (error) {
      state.textContent = error.message;
      this.shadowRoot.querySelector("#blink-light").classList.remove("online");
    } finally {
      button.disabled = false;
    }
  }

  async toggleMonitoring() {
    const button = this.shadowRoot.querySelector("#toggle-monitor");
    const enabled = button.dataset.running !== "true";
    const camera = this.shadowRoot.querySelector("#camera").value;
    button.disabled = true;
    try {
      await this.request("/api/blink/monitor", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled, camera }),
      });
      await this.refreshStatus();
    } catch (error) {
      this.shadowRoot.querySelector("#blink-state").textContent = error.message;
    } finally {
      button.disabled = false;
    }
  }

  async refreshStatus() {
    try {
      const status = await this.request("/api/status");
      const button = this.shadowRoot.querySelector("#toggle-monitor");
      const state = this.shadowRoot.querySelector("#blink-state");
      const light = this.shadowRoot.querySelector("#blink-light");
      button.dataset.running = String(status.running);
      button.textContent = status.running ? "Stop monitoring" : "Start monitoring →";
      button.disabled = !status.running && !this.shadowRoot.querySelector("#camera").value;
      light.classList.toggle("online", status.connected);
      if (status.last_error) {
        state.textContent = status.last_error;
      } else if (status.running) {
        state.textContent = status.connected ? `Monitoring ${status.camera}` : "Connecting to Blink…";
      } else if (status.connected) {
        state.textContent = "Connected to Blink.";
      } else if (!status.blink_configured) {
        state.textContent = "Run setup.py to connect your Blink account.";
      } else if (!status.connected) {
        state.textContent = "Blink is ready to connect.";
      }
      const syncMeta = this.shadowRoot.querySelector("#sync-meta");
      syncMeta.textContent = status.last_sync
        ? `Last checked ${new Date(status.last_sync).toLocaleTimeString()} · checks about once a minute`
        : "New clips are checked about once a minute.";
      this.renderEvents(status.events);
    } catch (error) {
      this.shadowRoot.querySelector("#blink-state").textContent = error.message;
    }
  }

  renderEvents(events) {
    const list = this.shadowRoot.querySelector("#events");
    const count = this.shadowRoot.querySelector("#event-count");
    count.textContent = String(events.length);
    list.replaceChildren();
    if (!events.length) {
      const empty = document.createElement("p");
      empty.className = "empty";
      empty.textContent = "No clips checked yet. Connect Blink to get started.";
      list.append(empty);
      return;
    }
    for (const event of events) {
      const row = document.createElement("div");
      row.className = "event-row";
      const mark = document.createElement("span");
      mark.className = event.detected ? "event-mark detected" : "event-mark";
      mark.textContent = event.detected ? "✳" : "○";
      const details = document.createElement("span");
      details.className = "event-details";
      const title = document.createElement("strong");
      title.textContent = event.detected ? "Cat detected" : "No cat";
      const subtitle = document.createElement("small");
      subtitle.textContent = `${event.camera} · ${new Date(event.timestamp).toLocaleString()}`;
      details.append(title, subtitle);
      const confidence = document.createElement("span");
      confidence.className = "event-confidence";
      confidence.textContent = `${Math.round(event.confidence * 100)}%`;
      row.append(mark, details, confidence);
      list.append(row);
    }
  }
}

customElements.define("cat-monitor", CatMonitor);
