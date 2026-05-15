// Browser client — vanilla WebRTC, no Pipecat client lib.
// Captures mic, exchanges SDP + spec with the runner, plays remote TTS audio.

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const statusLabel = statusEl.querySelector(".label");
const errorsEl = $("errors");
const logEl = $("log");
const connectBtn = $("connect");
const disconnectBtn = $("disconnect");
const remoteAudio = $("remote-audio");
const pickSpecBtn = $("pick-spec");
const clearSpecBtn = $("clear-spec");
const specFileInput = $("spec-file");
const specSummary = $("spec-summary");
const languageSelect = $("language");
const contextVarsInput = $("context-vars");

let pc = null;
let localStream = null;
let currentSpec = null;  // parsed spec object, or null
let currentSpecName = null;  // filename for display

const log = (msg) => {
  const line = `[${new Date().toLocaleTimeString()}] ${msg}`;
  logEl.textContent = (logEl.textContent ? logEl.textContent + "\n" : "") + line;
  logEl.scrollTop = logEl.scrollHeight;
};

const setStatus = (state, label) => {
  statusEl.className = `status ${state}`;
  statusLabel.textContent = label;
};

const showError = (msg) => {
  errorsEl.hidden = false;
  errorsEl.textContent = msg;
};

const clearError = () => {
  errorsEl.hidden = true;
  errorsEl.textContent = "";
};

const updateSpecUI = () => {
  if (currentSpec) {
    const flowCount = (currentSpec.flows || []).length;
    const agentId = currentSpec.agent?.id || "(no id)";
    specSummary.textContent = `${currentSpecName} — ${agentId}, ${flowCount} flows`;
    specSummary.classList.add("loaded");
    clearSpecBtn.hidden = false;
    connectBtn.disabled = pc !== null;
  } else {
    specSummary.textContent = "no spec loaded";
    specSummary.classList.remove("loaded");
    clearSpecBtn.hidden = true;
    connectBtn.disabled = true;
  }
  populateLanguages(currentSpec);
};

const populateLanguages = (spec) => {
  // Default to "all" (empty value). Concrete options come from
  // agent.meta.languages, locked once a session is connected. Hidden entirely
  // when 0 or 1 language is configured — there's nothing to choose between.
  const langs = spec?.agent?.meta?.languages ?? [];
  languageSelect.innerHTML = "";
  const all = document.createElement("option");
  all.value = "";
  all.textContent = "all languages";
  languageSelect.appendChild(all);
  for (const code of langs) {
    const opt = document.createElement("option");
    opt.value = code;
    opt.textContent = code;
    languageSelect.appendChild(opt);
  }
  languageSelect.value = "";
  languageSelect.disabled = !spec || pc !== null;
  languageSelect.hidden = langs.length <= 1;
};

const loadSpecFromText = (text, name) => {
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    showError(`couldn't parse JSON: ${err.message}`);
    return false;
  }
  if (!parsed || typeof parsed !== "object" || !parsed.agent || !Array.isArray(parsed.flows)) {
    showError("spec must be an object with `agent` and `flows` fields");
    return false;
  }
  currentSpec = parsed;
  currentSpecName = name;
  clearError();
  log(`loaded spec: ${name}`);
  updateSpecUI();
  return true;
};

const loadSpecFromFile = async (file) => {
  if (!file) return;
  try {
    const text = await file.text();
    loadSpecFromText(text, file.name);
  } catch (err) {
    showError(`couldn't read file: ${err.message}`);
  }
};

async function connect() {
  clearError();
  connectBtn.disabled = true;
  languageSelect.disabled = true;
  setStatus("connecting", "connecting…");
  log("requesting microphone");

  try {
    localStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (err) {
    setStatus("error", "mic denied");
    showError(`microphone access: ${err.message}`);
    connectBtn.disabled = false;
    return;
  }

  pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
  });

  pc.addEventListener("track", (event) => {
    log(`remote track: ${event.track.kind}`);
    remoteAudio.srcObject = event.streams[0];
  });

  pc.addEventListener("connectionstatechange", () => {
    log(`pc state: ${pc.connectionState}`);
    if (pc.connectionState === "connected") {
      setStatus("connected", "connected");
      disconnectBtn.disabled = false;
    } else if (["failed", "disconnected", "closed"].includes(pc.connectionState)) {
      teardown();
    }
  });

  pc.addEventListener("iceconnectionstatechange", () => {
    log(`ice: ${pc.iceConnectionState}`);
  });

  for (const track of localStream.getTracks()) {
    pc.addTrack(track, localStream);
  }
  // Pipecat's SmallWebRTC transport expects the answerer to send audio back —
  // but adding a recvonly audio transceiver explicitly avoids edge cases where
  // the offer omits it.
  pc.addTransceiver("audio", { direction: "sendrecv" });

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  log("posting offer");

  let answer;
  try {
    const offerBody = { sdp: offer.sdp, type: offer.type };
    if (currentSpec) {
      offerBody.spec = currentSpec;
    }
    if (languageSelect.value) {
      offerBody.language = languageSelect.value;
    }
    const raw = contextVarsInput?.value.trim();
    if (raw) {
      let parsed;
      try {
        parsed = JSON.parse(raw);
      } catch (e) {
        setStatus("error", "context vars invalid");
        showError(`context vars must be valid JSON: ${e.message}`);
        teardown();
        return;
      }
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        offerBody.context_vars = parsed;
      } else {
        setStatus("error", "context vars invalid");
        showError("context vars must be a JSON object (not an array or primitive)");
        teardown();
        return;
      }
    }
    const res = await fetch("/api/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(offerBody),
    });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`offer rejected (${res.status}): ${detail}`);
    }
    answer = await res.json();
  } catch (err) {
    setStatus("error", "offer failed");
    showError(err.message);
    teardown();
    return;
  }

  log(`got answer pc_id=${answer.pc_id}`);
  await pc.setRemoteDescription(answer);
}

function teardown() {
  if (pc) {
    pc.getSenders().forEach((s) => s.track && s.track.stop());
    pc.close();
    pc = null;
  }
  if (localStream) {
    localStream.getTracks().forEach((t) => t.stop());
    localStream = null;
  }
  remoteAudio.srcObject = null;
  setStatus("idle", "idle");
  connectBtn.disabled = !currentSpec;
  disconnectBtn.disabled = true;
  languageSelect.disabled = !currentSpec;
}

connectBtn.addEventListener("click", connect);
disconnectBtn.addEventListener("click", () => {
  log("disconnecting");
  teardown();
});

pickSpecBtn.addEventListener("click", () => specFileInput.click());
specFileInput.addEventListener("change", (e) => {
  loadSpecFromFile(e.target.files[0]);
  specFileInput.value = "";  // allow re-picking the same file
});

clearSpecBtn.addEventListener("click", () => {
  currentSpec = null;
  currentSpecName = null;
  clearError();
  log("spec cleared");
  updateSpecUI();
});

// Initial state — no spec yet, Connect disabled.
updateSpecUI();

// Tear down on tab close so the runner sees a clean disconnect.
window.addEventListener("beforeunload", teardown);
