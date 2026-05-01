// Bare audio test — no spec, no dispatcher. Used to iterate on voices,
// STT, VAD, and the audio path without spec interpretation in the way.

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const statusLabel = statusEl.querySelector(".label");
const errorsEl = $("errors");
const logEl = $("log");
const connectBtn = $("connect");
const disconnectBtn = $("disconnect");
const remoteAudio = $("remote-audio");

let pc = null;
let localStream = null;

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

async function connect() {
  clearError();
  connectBtn.disabled = true;
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
  pc.addTransceiver("audio", { direction: "sendrecv" });

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  log("posting offer to /api/offer/raw");

  let answer;
  try {
    const res = await fetch("/api/offer/raw", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sdp: offer.sdp, type: offer.type }),
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
  connectBtn.disabled = false;
  disconnectBtn.disabled = true;
}

connectBtn.addEventListener("click", connect);
disconnectBtn.addEventListener("click", () => {
  log("disconnecting");
  teardown();
});

window.addEventListener("beforeunload", teardown);
