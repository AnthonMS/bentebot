"use strict";
(() => {
  const $ = id => document.getElementById(id);
  let token = "", stream = null, timer = null, sequence = 0, busy = false, running = false;
  let interval = 1000, maxWidth = 1600, maxHeight = 1000, maxBytes = 1000000;
  const canvas = document.createElement("canvas");
  const status = text => { $("status").textContent = text; };
  async function request(path, options = {}) {
    const response = await fetch("/screen/" + path, {...options, cache: "no-store"});
    if (!response.ok) throw new Error((await response.text()).slice(0, 200));
    return response.json();
  }
  function stopTracks() {
    running = false;
    if (timer !== null) clearTimeout(timer);
    timer = null;
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    $("preview").srcObject = null;
    $("start").disabled = !token;
    $("stop").disabled = true;
  }
  async function stop() {
    stopTracks();
    const old = token;
    token = "";
    $("start").disabled = true;
    if (old) {
      try { await request("stop", {method: "POST", headers: {Authorization: "Bearer " + old}}); }
      catch (_) { /* Expired sessions are already stopped at the server. */ }
    }
    status("Sharing stopped. Pair again before sharing another window.");
  }
  $("pair-form").addEventListener("submit", async event => {
    event.preventDefault();
    if (token || running) await stop();
    try {
      const result = await request("pair", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({code: $("code").value.trim()})});
      token = result.token;
      sequence = 0;
      interval = result.frame_interval_ms;
      maxWidth = result.max_width;
      maxHeight = result.max_height;
      maxBytes = result.max_frame_bytes || 1000000;
      $("code").value = "";
      $("identity").textContent = "Paired to Discord user " + result.owner_id + ". Bound recipients: " + result.audience_ids.join(", ");
      $("start").disabled = false;
      status("Paired. Capture starts only when you choose a window.");
    } catch (error) { status(error.message); }
  });
  async function sendFrame() {
    if (!running || !stream || busy) return;
    busy = true;
    try {
      const video = $("preview");
      if (!video.videoWidth || !video.videoHeight) return;
      const scale = Math.min(1, maxWidth / video.videoWidth, maxHeight / video.videoHeight);
      canvas.width = Math.max(8, Math.round(video.videoWidth * scale));
      canvas.height = Math.max(8, Math.round(video.videoHeight * scale));
      canvas.getContext("2d", {alpha: false}).drawImage(video, 0, 0, canvas.width, canvas.height);
      let blob = null;
      for (const quality of [0.82, 0.65, 0.45]) {
        blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", quality));
        if (blob && blob.size <= maxBytes) break;
      }
      if (!running || !blob) return;
      if (blob.size > maxBytes) throw new Error("This picture is too large. Choose a smaller window or zoom out.");
      await request("frame", {method: "POST", headers: {Authorization: "Bearer " + token, "Content-Type": "image/jpeg", "X-Bentebot-Sequence": String(++sequence)}, body: blob});
      status("Sharing the selected window. Bentebot can use a fresh picture when answering you.");
    } catch (error) {
      stopTracks(); token = ""; $("start").disabled = true;
      status("Sharing stopped: " + error.message + ". Use /screen start to pair again.");
    } finally {
      busy = false;
      if (running) timer = setTimeout(sendFrame, interval);
    }
  }
  $("start").addEventListener("click", async () => {
    if (!token) return;
    if (!window.isSecureContext || !navigator.mediaDevices?.getDisplayMedia) {
      status("Screen capture requires HTTPS or localhost. Use the local screen_companion.py launcher for an Unraid HTTP endpoint.");
      return;
    }
    try {
      stream = await navigator.mediaDevices.getDisplayMedia({video: {frameRate: 2}, audio: false});
      stream.getVideoTracks()[0].addEventListener("ended", stop, {once: true});
      $("preview").srcObject = stream;
      await $("preview").play();
      running = true;
      $("start").disabled = true; $("stop").disabled = false;
      await sendFrame();
    } catch (error) { stopTracks(); status("Capture not started: " + error.message); }
  });
  $("stop").addEventListener("click", stop);
  window.addEventListener("pagehide", () => {
    stopTracks();
    if (token) fetch("/screen/stop", {method: "POST", keepalive: true, headers: {Authorization: "Bearer " + token}}).catch(() => {});
    token = "";
  });
})();
