# Local Intelligence Bro (Libby) — Agent Handoff

Generated: 2026-07-19

## Mission and product vision

Libby is a completely local, self-contained, always-on voice assistant running on a Steam Deck. The target experience is an appliance rather than a developer demo:

- Private/local operation: speech recognition, agent reasoning, model inference, speech synthesis, conversation history, microphone capture, playback, and UI all run on the Deck.
- As close to real-time conversation as the hardware permits.
- A feminine identity: Libby is a girl and currently speaks with Pocket TTS's **Eve** voice.
- A cute, highly legible fullscreen face on a black background. The face itself always consists of exactly three symbols: two eyes and one mouth.
- Expressive listening, thinking, speaking, happy, idle, and error states.
- Interruption/barge-in: tapping while Libby speaks cancels the active turn and starts a new recording.
- Persistent local conversation memory by thread ID.
- Future direction: wake-word activation, stronger acoustic echo cancellation, continuous/full-duplex interaction, richer local tools, and potentially lightweight room satellites. The current implementation is deliberately Deck-only; no Pi satellite is deployed.

Correct terminology: the local model is **Gemma**, not Gemini.

## Security and credential note

No password, private key, token, or other credential is included in this document. A Deck password was shared in the original chat and must be treated as compromised; rotate it if that has not already happened. Use SSH keys for ongoing access. Do not copy a private SSH key between machines or commit credentials.

The HTTP and WebSocket API intentionally bind only to `127.0.0.1` on the Deck.

## Connecting from another computer

The Deck user and mDNS hostname are:

```bash
ssh deck@steamdeck.local
```

The original workstation already has key-based access, but a new computer will need its own public key authorized. Recommended flow:

```bash
ssh-keygen -t ed25519
ssh-copy-id deck@steamdeck.local
ssh deck@steamdeck.local
```

Have the user enter the Deck password interactively for `ssh-copy-id`; never place it in a command, handoff, config file, or chat. If mDNS is unavailable, obtain the Deck's current LAN address from the Deck or router and use `deck@<current-lan-ip>`. Do not rely on an old DHCP address.

To inspect the local web UI/API from another machine without exposing it to the LAN:

```bash
ssh -L 18001:127.0.0.1:8001 deck@steamdeck.local
# Then open http://127.0.0.1:18001/
```

## Canonical deployment and source locations

The latest live source is on the Deck:

```text
/home/deck/local-intelligence-bro
```

The original workstation source directory was:

```text
~/Documents/local-intelligence-bro
```

For a new workstation, copy the current source from the Deck while excluding the Deck-specific virtual environment and private conversation database:

```bash
mkdir -p local-intelligence-bro
rsync -a \
  --exclude=.venv \
  --exclude=__pycache__ \
  --exclude=data/ \
  deck@steamdeck.local:/home/deck/local-intelligence-bro/ \
  local-intelligence-bro/
cd local-intelligence-bro
```

`data/libby.sqlite3` on the Deck is the live conversation database. Do not overwrite, publish, or casually copy it.

Important project files:

- `api.py` — FastAPI HTTP/WebSocket server, audio input protocol, agent token streaming, incremental phrase-to-speech pipeline, cancellation.
- `libby.py` — Gemma/Deep Agent configuration, feminine identity prompt, Ollama warm-up, SQLite path.
- `speech.py` — Pocket TTS service, **Eve** voice selection, startup warm-up, float-to-PCM conversion.
- `transcription.py` — persistent Faster Whisper `base.en` CPU INT8 service.
- `voice_client.py` — ALSA-based diagnostic microphone/WAV WebSocket client.
- `static/index.html` — entire fullscreen face, state machine, browser microphone capture/resampling, streamed audio playback, barge-in.
- `pyproject.toml` / `uv.lock` — Python 3.14 project and CPU-only PyTorch source configuration.
- `.python-version` — Python 3.14.

There are no known specs, ADRs, or issues to reference. Exact copies of the deployed service definitions are tracked under `deploy/systemd/`.

## Steam Deck platform

Last verified platform:

- SteamOS 3.8.14
- Valve/Neptune Linux 6.16.12
- AMD Steam Deck APU/GPU, 16 GB unified memory
- KDE Plasma desktop session (Wayland)
- Ubuntu 24.04 Distrobox container named `libby`
- uv 0.11.29
- uv-managed CPython 3.14.6
- Ollama 0.32.1 installed under `/home/deck/.local/ollama`
- Chromium installed as Flatpak `org.chromium.Chromium`

The project virtual environment is:

```text
/home/deck/local-intelligence-bro/.venv
```

The Hugging Face model/voice caches are under the Deck user's normal cache directories. Do not remove them unless intentionally forcing a redownload.

## Runtime architecture

```text
Deck microphone
  -> Chromium getUserMedia
  -> browser downsample to mono PCM S16LE at 16 kHz
  -> WebSocket /ws/chat
  -> Faster Whisper base.en (CPU INT8)
  -> Libby Deep Agent
  -> Gemma 4B via Ollama experimental Vulkan backend (AMD GPU)
  -> streamed AIMessageChunk text
  -> incremental phrase buffer
  -> Pocket TTS 2.1.0, Eve voice (CPU)
  -> streamed PCM S16LE at 24 kHz
  -> WebSocket binary frames
  -> browser Web Audio scheduling
  -> Deck speakers
```

### Model placement

Ollama model:

```text
gemma4:e4b-it-qat
```

Model storage:

```text
/home/deck/.ollama/models
```

Ollama was verified at 100% GPU placement through its experimental Vulkan backend with a 16,384-token context. The model is configured with `keep_alive=-1`.

Pocket TTS and Whisper run on CPU. The deployed Python runtime was verified as:

```json
{
  "torch_version": "2.13.0+cpu",
  "torch_cuda_version": null,
  "cuda_available": false,
  "nvidia_or_cuda_packages": []
}
```

`pyproject.toml` explicitly sources `torch` from `https://download.pytorch.org/whl/cpu`. Do not remove that source constraint: normal Linux PyTorch resolution previously installed CUDA, Triton, and many `nvidia-*` packages that are useless on this AMD device.

## Boot and service behavior

User lingering is enabled for `deck`. Three user services are enabled:

```text
/home/deck/.config/systemd/user/libby-ollama.service
/home/deck/.config/systemd/user/libby-api.service
/home/deck/.config/systemd/user/libby-face.service
```

Repository copies:

```text
deploy/systemd/libby-ollama.service
deploy/systemd/libby-api.service
deploy/systemd/libby-face.service
```

Inspect rather than recreating them from memory:

```bash
systemctl --user cat libby-ollama.service
systemctl --user cat libby-api.service
systemctl --user cat libby-face.service
```

Behavior:

1. `libby-ollama.service` enters the Ubuntu `libby` Distrobox and runs the manually installed Ollama server with `OLLAMA_VULKAN=1`, `OLLAMA_IGPU_ENABLE=1`, the Deck model path, and Ollama library path.
2. `libby-api.service` requires Ollama, enters the Distrobox, and runs uvicorn from the project venv on `127.0.0.1:8001`.
3. During API startup, Libby loads Gemma with infinite keep-alive, performs one real Deep Agent warm-up request, loads Pocket TTS and the Eve voice state, and performs a short discarded TTS warm-up. Health does not become ready until this finishes.
4. `libby-face.service` belongs to `graphical-session.target`, waits for `/health`, owns a Chromium Flatpak process, and launches `http://127.0.0.1:8001/` in kiosk mode with microphone permission and autoplay flags.
5. The face service kills stale Chromium instances before launch and on stop so systemd retains real process ownership.

Startup warm-up currently takes roughly 40–55 seconds. This is intentional: the UI waits, and the first user turn receives warm latency instead of a 30–45 second model cold start.

Current administrative commands:

```bash
systemctl --user status libby-ollama libby-api libby-face
systemctl --user restart libby-api
systemctl --user restart libby-face
journalctl --user -u libby-ollama -f
journalctl --user -u libby-api -f
journalctl --user -u libby-face -f
curl --fail http://127.0.0.1:8001/health
```

All three services were active and enabled at handoff time.

### Always-awake setup

The following system sleep targets were masked:

```text
sleep.target
suspend.target
hibernate.target
hybrid-sleep.target
suspend-then-hibernate.target
```

Verify with:

```bash
systemctl is-enabled sleep.target suspend.target hibernate.target hybrid-sleep.target suspend-then-hibernate.target
```

The screen may blank, but system services remain running. No software can prevent shutdown when the battery reaches zero; appliance use should remain powered.

## API and WebSocket behavior

HTTP routes:

- `GET /` — fullscreen Libby face (`static/index.html`), sent with `Cache-Control: no-store`.
- `GET /health` — `{"status":"ok"}` once startup warming is complete.
- `POST /chat` — diagnostic JSON chat endpoint.
- `WS /ws/chat` — primary text/audio protocol.

Conversation state uses `AsyncSqliteSaver` and thread IDs in:

```text
/home/deck/local-intelligence-bro/data/libby.sqlite3
```

Primary client events include:

```text
Client -> audio.input.start
Server -> audio.input.ready
Client -> binary PCM microphone frames
Client -> audio.input.stop
Server -> transcription.started
Server -> transcript
Server -> turn.started
Server -> assistant.delta (many)
Server -> audio.synthesizing
Server -> audio.started
Server -> binary PCM response chunks (many)
Server -> audio.completed
Server -> turn.completed
```

Text turns use `turn.start`; interruption uses `turn.cancel`. Cancellation was verified to emit `turn.cancelled` with no binary audio, `audio.completed`, or `turn.completed` events afterward.

## Incremental Gemma-to-Pocket speech

Do not regress this to full-response TTS. `api.py` starts a background Pocket audio consumer before Gemma generation. As `AIMessageChunk` tokens arrive, they are sent to the browser immediately and accumulated into speakable phrases.

Current phrase boundaries:

- Hard boundary at `.`, `!`, `?`, or newline after at least 24 characters.
- Soft boundary at comma, semicolon, or colon after at least 40 characters.
- Fallback split around 72 characters.
- Any remainder flushes when Gemma finishes.

The first ready phrase is synthesized while Gemma continues generating later text. Pocket streams each phrase's PCM chunks immediately. A verified four-sentence run produced:

```text
First response text:        1.485 s
Audio synthesis event:      2.269 s
First audio:                2.600 s
Last Gemma text:             4.321 s
Audio started before end:    true
Generation/audio overlap:    1.722 s
```

Pocket cannot generate intelligible speech from individual tokens; short phrase buffering is the quality/latency compromise. Phrase-to-phrase prosody and buffer thresholds are good future tuning targets.

## Fullscreen face and interaction

The face is a single static HTML file with no frontend build step. It has a black background, pink glow, captions, and exactly three face-symbol elements.

Verified states:

```text
Idle       ● ●  ⌣
Listening  ◉ ◉  ○
Thinking   ◔ ◕  ·
Speaking   ^ ^  ◡
Happy      ⌒ ⌒  ▽
Error      × ×  ︵
```

Controls:

- Tap Libby or press Space: begin listening.
- Tap/Space again: stop recording and submit.
- Tap while speaking/thinking during an active turn: cancel, discard pending audio, then begin a new recording.

The browser requests `echoCancellation`, `noiseSuppression`, `autoGainControl`, and mono input. It uses a `ScriptProcessorNode`, averages/downsamples the browser input rate to 16 kHz PCM16, and streams it over the socket. Output PCM chunks are converted to Web Audio buffers and scheduled against a monotonically increasing playback head.

The live Deck browser was verified with a real `Default` microphone track (`live`, unmuted), a 1280x800 viewport matching the screen, and a complete fixture-driven browser turn. Face transitions for that run were roughly:

```text
thinking  0.067 s
speaking  3.784 s
happy     8.057 s
idle      8.908 s
```

Production Chromium does not expose a remote debugging port. A localhost-only CDP port was enabled temporarily for verification and removed afterward.

The kiosk was verified in KDE Plasma Desktop Mode. Gaming Mode behavior was not verified; `libby-face.service` currently depends on the graphical session target and Wayland Plasma environment.

## Speech engine decisions and measured history

The project tried multiple TTS engines on the Deck:

- Chatterbox Turbo: too slow. Warm synthesis took about 17.6 seconds for 4.9 seconds of audio; approximately 3.6x slower than real time.
- KittenTTS Nano FP32: full utterance generation was fast, about 1.05 seconds for 7.7 seconds of audio (RTF about 0.137), but the tested path waited for the whole waveform.
- Pocket TTS 2.1.0: selected because its streaming API produced the first audio chunk around 0.21–0.27 seconds after receiving a phrase and supports voice prompting/cloning.
- Kokoro ONNX was researched but never deployed.

Trial environments were removed. Chatterbox, KittenTTS, CUDA, NVIDIA, and Triton packages are not part of the production venv.

Voice progression during tuning was Alba -> Jane -> **Eve**. The current required voice is Eve; do not accidentally revert `DEFAULT_VOICE` in `speech.py`.

## Deploying code changes from a workstation

From the project directory on a workstation with authorized SSH access:

```bash
rsync -a \
  --exclude=.venv \
  --exclude=.git \
  --exclude=__pycache__ \
  --exclude=data/ \
  --exclude='*.pyc' \
  ./ deck@steamdeck.local:/home/deck/local-intelligence-bro/
```

If dependencies changed:

```bash
ssh deck@steamdeck.local \
  distrobox enter libby -- \
  /home/deck/.local/bin/uv sync \
  --project /home/deck/local-intelligence-bro
```

Then restart and wait for the proactive warm-up:

```bash
ssh deck@steamdeck.local systemctl --user restart libby-api.service
ssh deck@steamdeck.local \
  curl --retry 180 --retry-connrefused --retry-delay 1 \
  --fail --silent http://127.0.0.1:8001/health
```

The face reconnects its WebSocket automatically across an API restart. Restart `libby-face.service` only for kiosk/service changes or if Chromium is unhealthy.

Do not sync a workstation `data/` directory onto the Deck; that can destroy live conversation history. Do not sync a workstation `.venv`; the Deck environment is managed by uv inside the Ubuntu Distrobox.

## Verification expectations for future changes

For anything affecting speech or protocol, verify on the real Deck rather than only importing modules:

1. `curl /health` succeeds after restart.
2. All three user services are active.
3. A text WebSocket turn streams `assistant.delta` and audio binary chunks.
4. For a multi-sentence response, `audio.started` occurs before the last `assistant.delta`.
5. Playback occurs through the Deck speakers.
6. A microphone turn reaches `transcript`, agent response, Pocket audio, and playback.
7. Cancellation after the first audio frame produces `turn.cancelled` and no stale events.
8. The face traverses listening -> thinking -> speaking -> happy -> idle.
9. `torch.__version__` ends in `+cpu` and there are no `nvidia-*`, `cuda-*`, or `triton` distributions in the project venv.

Useful diagnostic client:

```bash
/home/deck/local-intelligence-bro/.venv/bin/python \
  /home/deck/local-intelligence-bro/voice_client.py \
  --seconds 5 \
  --thread diagnostic
```

## Known limitations and next priorities

1. **Wake word is not implemented.** Interaction is push-to-talk via face tap or Space.
2. **Not true simultaneous full duplex.** Barge-in cancels the active turn and then starts recording; Libby does not continuously listen while speaking.
3. **Whisper is lazy-loaded.** The first microphone turn after API start can be slower. A safe transcriber warm-up could reduce that if a silent/no-speech warm-up can be implemented without affecting behavior.
4. **AEC is browser-provided only.** The Deck microphone and speaker were independently validated, but acoustic echo cancellation under loud simultaneous playback has not been characterized.
5. **Browser input uses deprecated `ScriptProcessorNode`.** Move capture to an `AudioWorklet` for lower jitter and long-term browser compatibility.
6. **Basic downsampling.** The browser averages source samples into 16 kHz buckets; a proper low-pass/resampler may improve recognition.
7. **Phrase prosody.** Pocket treats each queued phrase as a separate generation. Tune boundaries or investigate a continuous incremental-conditioning strategy if audible seams are distracting.
8. **Boot readiness is slow by design.** Full agent/TTS warm-up takes roughly 40–55 seconds.
9. **Desktop Mode dependency.** Confirm or implement launch behavior in SteamOS Gaming Mode if that becomes a requirement.
10. **Password hygiene.** Confirm the previously exposed Deck password was rotated.
11. **No automated permanent test suite exists.** Verification so far used actual WebSocket clients, browser automation against the live Deck kiosk, and speaker/microphone smoke tests.

## Suggested skills for the next agent

- Invoke the local `handoff` skill before ending a substantial session so the next machine/session receives updated deployment state and measured results.
- If the next agent environment offers a browser/UI automation skill, use it for every face, microphone, playback, or kiosk change; this project has already benefited from driving the real Deck Chromium over a temporary localhost-only SSH/CDP tunnel.
- If an audio/DSP or WebRTC skill is available, use it before redesigning echo cancellation, resampling, wake-word capture, or true full-duplex behavior.
- If a systemd/SteamOS deployment skill is available, use it before changing boot targets, Gaming Mode integration, sleep policy, or Flatpak kiosk lifecycle.

## Immediate continuation recommendation

Start by connecting with SSH, reading the deployed source and all three service units, and confirming current health. Then prioritize one of:

1. Warm Faster Whisper safely during API startup.
2. Replace `ScriptProcessorNode` with an `AudioWorklet`.
3. Measure and improve phrase-boundary prosody while retaining the verified Gemma/Pocket overlap.
4. Add a local wake-word path and a deterministic AEC/barge-in test.
5. Make the face launch reliably in SteamOS Gaming Mode as well as Plasma Desktop Mode.

Preserve the current invariants: local-only, Eve voice, feminine identity, exactly three face symbols, CPU-only PyTorch, AMD Vulkan Ollama, persistent SQLite memory, early phrase-level speech, and cancellation with no stale audio.
