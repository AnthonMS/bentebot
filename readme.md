# Bentebot

A self-hosted Discord companion for text, live voice conversations, web research,
screen context and owner-requested Python analysis. It can listen while you
speak, answer aloud, write code and sources in a linked text channel, and bring
useful research back while the conversation moves on.

**Version 0.3.3. Development preview.** Core and GPU speech have been used in
real Discord calls, including a GTX 1080 server. Sound/emotion accuracy,
multispeaker behavior and provider compatibility need broader testing.

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Fresh-server setup](#fresh-server-setup)
- [Voice services](#voice-services)
- [Optional capabilities](#optional-capabilities)
- [Commands and conversation](#commands-and-conversation)
- [Benchmarks](#benchmarks)
- [Privacy and permissions](#privacy-and-permissions)
- [Maintenance and troubleshooting](#maintenance-and-troubleshooting)
- [Planned work](#planned-work)
- [Development](#development)

## Features

| Area | Available now |
| --- | --- |
| Text | Mentions and allowlisted DMs; streamed chat through LiteLLM; per-channel models and server defaults. |
| Voice | Per-speaker streaming STT, sentence-sized TTS, interruption, stop/pause/resume and timing/error diagnostics. |
| Continuity | Admitted research survives ordinary continued speech. Useful findings are reconsidered against current context and returned later. |
| Call lifetime | Departures do not interrupt answers while humans remain. An empty call silences playback; a one-minute return window permits resumption before leaving. |
| Persona | Persistent server rules and local channel changes, stable IDs, modal editing and scoped undo. Server edits propagate; channel reset inherits the server; server reset restores built-ins everywhere. |
| Style | Direct replies, useful criticism, less automatic agreement, praise and padding. Still model-dependent. |
| Research | SearXNG search and bounded public-page reading; provider-native web search for qualified cloud aliases through LiteLLM. Spoken acknowledgements and written sources. |
| Utilities | Arithmetic, unit conversion and current IANA-zone time through deterministic commands and model tools. |
| Python | Owner-only Jupyter jobs, fresh kernels, bounded execution/output, attachments and PNG/CSV delivery. Duplicate artifacts are suppressed. |
| Languages | Automatic English/Danish recognition and optional Danish Piper synthesis. Different speakers are currently used for the two languages. |
| Sound/tone | Optional CUDA or CPU classification with confidence gates and cooldowns. Experimental; yawning unsupported. |
| Screen | Paired capture, short-lived frames, relevant followups and spatial crops; on-demand vision inference. |
| Memory | SQLite storage, keyword retrieval, summaries, retention/deletion and source-aware audience checks. |

## Requirements

Recommended full voice setup: **Linux, Docker Compose and NVIDIA CUDA**. The core
does not use CUDA; inference runs separately. CPU profiles exist, but CPU-only
conversational latency is not the primary target.

```text
Discord text/voice -> Bentebot core -> local SQLite volume
                         |
                         +-- LiteLLM -> local model server / authorized cloud provider
                         +-- WhisperLiveKit: streaming STT
                         +-- Speaches/Kokoro: English TTS
                         +-- Optional sound/tone + Danish Piper helper
                         +-- Optional SearXNG / Jupyter / paired screen + vision model
```

You need a Discord bot token, allowed server/user IDs, Docker with GPU-capable
Compose, an NVIDIA driver and NVIDIA Container Toolkit, a reachable LiteLLM
gateway, and model backends or authorized cloud credentials. Verify GPU access
inside a container as well as with host `nvidia-smi`. Internet is required for
Discord and initial image/model downloads.

Keep SQLite on local storage, **never SMB/NFS**. Allow several GB for images,
dependencies and caches. Optional helper models total roughly 440MB. The CUDA
helper has a 3GB system-RAM limit and a default 1024MiB combined ONNX arena budget.
That arena is not a total VRAM limit; CUDA libraries and other processes allocate more.

A GTX 1080 with 8GB has run this project's speech pipeline. Large chat/vision
models or video workloads can exhaust remaining VRAM. Choose a small quantized
chat model or a separate backend. Preserve the pinned Pascal-compatible runtime
until upgrades are qualified; arbitrary newer CUDA/framework builds may drop support.
Text-only core and tools can run without CUDA if inference is hosted elsewhere.

## Fresh-server setup

Commands use a Linux shell. On Unraid choose local appdata storage. Docker
deployment does not need host Python.

### 1. Clone and configure

```sh
git clone https://github.com/AnthonMS/bentebot.git
cd bentebot
cp example.env .env
chmod 600 .env
mkdir -p secrets
chmod 700 secrets
```

Edit `.env` in an editor. Never post it in Discord or commit it. The address
`192.168.50.10` below is an **example**: replace it with your server's private LAN
address. `example.env` documents settings; leave optional features disabled until ready.

```dotenv
BENTEBOT_DISCORD_TOKEN=YOUR_BOT_TOKEN
BENTEBOT_GUILD_IDS=YOUR_SERVER_ID
BENTEBOT_OWNER_IDS=YOUR_USER_ID
BENTEBOT_DM_USER_IDS=
BENTEBOT_CHAT_BASE_URL=http://192.168.50.10:4000/v1
BENTEBOT_CHAT_API_KEY=YOUR_PRIVATE_GATEWAY_KEY
BENTEBOT_CHAT_MODEL=bentebot-chat
```

IDs are numeric and lists are comma-separated. Use Discord Developer Mode to
copy IDs. Ownership grants privileged commands but does not replace server/DM
allowlists. An empty DM allowlist disables DMs.

### 2. Discord application

In the [Developer Portal](https://discord.com/developers/applications), create
the application/bot and enable **Message Content Intent** and **Server Members
Intent**. Invite with `bot` and `applications.commands` scopes.

Grant View Channel, Read Message History, Send Messages, Embed Links and Attach
Files where needed. Voice needs Connect, Speak and Use Voice Activity. Administrator
is unnecessary. Both user and bot need access to selected gated channels.

Commands register globally at startup; old guild copies are removed to avoid
duplicates. Missing commands can indicate an old image or failed startup, not
only client caching. The registration diagnostic reads Discord's actual inventory.

### 3. Chat gateway

Use an existing LiteLLM `/v1` URL/key if available. The alias must appear in
`/v1/models` and support streamed chat completion. Routing/fallback and upstream
authorization remain LiteLLM's responsibility. Bentebot does not directly invoke
a cloud subscription proxy or coding CLI.

For a new gateway, copy the minimal bring-your-own-backend example:

```sh
mkdir -p secrets/litellm
cp deploy/gateway/litellm.example.yaml secrets/litellm/config.yaml
```

Create private `secrets/litellm/gateway.env` in an editor:

```dotenv
LITELLM_MASTER_KEY=YOUR_LONG_RANDOM_GATEWAY_KEY
CHAT_UPSTREAM_MODEL=openai/YOUR_LOCAL_MODEL_ID
CHAT_UPSTREAM_BASE_URL=http://YOUR_MODEL_HOST:8080/v1
CHAT_UPSTREAM_API_KEY=YOUR_BACKEND_KEY_OR_LOCAL_PLACEHOLDER
```

`openai/` selects OpenAI-compatible transport, not necessarily hosted OpenAI.
For another provider use its actual provider/model and required credentials.
The example exposes alias `bentebot-chat`. Put the gateway key in Bentebot's
`BENTEBOT_CHAT_API_KEY`. Generate private random keys with `openssl rand -hex 32`;
save them in private files rather than shell command arguments.

```sh
chmod 600 secrets/litellm/gateway.env
SERVER_IP=192.168.50.10
docker pull docker.litellm.ai/berriai/litellm:latest
LITELLM_IMAGE=$(docker image inspect docker.litellm.ai/berriai/litellm:latest --format '{{index .RepoDigests 0}}')
docker run -d --name bentebot-gateway --restart unless-stopped \
  --env-file secrets/litellm/gateway.env \
  -v "$PWD/secrets/litellm/config.yaml:/app/config.yaml:ro" \
  -p "$SERVER_IP:4000:4000" \
  "$LITELLM_IMAGE" --config /app/config.yaml --port 4000
```

The pulled image is resolved to a digest before use. Record it for repeatability;
qualify new versions before replacing a working gateway. More options are in the
[official Docker guide](https://docs.litellm.ai/docs/proxy/docker_quick_start).
Keep the gateway authenticated and private.

Local chat still needs a separate model server and appropriately licensed
weights, for example a CUDA-enabled llama.cpp deployment. Alternatively use an
authorized provider. Bentebot does not download or schedule chat weights for you.

### 4. Build, check and start

```sh
docker compose build bentebot
docker compose run --rm --no-deps bentebot python -m bentebot --check
docker compose run --rm --no-deps bentebot python -m bentebot --validate-config
docker compose up -d bentebot
docker compose logs --tail=80 bentebot
docker compose exec -T bentebot python -m bentebot --check-commands
```

Offline checks use in-memory SQLite with no login/inference. Config validation
does not prove endpoint reachability. Try `/diagnostics`, `/model list`,
`/persona show` and a text mention.

Core uses UID 99/GID 100, a named local data volume and internal readiness checks.
No host port is published by default. For a bind mount, create local storage
writable by those IDs and use private `docker-compose.override.yml`:

```yaml
services:
  bentebot:
    volumes:
      - /YOUR/LOCAL/APPDATA/bentebot/data:/app/data
```

Unraid normally uses local `/mnt/user/appdata/...`. Preserve working mounts during
updates. Allow outbound HTTPS, WebSockets and Discord voice UDP.

## Voice services

### Streaming STT and English TTS

```sh
cp deploy/speech/example.env deploy/speech/speech.env
chmod 600 deploy/speech/speech.env
```

Edit the private speech file, for example:

```dotenv
SPEECH_BIND_IP=192.168.50.10
STT_PORT=8765
TTS_PORT=8766
STT_MODEL=small
STT_LANGUAGE=auto
STT_BEAM_SIZE=3
STT_CUDA_COMPUTE_TYPE=int8_float32
```

```sh
docker compose --env-file deploy/speech/speech.env \
  -f deploy/speech/compose.yaml -f deploy/speech/compose.cuda.yaml up -d --build
```

This builds pinned WhisperLiveKit/CTranslate2, starts pinned CUDA Speaches and
creates new model caches. No old test volumes/external networks are needed.
Initial downloads and GPU initialization take time. CPU PyTorch handles VAD;
CUDA CTranslate2 handles transcription. A CPU Torch banner alone does not prove
CPU transcription fallback.

Install Kokoro in the running TTS service:

```sh
curl --fail --request POST \
  http://192.168.50.10:8766/v1/models/speaches-ai/Kokoro-82M-v1.0-ONNX
```

Set core `.env`:

```dotenv
BENTEBOT_STT_URL=ws://192.168.50.10:8765/v1/listen
BENTEBOT_STT_LANGUAGE=auto
BENTEBOT_TTS_BASE_URL=http://192.168.50.10:8766/v1
BENTEBOT_TTS_MODEL=speaches-ai/Kokoro-82M-v1.0-ONNX
BENTEBOT_TTS_VOICE=af_heart
BENTEBOT_TTS_VOICES=af_heart
```

Both endpoints are required for voice. STT needs WhisperLiveKit's
**Deepgram-compatible `/v1/listen` WebSocket**, not `/asr` or a REST file API:
PCM16, 16kHz, mono, interim results and endpointing. Silence is padded when
Discord stops sending packets. Independent streams are bounded by
`BENTEBOT_MAX_SPEAKERS` (default six).

Reload `.env` by recreating core:

```sh
docker compose up -d --no-deps bentebot
```

`docker compose restart` does not reload environment changes. Join a channel,
run `/voice join` and inspect `/voice status` after an answer. The example speech
APIs are unauthenticated: restrict them to trusted LAN clients, not the Internet.

For CPU experiments omit `-f deploy/speech/compose.cuda.yaml` and set
`STT_COMPUTE_TYPE=int8`. CPU speech works but can be slow. The CPU TTS tag floats;
record its digest after qualification. Pascal requires compatible pinned packages.

### Sound, tentative tone and Danish TTS

Set core `.env` for the optional YAMNet/emotion2vec/Piper helper:

```dotenv
BENTEBOT_AUDIO_BIND_IP=192.168.50.10
BENTEBOT_AUDIO_URL=http://192.168.50.10:18769
BENTEBOT_AUDIO_TOKEN=YOUR_RANDOM_TOKEN_OF_AT_LEAST_24_CHARACTERS
BENTEBOT_AUDIO_AWARENESS_ENABLED=true
BENTEBOT_AUDIO_REACTIONS_ENABLED=true
BENTEBOT_AUDIO_GPU_MEMORY_MB=1024
BENTEBOT_TTS_DANISH_BASE_URL=http://192.168.50.10:18769/v1
```

Use a separate random helper token. Download checksum-pinned models and start:

```sh
mkdir -p data/audio-models
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" -w /work python:3.12-slim-bookworm \
  python deploy/audio/download_models.py --directory data/audio-models

docker compose --env-file .env -f deploy/audio/compose.yaml \
  -f deploy/audio/compose.cuda.yaml up -d --build
curl --fail http://192.168.50.10:18769/health
docker compose up -d --no-deps bentebot
```

Models total roughly 440MB; images/caches need more disk. Preserve the downloaded
manifest. Startup verifies hashes; models mount read-only. Review upstream model
licenses/cards before redistributing weights.

CUDA readiness requires actual GPU compute in **both** classifiers. Health shows
providers and positive startup compute counts, not just visibility. Voice status
reports the last successful analysis backend. Omit the CUDA overlay to explicitly
select CPU; requested CUDA cannot silently become full CPU inference.

Cues include laughter, cough, sneeze, gasp, sigh, scream and crying. Reactions
are optional with cooldowns. Tone needs confidence and two agreeing windows;
`unknown` is valid. A verbal "achoo" or empathetic reply is not acoustic proof.
Private status shows candidates and rejection gates. **Yawning is unsupported.**
This does not replace STT or guarantee suppression of noise transcripts.

Danish uses CPU Piper `da_DK-talesyntese-medium`; English retains its configured
voice. These are different speakers. Ambiguous short text can retain English,
and helper errors fall back to existing TTS. Consistent bilingual voice identity
and a complete cloning backend remain planned.

## Optional capabilities

### Web research

After core creates `bentebot_default`:

```sh
docker compose -f deploy/web-tools/compose.yaml up -d
```

Adjust the external network name if the checkout/project name differs. Configure:

```dotenv
BENTEBOT_WEB_TOOLS_ENABLED=true
BENTEBOT_WEB_SEARCH_URL=http://searxng:8080
BENTEBOT_WEB_TOOL_MODELS=YOUR_TOOL_CAPABLE_LOCAL_ALIAS
```

An existing internal SearXNG also works if it supports JSON, is reachable from
core and has no human login redirect. Local aliases need reliable typed streaming
tool calls on **every** gateway upstream/fallback.

For provider-native research use exact, disjoint lists:

```dotenv
BENTEBOT_NATIVE_WEB_OPENAI_MODELS=YOUR_QUALIFIED_OPENAI_ALIAS
BENTEBOT_NATIVE_WEB_ANTHROPIC_MODELS=YOUR_QUALIFIED_ANTHROPIC_ALIAS
```

The gateway must forward Responses or Messages APIs and provider search tools.
Names alone do not prove support. Never put an alias in multiple web lists.
Native research still uses LiteLLM, consumes normal provider quota and is not
guaranteed to beat local tools.

Acknowledgement follows initial model planning. Admitted research survives
ordinary speech and is assessed against current context. Explicitly say
"cancel the lookup" to abandon it; stop-speaking governs audible output. Jobs
are bounded and expire. Written sources retain destination permission checks.

### Owner-only Jupyter

Bring an existing authenticated Jupyter Server with REST and kernel WebSocket
APIs. A browser login page alone is insufficient:

```dotenv
BENTEBOT_UTILITY_TOOLS_ENABLED=true
BENTEBOT_JUPYTER_URL=http://YOUR_JUPYTER_HOST:8888
BENTEBOT_JUPYTER_TOKEN=YOUR_PRIVATE_JUPYTER_TOKEN
BENTEBOT_JUPYTER_TIMEOUT=25
```

Install numpy, pandas, matplotlib and IPython in its Python kernel.
**Fresh kernels and the conservative AST guard are not an OS sandbox.** The
server controls filesystem/network privileges; isolate shared execution properly.
Only configured owners can execute Python. Only kernels created by the bot are deleted.

Try `/tools python code:print(6 * 7)`. Upload UTF-8 CSV/text and explicitly request
analysis. Two attachments of up to 128KB each become `_inputs[filename]` bytes;
arbitrary server files are not inputs. Plot with `display(fig)` then `plt.close(fig)`.
PNG/CSV output is bounded and audience-checked.

### Paired screen capture

Configure a vision-capable alias:

```dotenv
BENTEBOT_SCREEN_ENABLED=true
BENTEBOT_VISION_MODEL=YOUR_VISION_ALIAS
BENTEBOT_SCREEN_BIND_IP=192.168.50.10
BENTEBOT_SCREEN_PUBLIC_URL=http://192.168.50.10:8768
```

```sh
docker compose -f docker-compose.yml -f deploy/screen/compose.override.yaml up -d --build bentebot
```

Keep the same `-f` arguments for subsequent changes, or put the settings in a
private root override. Plain HTTP LAN pages cannot directly use browser screen
capture. On the sharing computer run the Python localhost relay:

```sh
python scripts/screen_companion.py --target http://192.168.50.10:8768
```

Open its printed localhost URL, get a one-time code with `/screen start` and
select a window. `/screen stop` revokes it. Membership/permission changes require
fresh pairing. Frames expire in RAM. `BENTEBOT_VISION_MODE=direct` uses relevant
frames for replies; `observer` extracts bounded evidence first. Neither polls
the model continuously. Pixels are evidence only for their captured time/view.

Experimental native-viewer code is present but not needed for this setup. It
uses separate credentials and unsupported Discord behavior; automating normal
accounts can violate account-automation rules. Leave it disabled for the paired
companion path. No production viewer credentials/account IDs belong in Git.

### Local warming

An empty `BENTEBOT_VOICE_WARM_MODELS` disables model warming. List only exact
aliases whose **every** upstream/fallback is local. One-token refreshes happen
during an active call; use intervals below the backend's unload TTL. Known cloud
aliases are excluded, but the bot cannot inspect hidden routing behind arbitrary
custom names. Never mark an alias local if it can route to cloud. TTS warming
has separate controls. Normal user requests still consume provider quota.

## Commands and conversation

| Commands | Purpose |
| --- | --- |
| `/voice join`, `leave`, `stop`, `pause`, `resume`, `status` | Call control and audio/request/research/timing diagnostics. |
| `/model list`, `set` | Effective model and channel/server selection. |
| `/persona show`, `add`, `edit`, `remove`, `reset`, `undo` | Inherited personality; editing needs owner or Manage Server. |
| `/tools status`, `calculate`, `convert`, `time`, `python` | Ephemeral utilities; Python owner-only. |
| `/memory status`, `search`, `summarize`, `forget` | Permitted stored context and summaries. |
| `/speech voice`, `voices`, `interruptions` | Speech preferences. |
| `/speech clone`, `cancel_clone`, `delete_voice` | Reference enrollment; compatible external backend required. |
| `/screen start`, `stop`, `status`, `probe`, `inspect` | Capture and experimental screen diagnostics. |
| `/settings collection` | Channel collection settings. |
| `/diagnostics`, `/help` | Version/configuration/gateway status and overview. |

Mention the bot in permitted text or use an allowed DM. Model commands default
to the active bot call if the invoker is in it, otherwise the invoking text
channel. An explicit channel overrides that. Server defaults leave channel
overrides intact. Status separates current selection from earlier requests;
in-flight research retains its original model.

Voice-requested code is written with a spoken acknowledgement. Source links
are written rather than read aloud. Stop interrupts output while listening
continues; pause holds both. Interruption is distinct from abandoning research.

```text
/persona add text:Keep casual answers brief. scope:server
/persona edit rule_id:8 scope:channel channel:YOUR_CHANNEL
/persona reset scope:channel channel:YOUR_CHANNEL
```

Use the real ID from `/persona show`. Server edits replace local changes to that
same rule; unrelated local rules survive. Channel reset inherits current server
rules; server reset clears all channels to built-ins. Undo is scoped/bounded.
Editable style cannot override permissions or actual tool capabilities.

## Benchmarks

These are **development observations, not controlled same-input comparisons**.
CPUs, utterances, models, settings, cold/warm states and other workloads differ.
Do not calculate hardware speedup ratios from this table. Sanitized measurement
metadata is in [benchmarks/reference.json](benchmarks/reference.json).

| Task / metric | CPU observation | GTX 1080 observation | Interpretation |
| --- | --- | --- | --- |
| Streaming STT: speech end to final | 7.283-7.324s, i5-class server, two different real turns | 0.750-0.766s, three synthetic Whisper-base streams | GPU reduced the observed speech bottleneck; endpointing/settings matter. |
| Whole-file STT, 3.221s input | Not measured comparably | 0.804s first / 0.206s warm | Whole-file throughput is not live-call latency. |
| English Kokoro TTS | 3.063s first / 1.438-1.625s warm, 1.579s output, server CPU | 2.312s first / 0.188s warm, 1.877s output | Warm GPU synthesis was faster in these observations; output differs. |
| Sound/tone | 0.344s, 3.204s synthetic window, desktop CPU | 0.062-0.325s, seven status snapshots | Different inputs/hardware and mixed tone scheduling; not an accuracy comparison. |
| Danish Piper TTS | 0.305s for 3.204s output, desktop CPU | Still CPU in CUDA helper | GPU selection does not accelerate this voice. |
| Chat | No comparable CPU-only benchmark | No general GTX 1080 benchmark | External model, quantization, context and routing dominate. |
| Screen/vision | Not measured comparably | Not measured comparably | External model and image size matter. |
| Python/CSV/plots | Jupyter workload dependent | Not automatically GPU accelerated | Speech GPU access does not accelerate ordinary pandas/matplotlib. |

A separate RTX 4090 qualification measured 0.172-0.188s per three-second synthetic
sound/tone window. **That is not a GTX 1080 result.** GPU startup compute was
verified, but synthetic success does not prove microphone recognition accuracy.
Initial manual sound/tone snapshots still showed unknown/no event.

CPU servers can run core/tools with externally hosted chat. For voice, CPU
STT/TTS can dominate delay even with fast cloud chat. CUDA can also fail under
VRAM contention; reserve memory for other GPU applications. OOM immunity is not promised.

Measure your endpoints using the same phrase:

```sh
docker compose exec -T bentebot python -m bentebot.benchmark \
  --stt-url ws://192.168.50.10:8765/v1/listen \
  --tts-base-url http://192.168.50.10:8766/v1 \
  --phrase "The capital of Sweden is Stockholm." --repeats 3
```

Repeat against CPU and CUDA without changing model, phrase, voice or endpointing.
Record hardware, versions, cold/warm state and background load. This reports
synthesis and finalization, not LLM time or listener audio arrival. Voice status
has stage timings and actual upstream; overlapping timings must not all be added.

## Privacy and permissions

- Allowlisted guilds/DM users only. Enabled channel messages are collected while
  online by default, even without mentions, unless collection is disabled. This
  is not a bulk server-history importer.
- Context retains provenance and is checked against response recipients. Private
  channels, DMs and screens are not automatically shareable. Incomplete private
  permission/member information fails closed.
- SQLite stores transcripts/conversation, settings and summaries. Retention
  defaults to 90 days; model context has a separate budget. Retrieval is FTS5
  keyword search, not embedding-based semantic search.
- Core does not persist raw call audio. Frames and tentative cues are short-lived
  RAM data. External providers may store more. Optional voice references persist
  on their external provider until deletion.
- Chat, vision and research can send permitted context upstream. Understand data
  handling, quotas and costs. Warming exclusions do not make real requests free
  or enforce an upstream subscription budget.
- Fresh Jupyter kernels/AST guards do not isolate a host. Use proper isolation.
- Keep secrets, private overrides, databases, backups, logs, recordings, weights
  and AI working documents out of Git. Ignore rules do not erase a committed
  secret from history; rotate exposed credentials.
- Restrict exposed APIs to intended LAN clients. Do not accidentally publish
  speech, Jupyter, screen ingress or helper APIs to the Internet.

## Maintenance and troubleshooting

Back up state, preserve private configuration and mounts, then update:

```sh
git pull --ff-only
docker compose up -d --build --no-deps bentebot
docker compose exec -T bentebot python -m bentebot --check-commands
```

Keep optional Compose overlays on later invocations. Do not use `down -v` for
normal upgrades: it deletes named volumes. Memory and speech caches are separate.
Stop core before file-level database backups and restart afterward, or use
SQLite's online backup API. Copying only a live database can omit WAL state.
Backups contain private conversation data.

```sh
docker compose ps
docker compose logs --since=10m --tail=150 bentebot
docker compose exec -T bentebot python -c 'from bentebot import __version__; print(__version__)'
nvidia-smi
```

| Symptom | Check |
| --- | --- |
| Commands missing | Running version, startup logs, `--check-commands`; image build alone is insufficient. |
| Voice unavailable | Both speech URLs, service reachability, permissions and same-call authorization. |
| Recoverable STT warning | Actual STT traceback and GPU load. `OOMKilled=false` does not exclude CUDA OOM. |
| Wrong model appears | Voice/text target, channel override/server default and request timing. |
| No written source | Linked audience, provenance checks, successful page read and `web_source_delivery`. |
| Unknown tone/no reaction | Backend, analysis counts, gate/confidence, age and cooldown. Silence can be valid. |
| Helper native-library failure | Rebuild current image; it includes GLES/EGL and a loader check. |
| Jupyter failure | REST/WebSocket access, token, owners, packages and timeout. |
| Danish changes speaker | Separate language voices; consistent identity is unfinished. |
| Docker subnet exhaustion | Inspect networks; do not indiscriminately prune other applications. Helper uses the existing bridge. |
| Tool stream fails after HTTP 200 | Terminal SSE errors/completion and gateway logs. An older LiteLLM logging bug needed a guarded hotfix; qualify your installed version. |

Review logs before sharing. Core avoids transcript bodies and credentials in
operational logs; external services may log differently.

## Planned work

- More reliable microphone sound/tone recognition and confidence calibration.
- Consistent bilingual speaker identity and a qualified cloning backend.
- Natural timing, lower latency and broader multispeaker/provider validation.
- Better implicit screen awareness and multiple simultaneous shares.
- Stronger isolated execution and richer durable artifacts.
- Coordination of chat, speech, vision and other GPU workloads.
- More useful tools driven by conversational needs.
- On-demand images/music/other media, potentially on another larger GPU.
- Gateway compatibility and durable fixes across image updates.

These are plans, not implemented promises. Current caveats include imperfect
transcription, partial sound classes, model-dependent style, external API drift,
experimental voice receive and approximate playback progress. Viewer/reference
enrollment code does not establish support for every Go Live/cloning workflow.

## Development

Core requires **Python 3.12** because conversion uses `audioop`. Voice receive
is pinned to a reviewed community DAVE patch, not upstream stable. Inference
images have separate dependencies from the lightweight core.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/check.py
```

Windows activation: `.venv\Scripts\Activate.ps1`. The checker runs Ruff, formatting,
offline tests and startup inventory. Tests use mocks/synthetic fixtures, not
production credentials or live APIs. `python scripts/check.py --docker` also
builds/smoke-tests containers. Fresh-checkout CI is the reference for published
files; explicitly local operator tools/tests remain private.

Before publishing, run `python scripts/check_publication.py` to check tracked
paths and a secret scanner such as Gitleaks for file contents/history. CI checks
the path policy and runs a checksum-verified, pinned Gitleaks release with
redacted output. These checks complement credential handling; they cannot prove
that every possible sensitive string will be recognized automatically.

The public README is the setup reference. AI plans, handoffs, research and test
diaries are ignored. Legacy `src/` is historical; the active entry point is
`python -m bentebot` through `main.py`.

No project license was added here. Check repository licensing and upstream
model/dependency terms before redistribution or operating a public service.
