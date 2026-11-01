"""Bounded metadata lookup and verified single-recording MeTube transfers."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
import shutil
import sys
import uuid
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

import httpx
from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from .music_library import MusicError, component

MAX_RECORDING = 40_000_000


def audio_signature(data):
    """Compare compressed audio after removing ID3 tags in memory only."""
    from mutagen.id3 import delete
    from mutagen.mp3 import MP3

    if not 1024 <= len(data) <= MAX_RECORDING:
        raise MusicError("The recording exceeds the import limits.")
    stream = io.BytesIO(data)
    try:
        if not 1 <= MP3(stream).info.length <= 950:
            raise ValueError("duration")
        delete(stream)
    except Exception as exc:
        raise MusicError("The downloaded file is not a usable MP3 recording.") from exc
    return hashlib.sha256(stream.getvalue()).hexdigest()


def library_bytes(library, track):
    path = Path(track.path)
    if path.is_symlink():
        raise MusicError("Refusing to clean up a download for a linked library file.")
    path = library.safe(path)
    with path.open("rb") as source:
        data = source.read(MAX_RECORDING + 1)
    return data


def youtube_id(url):
    if not isinstance(url, str):
        return ""
    try:
        parsed = urlsplit(url)
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"}:
        return ""
    if parsed.hostname in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
        value = parse_qs(parsed.query).get("v", [""])[0]
    elif parsed.hostname == "youtu.be":
        value = parsed.path.strip("/")
    else:
        return ""
    return value if re.fullmatch(r"[A-Za-z0-9_-]{11}", value) else ""


async def run_metadata(query):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "yt_dlp",
        "--ignore-config",
        "--no-cache-dir",
        "--no-playlist",
        "--flat-playlist",
        "--dump-single-json",
        "--skip-download",
        "--socket-timeout",
        "10",
        "--retries",
        "1",
        "--",
        "ytsearch5:" + query,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        async with asyncio.timeout(45):
            # Drain incrementally with a hard output bound.
            output = bytearray()
            while chunk := await process.stdout.read(65536):
                output.extend(chunk)
                if len(output) > 2_000_000:
                    raise MusicError("Music discovery returned too much metadata.")
            await process.wait()
        if process.returncode:
            raise MusicError(
                "Music lookup failed. The source may require authentication or an extractor update."
            )
        return json.loads(output)
    except (TimeoutError, ValueError) as exc:
        if isinstance(exc, MusicError):
            raise
        raise MusicError("Music lookup did not return usable metadata in time.") from exc
    finally:
        if process.returncode is None:
            process.kill()
        await process.wait()


class MeTube:
    def __init__(self, url, library):
        self.url = url.rstrip("/")
        self.library = library
        self.http = httpx.AsyncClient(timeout=30, follow_redirects=False)
        self.lock = asyncio.Lock()

    async def history(self):
        async with self.http.stream("GET", self.url + "/history") as response:
            response.raise_for_status()
            data = bytearray()
            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                if len(data) > 8_000_000:
                    raise MusicError("MeTube history is too large to inspect safely.")
        return json.loads(data)

    async def cleanup_mode(self):
        """Read MeTube's public configuration without changing shared settings."""
        try:
            await self._cleanup_mode()
        except MusicError:
            raise
        except (TimeoutError, OSError, ValueError, WebSocketException) as exc:
            raise MusicError("Could not verify MeTube's file cleanup setting. Check its connection.") from exc

    async def _cleanup_mode(self):
        parsed = urlsplit(self.url)
        url = ("wss" if parsed.scheme == "https" else "ws") + "://" + parsed.netloc
        url += parsed.path.rstrip("/") + "/socket.io/?EIO=4&transport=websocket"
        async with asyncio.timeout(8):
            async with connect(url, open_timeout=5, max_size=8_000_000) as socket:
                hello = await socket.recv()
                if not isinstance(hello, str) or not hello.startswith("0"):
                    raise MusicError("Could not verify MeTube's file cleanup setting.")
                await socket.send("40")
                for _ in range(32):
                    packet = await socket.recv()
                    if packet == "2":
                        await socket.send("3")
                    elif isinstance(packet, str) and packet.startswith("42"):
                        event = json.loads(packet[2:])
                        if isinstance(event, list) and len(event) == 2 and event[0] == "configuration":
                            config = json.loads(event[1]) if isinstance(event[1], str) else event[1]
                            if isinstance(config, dict) and config.get("DELETE_FILE_ON_TRASHCAN") in (
                                "ask",
                                "true",
                            ):
                                return
                            raise MusicError(
                                "Set DELETE_FILE_ON_TRASHCAN=ask in MeTube to move music into the library."
                            )
        raise MusicError("Could not verify MeTube's file cleanup setting.")

    def source_route(self, row):
        filename = row.get("filename", "")
        parts = filename.replace("\\", "/").split("/") if isinstance(filename, str) else []
        if not parts or any(p in {"", ".", ".."} for p in parts) or not filename.lower().endswith(".mp3"):
            raise MusicError("MeTube returned an unsafe or unsupported filename.")
        return self.url + "/audio_download/" + "/".join(quote(p, safe="") for p in parts)

    async def source_bytes(self, row):
        data = bytearray()
        async with self.http.stream("GET", self.source_route(row)) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(65536):
                data.extend(chunk)
                if len(data) > MAX_RECORDING:
                    raise MusicError("The recording exceeds the 40MB import limit.")
        return bytes(data)

    async def cleanup_import(self, row, track, source_digest):
        """Delete exactly one unchanged completed source after validating its copy."""
        try:
            async with asyncio.timeout(15):
                await self._cleanup_import(row, track, source_digest)
        except MusicError:
            raise
        except (TimeoutError, OSError, ValueError, httpx.HTTPError) as exc:
            raise MusicError(
                "The library file is saved, but source cleanup could not be confirmed. "
                "Repeat /music download for this artist/title to retry cleanup."
            ) from exc

    async def _cleanup_import(self, row, track, source_digest):
        await self.cleanup_mode()
        history = await self.history()
        identity = ("id", "url", "filename", "folder", "format", "download_type", "status")
        current = next(
            (r for r in history.get("done", []) if all(r.get(k) == row.get(k) for k in identity)), None
        )
        if not current or not isinstance(current.get("id"), str) or not current["id"]:
            raise MusicError("The MeTube job changed; its download was kept.")
        if (
            current.get("status") != "finished"
            or current.get("download_type") != "audio"
            or current.get("format") != "mp3"
            or youtube_id(current.get("url")) != track.video_id
            or current.get("chapter_files")
            or current.get("split_by_chapters")
        ):
            raise MusicError(
                "The MeTube job is not an individual completed recording; its download was kept."
            )
        if (
            current["filename"].replace("\\", "/")
            == Path(track.path).relative_to(self.library.root).as_posix()
        ):
            raise MusicError("The MeTube source overlaps the library file; its download was kept.")
        for group in ("done", "queue", "pending"):
            for other in history.get(group, []):
                if (
                    other is not current
                    and other.get("filename") == current.get("filename")
                    and other.get("folder") == current.get("folder")
                ):
                    raise MusicError("Another MeTube job uses this file; its download was kept.")
        source = await self.source_bytes(current)
        if hashlib.sha256(source).hexdigest() != source_digest:
            raise MusicError("The MeTube source changed; its download was kept.")
        local = await asyncio.to_thread(library_bytes, self.library, track)
        if await asyncio.to_thread(audio_signature, local) != await asyncio.to_thread(
            audio_signature, source
        ):
            raise MusicError("The library audio differs from MeTube; its download was kept.")
        latest = (await self.history()).get("done", [])
        if not any(all(r.get(k) == current.get(k) for k in identity) for r in latest):
            raise MusicError("The MeTube job changed; its download was kept.")
        # Mutagen changes tags, so whole-file hashes cannot establish audio identity.
        # The comparison above validates the actual compressed audio instead.
        # MeTube's PersistentQueue keys are info.url. info.id is the extractor's
        # video ID, and clear() silently ignores keys that aren't in that queue.
        response = await self.http.post(
            self.url + "/delete", json={"ids": [current["url"]], "where": "done", "delete_files": True}
        )
        response.raise_for_status()
        if response.json().get("status") != "ok":
            raise MusicError("The library file is saved, but MeTube could not remove its source copy.")
        remaining = await self.history()
        if any(r.get("url") == current["url"] for r in remaining.get("done", [])):
            raise MusicError("MeTube did not remove the completed job; its source copy was kept.")
        check = await self.http.head(self.source_route(current))
        if check.status_code != 404:
            raise MusicError(
                "The library file is saved, but MeTube kept its source copy. Check MeTube file permissions."
            )

    async def cleanup_existing(self, track):
        """Explicit maintenance can finish a previously interrupted transfer."""
        if not track.video_id:
            return
        async with self.lock:
            rows = (await self.history()).get("done", [])
            row = next(
                (
                    r
                    for r in rows
                    if youtube_id(r.get("url")) == track.video_id
                    and r.get("format") == "mp3"
                    and r.get("download_type") == "audio"
                    and r.get("status") == "finished"
                ),
                None,
            )
            if row:
                try:
                    source = await self.source_bytes(row)
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 404:
                        return
                    raise
                task = asyncio.create_task(
                    self.cleanup_import(row, track, hashlib.sha256(source).hexdigest())
                )
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    await task
                    raise

    async def finish_import(self, staging, track, row, source_digest):
        result = await self.library.import_file(staging, track)
        await self.cleanup_import(row, result, source_digest)
        return result

    async def prepare(self, track):
        async with self.lock:
            if existing := self.library.existing(track):
                return existing
            # Another library manager may have added files since first use.
            # Refresh before admitting any new shared download job.
            await self.library.scan()
            if existing := self.library.existing(track):
                return existing
            if not re.fullmatch(r"[A-Za-z0-9_-]{11}", track.video_id):
                raise MusicError("A valid individual YouTube recording is required.")
            space = await asyncio.to_thread(shutil.disk_usage, self.library.root)
            if space.free < 500_000_000:
                raise MusicError("Music downloads need at least 500MB of free library storage.")
            url = "https://www.youtube.com/watch?v=" + track.video_id

            def matching(data):
                return next(
                    (
                        r
                        for name in ("done", "queue", "pending")
                        for r in data.get(name, [])
                        if youtube_id(r.get("url", "")) == track.video_id
                        and r.get("format") == "mp3"
                        and r.get("download_type") == "audio"
                    ),
                    None,
                )

            row = matching(await self.history())
            if row and row.get("status") == "error":
                raise MusicError("MeTube has a failed job for this recording; inspect or retry it in MeTube.")
            if not row:
                await self.cleanup_mode()
                response = await self.http.post(
                    self.url + "/add",
                    json={
                        "url": url,
                        "download_type": "audio",
                        "format": "mp3",
                        "quality": "192",
                        "playlist_item_limit": 1,
                        "auto_start": True,
                    },
                )
                response.raise_for_status()
                if response.json().get("status") != "ok":
                    raise MusicError("MeTube rejected this recording. Check its logs.")
            async with asyncio.timeout(180):
                while True:
                    row = matching(await self.history())
                    if row and row.get("status") == "finished":
                        break
                    if row and row.get("status") == "error":
                        raise MusicError("MeTube could not download this recording. Check its logs.")
                    await asyncio.sleep(2)
            route = self.source_route(row)
            await self.cleanup_mode()
            # Keep the temporary file beside its final name. Unraid user shares
            # can span disks; a staging folder at library root can land elsewhere.
            staging_dir = self.library.safe(self.library.root / component(track.artist))
            await asyncio.to_thread(staging_dir.mkdir, parents=True, exist_ok=True)
            staging = staging_dir / (".bentebot-" + uuid.uuid4().hex + ".part")
            try:
                async with self.http.stream("GET", route) as response:
                    response.raise_for_status()
                    size = 0
                    digest = hashlib.sha256()
                    with staging.open("xb") as output:
                        async for chunk in response.aiter_bytes(65536):
                            size += len(chunk)
                            digest.update(chunk)
                            if size > MAX_RECORDING:
                                raise MusicError("The recording exceeds the 40MB import limit.")
                            writing = asyncio.create_task(asyncio.to_thread(output.write, chunk))
                            try:
                                await asyncio.shield(writing)
                            except asyncio.CancelledError:
                                await writing
                                raise
                    if size < 1024:
                        raise MusicError("MeTube returned an empty recording.")
                task = asyncio.create_task(self.finish_import(staging, track, row, digest.hexdigest()))
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    await task
                    raise
            finally:
                await asyncio.to_thread(staging.unlink, missing_ok=True)

    async def close(self):
        await self.http.aclose()
