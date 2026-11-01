"""Local music metadata, safe imports and credential-free catalog discovery."""

from __future__ import annotations

import asyncio
import os
import re
import threading
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path

import httpx

EXTENSIONS = {".mp3", ".flac", ".wav", ".ogg", ".opus", ".m4a", ".aac", ".aiff", ".wma"}


class MusicError(ValueError):
    pass


def normalized(value):
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", value).casefold()))


def component(value):
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")[:100]
    if not value or value.upper().split(".")[0] in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(10)),
        *(f"LPT{i}" for i in range(10)),
    }:
        value = "Unknown"
    return value


def recording_title(value):
    # Packaging labels do not distinguish recordings; live/acoustic/remaster
    # annotations are deliberately preserved as potentially different versions.
    labels = r"(?:official(?: music)? (?:video|audio)|lyric(?:s| video)?|hd|hq|4k)"
    value = re.sub(r"[\[(]\s*" + labels + r"\s*[\])]", "", value, flags=re.I)
    value = re.sub(r"\s*[-|]?\s*\b" + labels + r"\s*$", "", value, flags=re.I)
    return value.strip()


@dataclass(frozen=True)
class Track:
    artist: str
    title: str
    genre: str = ""
    path: str = ""
    video_id: str = ""
    duration: float = 0

    @property
    def key(self):
        return normalized(self.artist), normalized(recording_title(self.title))


class Library:
    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.tracks: list[Track] = []
        self.lock = asyncio.Lock()
        self.scanned = False
        self._metadata = {}

    def safe(self, path: Path):
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root.resolve()):
            raise MusicError("Music path escapes the configured library.")
        return resolved

    def _scan(self, stopped):
        import mutagen

        if not self.root.is_dir():
            raise MusicError("The music library is unavailable. Check its container mount.")
        found = []

        def finished():
            self._metadata = {t.path: self._metadata[t.path] for t in found}
            return found

        for base, dirs, files in os.walk(self.root, followlinks=False):
            if stopped.is_set():
                return finished()
            dirs[:] = [d for d in dirs if not d.startswith(".") and not Path(base, d).is_symlink()]
            for name in files:
                if stopped.is_set() or len(found) >= 100_000:
                    return finished()
                path = Path(base, name)
                if path.suffix.lower() not in EXTENSIONS or path.is_symlink():
                    continue
                try:
                    path = self.safe(path)
                    stat = path.stat()
                    signature = (stat.st_size, stat.st_mtime_ns)
                    cached = self._metadata.get(str(path))
                    if cached and cached[0] == signature:
                        found.append(cached[1])
                        continue
                    tags = mutagen.File(path, easy=True)

                    def first(key, fallback, tags=tags):
                        return str((tags.get(key) or [fallback])[0]) if tags else fallback

                    artist = first(
                        "artist",
                        path.relative_to(self.root.resolve()).parts[0]
                        if path.parent != self.root
                        else "Unknown",
                    )
                    title = first("title", path.stem)
                    video = re.search(r"\[([A-Za-z0-9_-]{11})\]$", path.stem)
                    found.append(
                        Track(
                            artist,
                            title,
                            first("genre", ""),
                            str(path),
                            video[1] if video else "",
                            getattr(getattr(tags, "info", None), "length", 0),
                        )
                    )
                    self._metadata[str(path)] = signature, found[-1]
                except (OSError, ValueError, mutagen.MutagenError):
                    continue
                if len(found) >= 100_000:
                    return finished()
        return finished()

    async def scan(self):
        async with self.lock:
            stopped = threading.Event()
            scanning = asyncio.create_task(asyncio.to_thread(self._scan, stopped))
            try:
                self.tracks = await asyncio.shield(scanning)
            except asyncio.CancelledError:
                stopped.set()
                await scanning
                raise
            self.scanned = True
        return len(self.tracks)

    async def ensure(self):
        if not self.scanned:
            await self.scan()

    def existing(self, track):
        return next(
            (
                t
                for t in self.tracks
                if (t.key == track.key or track.video_id and t.video_id == track.video_id)
                and Path(t.path).is_file()
            ),
            None,
        )

    def select(self, query, mode):
        query = normalized(query)
        if mode == "radio":
            aliases = {"rnb": "r b", "r and b": "r b", "hiphop": "hip hop"}
            query = aliases.get(query, query)
            return [t for t in self.tracks if query in aliases.get(normalized(t.genre), normalized(t.genre))]
        if mode == "artist":
            return [t for t in self.tracks if normalized(t.artist) == query]
        terms = set(query.split()) - {"by", "af", "song", "sangen"}
        return [
            t for t in self.tracks if terms and terms <= set(normalized(t.artist + " " + t.title).split())
        ]

    def _import(self, staging: Path, track: Track):
        from mutagen.id3 import ID3, TALB, TCON, TIT2, TPE1
        from mutagen.mp3 import MP3

        try:
            audio = MP3(staging)
            if not 1 <= audio.info.length <= 950:
                raise ValueError("duration")
        except Exception as exc:
            raise MusicError("The downloaded file is not a usable MP3 recording.") from exc

        folder = self.safe(self.root / component(track.artist))
        folder.mkdir(parents=True, exist_ok=True)
        target = self.safe(folder / f"{component(track.title)} [{track.video_id}].mp3")
        if not target.exists():
            tags = ID3()
            tags.add(TPE1(encoding=3, text=track.artist))
            tags.add(TIT2(encoding=3, text=track.title))
            tags.add(TALB(encoding=3, text="Individual tracks"))
            if track.genre:
                tags.add(TCON(encoding=3, text=track.genre))
            tags.save(staging)
            with staging.open("r+b") as saved:
                os.fsync(saved.fileno())
            # Same filesystem hard link publishes the finished file atomically,
            # refuses overwrite even if another library writer raced us.
            try:
                os.link(staging, target)
            except FileExistsError:
                pass
            if os.name != "nt":
                directory = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        return replace(track, path=str(target))

    async def import_file(self, staging, track):
        async with self.lock:
            if existing := self.existing(track):
                return existing
            importing = asyncio.create_task(asyncio.to_thread(self._import, staging, track))
            try:
                result = await asyncio.shield(importing)
            except asyncio.CancelledError:
                result = await importing
                self.tracks.append(result)
                raise
            self.tracks.append(result)
            return result


class Catalog:
    """Public MusicBrainz API: one request per second, bounded results, no LLM."""

    def __init__(self):
        self.http = httpx.AsyncClient(
            timeout=20, headers={"User-Agent": "Bentebot/0.3.7 (https://github.com/AnthonMS/bentebot)"}
        )
        self.lock = asyncio.Lock()
        self.last = 0.0

    async def request(self, entity, query, offset=0):
        async with self.lock:
            loop = asyncio.get_running_loop()
            await asyncio.sleep(max(0, 1.1 - (loop.time() - self.last)))
            try:
                for attempt in range(3):
                    response = await self.http.get(
                        f"https://musicbrainz.org/ws/2/{entity}/",
                        params={"query": query, "fmt": "json", "limit": 25, "offset": offset},
                    )
                    if response.status_code not in {429, 503} or attempt == 2:
                        break
                    await asyncio.sleep(2 * (attempt + 1))
                response.raise_for_status()
                if len(response.content) > 4_000_000:
                    raise MusicError("Music discovery returned too much metadata.")
                return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                raise MusicError(
                    "Music discovery is unavailable; existing library music can still play."
                ) from exc
            finally:
                self.last = loop.time()

    async def discover(self, query, mode, offset=0):
        # Escape Lucene reserved syntax; user input is data, never a query operator.
        term = re.sub(r'([+\-!(){}\[\]^"~*?:\\/|&])', r"\\\1", query)
        if mode == "artist":
            artists = (await self.request("artist", f'artist:"{term}"'))["artists"]
            artist = next(
                (
                    a
                    for a in artists
                    if normalized(a["name"]) == normalized(query)
                    or any(normalized(alias["name"]) == normalized(query) for alias in a.get("aliases", []))
                ),
                None,
            )
            if not artist:
                return []
            search = f"arid:{artist['id']} AND status:official"
        else:
            tag = {"rnb": "r&b", "r and b": "r&b", "hiphop": "hip hop"}.get(normalized(query), query)
            tag = re.sub(r'([+\-!(){}\[\]^"~*?:\\/|&])', r"\\\1", tag)
            search = f'tag:"{tag}" AND status:official'
        data = await self.request("recording", search, offset)
        tracks = []
        for row in data.get("recordings", []):
            credit = row.get("artist-credit", [])
            artist_name = "".join(
                a.get("name", "") + a.get("joinphrase", "") for a in credit if isinstance(a, dict)
            )
            length = (row.get("length") or 0) / 1000
            if artist_name and row.get("title") and 30 <= length <= 900:
                track = Track(artist_name, row["title"], query if mode == "radio" else "", duration=length)
                if track.key not in {t.key for t in tracks}:
                    tracks.append(track)
        return tracks

    async def video(self, track):
        # Only metadata is fetched here; MeTube performs the single-file download.
        from .music_download import run_metadata

        result = await run_metadata(track.artist + " " + track.title + " official audio")
        entries = result.get("entries", [])
        for entry in entries[:5]:
            video_id = entry.get("id", "")
            duration = entry.get("duration") or 0
            evidence = normalized(entry.get("title", "") + " " + entry.get("channel", ""))
            expected = set(normalized(track.title).split())
            artist = normalized(track.artist)
            if (
                re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id)
                and 30 <= duration <= 900
                and not entry.get("is_live")
                and expected <= set(evidence.split())
                and artist in evidence
                and (not track.duration or abs(duration - track.duration) < max(25, track.duration * 0.2))
            ):
                return replace(track, video_id=video_id, duration=duration)
        raise MusicError("No suitable individual recording was found.")

    async def single(self, query):
        from .music_download import run_metadata

        data = await run_metadata(query)
        for entry in data.get("entries", [])[:5]:
            title = entry.get("title", "")
            artist = entry.get("channel", "Unknown").removesuffix(" - Topic")
            if " - " in title:
                artist, title = title.split(" - ", 1)
            video_id, duration = entry.get("id", ""), entry.get("duration") or 0
            if (
                re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id)
                and 30 <= duration <= 900
                and not entry.get("is_live")
            ):
                return Track(artist, recording_title(title), video_id=video_id, duration=duration)
        raise MusicError("No suitable individual recording was found.")

    async def close(self):
        await self.http.aclose()
