"""Bounded looping music sessions; no chat-model calls or scheduled inference."""

from __future__ import annotations

import asyncio
import logging
import random
from contextlib import asynccontextmanager

import discord

from .music_download import MeTube
from .music_intent import GENRES
from .music_intent import music_intent as music_intent
from .music_library import Catalog, Library, MusicError, Track

log = logging.getLogger(__name__)


class MusicService:
    def __init__(self, settings):
        self.settings = settings
        self.library = Library(settings.music_library_path)
        self.catalog = Catalog()
        self.downloader = MeTube(settings.music_metube_url, self.library)
        self.sessions = {}
        self.jobs = set()
        self.index_task = None
        self.closed = False

    def start_index(self):
        if self.closed or self.library.scanned or self.index_task is not None:
            return self.index_task

        async def scan():
            try:
                await self.library.scan()
            except Exception as exc:
                log.warning("music_index error=%s", type(exc).__name__)

        self.index_task = asyncio.create_task(scan(), name="music-library-index")
        return self.index_task

    @asynccontextmanager
    async def maintenance(self):
        if self.closed or len(self.jobs) >= 2:
            raise MusicError("Music maintenance is busy. Try again after the current job finishes.")
        task = asyncio.current_task()
        self.jobs.add(task)
        try:
            yield
        finally:
            self.jobs.discard(task)

    async def rescan(self):
        async with self.maintenance():
            return await self.library.scan()

    async def download(self, artist, title):
        if not 1 <= len(artist.strip()) <= 180 or not 1 <= len(title.strip()) <= 180:
            raise MusicError("Specify an artist and song title.")
        async with self.maintenance():
            await self.library.scan()
            track = Track(artist.strip(), title.strip())
            if existing := self.library.existing(track):
                if self.settings.music_downloads_enabled and self.settings.music_metube_url:
                    await self.downloader.cleanup_existing(existing)
                return existing
            return await self.prepare(track)

    def session(self, guild_id, playback, notify):
        if guild_id not in self.sessions:
            self.sessions[guild_id] = MusicSession(self, playback, notify)
        return self.sessions[guild_id]

    async def prepare(self, track):
        if existing := self.library.existing(track):
            return existing
        if not self.settings.music_downloads_enabled:
            raise MusicError("This recording is absent locally and music downloads are disabled.")
        if not self.settings.music_metube_url:
            raise MusicError("Set the MeTube URL before enabling music downloads.")
        if not track.video_id:
            track = await self.catalog.video(track)
        if existing := self.library.existing(track):
            return existing
        return await self.downloader.prepare(track)

    async def release(self, guild_id):
        if session := self.sessions.pop(guild_id, None):
            session.closed = True
            await session.stop()

    async def close(self):
        self.closed = True
        tasks = [t for t in self.jobs if t is not asyncio.current_task()]
        if self.index_task:
            tasks.append(self.index_task)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for guild in list(self.sessions):
            await self.release(guild)
        await self.downloader.close()
        await self.catalog.close()


class MusicSession:
    def __init__(self, service, playback, notify):
        self.service, self.playback, self.notify = service, playback, notify
        self.tracks = []
        self.position = 0
        self.current = None
        self.paused = False
        self.task = self.expansion = None
        self.prepared = {}
        self.audio_task = None
        self.error = ""
        self.query = self.mode = ""
        self.offset = 0
        self.navigation = asyncio.Event()
        self.revision = 0
        self.commands = asyncio.Lock()
        self.closed = False
        self.saved = None

    def status(self):
        playing = self.current
        return (
            f"Music: {'paused' if self.paused else 'active' if self.task and not self.task.done() else 'stopped'}\n"
            + (f"Now: {playing.artist} - {playing.title}\n" if playing else "")
            + f"Mode: {self.mode or 'none'}; playlist: {len(self.tracks)}; prepared/pending: {len(self.prepared)}\n"
            + (f"Last issue: {self.error}" if self.error else "")
            + (
                f"\nRetained playlist: {self.saved[0]}; use resume to play it again."
                if self.saved and not self.task
                else ""
            )
        )

    async def start(self, query, mode="auto"):
        if not 1 <= len(query) <= 180 or mode not in {"auto", "track", "artist", "radio"}:
            raise MusicError("Choose a short song, artist or genre and a supported mode.")
        await self.stop()
        if self.closed:
            raise MusicError("The voice session ended. Join again before requesting music.")
        self.query, self.mode = query, mode
        self.error = ""
        self.task = asyncio.create_task(self._run(), name="music-playlist")

    async def _notice(self, text):
        try:
            await self.notify(discord.utils.escape_mentions(text)[:1900])
        except Exception as exc:
            log.warning("music_notice error=%s", type(exc).__name__)

    async def _expand(self):
        try:
            additions = await self.service.catalog.discover(self.query, self.mode, self.offset)
            self.offset += 25
            for track in additions:
                if track.key not in {t.key for t in self.tracks} and len(self.tracks) < 50:
                    self.tracks.append(self.service.library.existing(track) or track)
        except Exception as exc:
            self.error = str(exc) if isinstance(exc, MusicError) else "Music catalog discovery failed."
            log.warning("music_discovery error=%s", type(exc).__name__)

    async def _run(self):
        try:
            if not self.tracks:
                await self.service.library.ensure()
                mode = self.mode
                if mode == "auto":
                    mode = (
                        "radio"
                        if self.query.casefold() in GENRES
                        else "artist"
                        if self.service.library.select(self.query, "artist")
                        else "track"
                    )
                self.mode = mode
                self.tracks = list(
                    {t.key: t for t in self.service.library.select(self.query, mode)}.values()
                )[:50]
                if mode == "track":
                    self.tracks = self.tracks[:1]
                    if not self.tracks and self.service.settings.music_downloads_enabled:
                        self.tracks = [await self.service.catalog.single(self.query)]
                elif not self.tracks and self.service.settings.music_downloads_enabled:
                    await self._expand()
                if not self.tracks:
                    raise MusicError(
                        self.error
                        or "No matching recordings were found. Try an artist, exact song or another genre."
                    )
                if mode == "radio":
                    random.shuffle(self.tracks)
                if (
                    mode in {"artist", "radio"}
                    and self.service.settings.music_downloads_enabled
                    and self.offset == 0
                    and len(self.tracks) < 50
                ):
                    self.expansion = asyncio.create_task(self._expand())
            failures = 0
            announced = False
            while self.tracks:
                self.position %= len(self.tracks)
                index = self.position
                revision = self.revision
                self.navigation.clear()
                self.current = None
                # Download at most two upcoming recordings. Retain only this
                # lookahead and current slot, never a growing task collection.
                wanted = {(index + i) % len(self.tracks) for i in range(min(3, len(self.tracks)))}
                for old in set(self.prepared) - wanted:
                    task = self.prepared.pop(old)
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                if index not in self.prepared:
                    self.prepared[index] = asyncio.create_task(self.service.prepare(self.tracks[index]))
                try:
                    navigation = asyncio.create_task(self.navigation.wait())
                    try:
                        await asyncio.wait(
                            {self.prepared[index], navigation}, return_when=asyncio.FIRST_COMPLETED
                        )
                        if revision != self.revision:
                            continue
                        self.current = await self.prepared[index]
                    finally:
                        navigation.cancel()
                        await asyncio.gather(navigation, return_exceptions=True)
                    self.tracks[index] = self.current
                    # Prioritize the first audible recording before allowing
                    # speculative downloads to occupy the shared MeTube slot.
                    for slot in sorted(wanted, key=lambda s: (s - index) % len(self.tracks)):
                        if slot not in self.prepared:
                            self.prepared[slot] = asyncio.create_task(self.service.prepare(self.tracks[slot]))
                    log.info(
                        "music_playback outcome=started mode=%s candidates=%s", self.mode, len(self.tracks)
                    )
                    if not announced:
                        await self._notice(f"Now playing: {self.current.artist} - {self.current.title}")
                        announced = True
                    self.audio_task = asyncio.create_task(self.playback.play_music(self.current.path))
                    try:
                        await self.audio_task
                        failures = 0
                    except asyncio.CancelledError:
                        if asyncio.current_task().cancelling():
                            raise
                    finally:
                        self.audio_task = None
                except Exception as exc:
                    failures += 1
                    self.error = (
                        str(exc)
                        if isinstance(exc, MusicError)
                        else "Recording playback or preparation failed. Check the music service logs."
                    )
                    log.warning("music_track error=%s", type(exc).__name__)
                    failed = self.prepared.pop(index, None)
                    if failed:
                        await asyncio.gather(failed, return_exceptions=True)
                    if failures >= min(5, len(self.tracks)):
                        raise MusicError(self.error) from exc
                if self.revision == revision:
                    self.position += 1
                if (
                    self.mode == "radio"
                    and self.offset < 200
                    and len(self.tracks) < 50
                    and (not self.expansion or self.expansion.done())
                    and self.service.settings.music_downloads_enabled
                ):
                    self.expansion = asyncio.create_task(self._expand())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = (
                str(exc)
                if isinstance(exc, MusicError)
                else "Music service unavailable. Check the music configuration and logs."
            )
            log.warning("music_session error=%s", type(exc).__name__)
            await self._notice(self.error)
        finally:
            await self._clear_jobs()
            self.current = None

    async def _clear_jobs(self):
        tasks = list(self.prepared.values()) + ([self.expansion] if self.expansion else [])
        self.prepared.clear()
        self.expansion = None
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def control(self, action):
        if self.closed:
            raise MusicError("The voice session ended.")
        if action == "stop":
            await self.stop(remember=True)
            return
        if (not self.task or self.task.done()) and action in {"resume", "restart_music"} and self.saved:
            self.query, self.mode, tracks, position, self.offset = self.saved
            self.tracks = list(tracks)
            self.position = 0 if action == "restart_music" else position
            self.paused = False
            self.playback.pause_music(False)
            self.task = asyncio.create_task(self._run(), name="music-playlist-restored")
            return
        if not self.task or self.task.done():
            raise MusicError("No music is queued. Request a song, artist or genre first.")
        if action in {"pause", "resume"}:
            self.paused = action == "pause"
            self.playback.pause_music(self.paused)
            return
        if action == "restart_music":
            self.position = 0
            self.paused = False
            self.playback.pause_music(False)
            action = "restart"
        if action in {"next", "previous", "restart"} and self.tracks:
            self.revision += 1
            self.navigation.set()
            self.position = (self.position + {"next": 1, "previous": -1, "restart": 0}[action]) % len(
                self.tracks
            )
            if self.audio_task:
                task = self.audio_task
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        elif action in {"next", "previous", "restart"}:
            raise MusicError("The playlist is still being prepared.")

    async def stop(self, *, remember=False):
        if remember and self.tracks:
            self.saved = (self.query, self.mode, tuple(self.tracks), self.position, self.offset)
        elif not remember:
            self.saved = None
        task, self.task = self.task, None
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._clear_jobs()
        self.tracks, self.current, self.position, self.offset = [], None, 0, 0
        self.paused = False
        self.playback.pause_music(False)
