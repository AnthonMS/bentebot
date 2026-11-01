"""A completed import authorizes cleanup of only the verified MeTube source."""

import asyncio
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from bentebot.music_download import MeTube, audio_signature
from bentebot.music_library import Library, MusicError, Track


@pytest.fixture
def transfer(tmp_path, monkeypatch):
    library = Library(str(tmp_path))
    downloader = MeTube("http://metube.test", library)
    monkeypatch.setattr("mutagen.mp3.MP3", lambda _: SimpleNamespace(info=SimpleNamespace(length=20)))
    monkeypatch.setattr(downloader, "cleanup_mode", AsyncMock())
    track = Track("Artist", "Song", video_id="abcdefghijk")
    row = {
        "id": track.video_id,
        "url": "https://www.youtube.com/watch?v=" + track.video_id,
        "filename": "Artist - Song.mp3",
        "folder": "",
        "format": "mp3",
        "download_type": "audio",
        "status": "finished",
    }
    return SimpleNamespace(library=library, downloader=downloader, track=track, row=row, data=b"a" * 2048)


async def install_http(case, handler):
    await case.downloader.http.aclose()
    case.downloader.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def imported(case):
    staging = case.library.root / "incoming.mp3"
    staging.write_bytes(case.data)
    track = await case.library.import_file(staging, case.track)
    staging.unlink()
    return track


@pytest.mark.parametrize("mode", ["ask", "true", "false", None])
async def test_cleanup_reads_public_configuration_and_handles_ping(monkeypatch, tmp_path, mode):
    sends = []
    packets = iter(
        [
            "0{}",
            "2",
            '42["other", "ignored"]',
            "42" + json.dumps(["configuration", json.dumps({"DELETE_FILE_ON_TRASHCAN": mode})]),
        ]
    )

    class Socket:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def recv(self):
            return next(packets)

        async def send(self, packet):
            sends.append(packet)

    def connect(url, **kwargs):
        assert url == "ws://metube.test/prefix/socket.io/?EIO=4&transport=websocket"
        return Socket()

    monkeypatch.setattr("bentebot.music_download.connect", connect)
    downloader = MeTube("http://metube.test/prefix", Library(str(tmp_path)))
    try:
        if mode in {"ask", "true"}:
            await downloader.cleanup_mode()
        else:
            with pytest.raises(MusicError, match="DELETE_FILE_ON_TRASHCAN=ask"):
                await downloader.cleanup_mode()
        assert sends == ["40", "3"]
    finally:
        await downloader.close()


@pytest.mark.parametrize(
    "failure", ["different_audio", "changed_source", "changed_job", "shared_filename", "chapters", "overlap"]
)
async def test_cleanup_never_deletes_unverified_or_shared_source(transfer, failure):
    c = transfer
    track = await imported(c)
    digest = hashlib.sha256(c.data).hexdigest()
    rows = [dict(c.row)]
    if failure == "different_audio":
        from pathlib import Path

        await asyncio.to_thread(Path(track.path).write_bytes, b"b" * 2048)
    elif failure == "changed_source":
        c.data = b"b" * 2048
    elif failure == "changed_job":
        rows[0]["id"] = "differentid"
    elif failure == "shared_filename":
        rows.append({**c.row, "id": "differentid"})
    elif failure == "chapters":
        rows[0]["chapter_files"] = ["chapter.mp3"]
    elif failure == "overlap":
        from pathlib import Path

        rows[0]["filename"] = Path(track.path).relative_to(c.library.root).as_posix()
        c.row = dict(rows[0])

    def handler(request):
        assert request.method == "GET"
        if request.url.path == "/history":
            return httpx.Response(200, json={"done": rows})
        return httpx.Response(200, content=c.data)

    await install_http(c, handler)
    try:
        with pytest.raises(MusicError, match="kept"):
            await c.downloader.cleanup_import(c.row, track, digest)
    finally:
        await c.downloader.close()


async def test_explicit_download_cleanup_reuses_tagged_library_and_leaves_unrelated_jobs(transfer):
    c = transfer
    track = await imported(c)
    other = {
        **c.row,
        "id": "unrelated",
        "url": "https://www.youtube.com/watch?v=zyxwvutsrqp",
        "filename": "Other.mp3",
    }
    rows = [c.row, other]
    source_present = True

    def handler(request):
        nonlocal source_present
        if request.url.path == "/history":
            return httpx.Response(200, json={"done": rows})
        if request.url.path == "/delete":
            assert json.loads(request.content) == {
                "ids": [c.row["url"]],
                "where": "done",
                "delete_files": True,
            }
            # Match MeTube's actual queue contract, including its success
            # response for unknown keys. Video IDs must not delete anything.
            for key in json.loads(request.content)["ids"]:
                if key == c.row["url"]:
                    rows.remove(c.row)
                    source_present = False
            return httpx.Response(200, json={"status": "ok"})
        assert request.url.path == "/audio_download/Artist - Song.mp3"
        return httpx.Response(200 if source_present else 404, content=c.data)

    await install_http(c, handler)
    try:
        await c.downloader.cleanup_existing(track)
        assert rows == [other] and not source_present
        await c.downloader.cleanup_existing(track)
        assert c.library.existing(c.track) == track
    finally:
        await c.downloader.close()


@pytest.mark.parametrize("failure", ["import", "delete_rejected", "job_kept", "file_kept"])
async def test_transfer_failure_never_claims_success_or_removes_unimported_source(
    transfer, monkeypatch, failure
):
    c = transfer
    staging = c.library.root / "incoming.mp3"
    staging.write_bytes(c.data)
    deletes = []
    rows = [c.row]
    if failure == "import":
        monkeypatch.setattr(c.library, "import_file", AsyncMock(side_effect=MusicError("Invalid import")))

    def handler(request):
        if request.url.path == "/history":
            return httpx.Response(200, json={"done": rows})
        if request.url.path == "/delete":
            deletes.append(request)
            assert c.library.existing(c.track) is not None
            if failure == "file_kept":
                rows.clear()
            return httpx.Response(200, json={"status": "error" if failure == "delete_rejected" else "ok"})
        return httpx.Response(200, content=c.data)

    await install_http(c, handler)
    try:
        expected = {"job_kept": "completed job", "file_kept": "file permissions"}.get(failure)
        with pytest.raises(MusicError, match=expected):
            await c.downloader.finish_import(staging, c.track, c.row, hashlib.sha256(c.data).hexdigest())
        assert bool(deletes) == (failure != "import")
    finally:
        await c.downloader.close()


async def test_meTube_unknown_video_id_is_a_successful_no_op(transfer):
    c = transfer
    rows = {c.row["url"]: c.row}

    def handler(request):
        assert request.url.path == "/delete"
        for key in json.loads(request.content)["ids"]:
            rows.pop(key, None)
        return httpx.Response(200, json={"status": "ok"})

    await install_http(c, handler)
    try:
        response = await c.downloader.http.post(
            c.downloader.url + "/delete",
            json={"ids": [c.row["id"]], "where": "done", "delete_files": True},
        )
        assert response.json() == {"status": "ok"} and list(rows) == [c.row["url"]]
    finally:
        await c.downloader.close()


async def test_cancelling_after_import_waits_for_cleanup(transfer, monkeypatch):
    c = transfer
    track = await imported(c)
    entered, release = asyncio.Event(), asyncio.Event()

    async def cleanup(*args):
        entered.set()
        await release.wait()

    monkeypatch.setattr(c.downloader, "cleanup_import", cleanup)

    def handler(request):
        return (
            httpx.Response(200, json={"done": [c.row]})
            if request.url.path == "/history"
            else httpx.Response(200, content=c.data)
        )

    await install_http(c, handler)
    task = asyncio.create_task(c.downloader.cleanup_existing(track))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await c.downloader.close()


def test_audio_comparison_ignores_id3_changes_only(transfer):
    import io

    from mutagen.id3 import ID3, TIT2

    c = transfer
    stream = io.BytesIO(c.data)
    tags = ID3()
    tags.add(TIT2(encoding=3, text="Library title"))
    tags.save(stream)
    assert audio_signature(c.data) == audio_signature(stream.getvalue())
    assert audio_signature(c.data) != audio_signature(b"b" * 2048)
