import io

import ezdxf
import httpx
import pytest
from sqlalchemy import select

from app.api import app
from app.app_factory import create_app
from app.config import get_settings
from app.infra.db import SessionLocal
from app.infra.models import Job
from app.tasks.job import process_run
from tools.smoke_backend import drawing

pytestmark = [pytest.mark.anyio, pytest.mark.usefixtures("clean_db")]


async def request(method, path, user="alice", **kwargs):
    token = {"alice": "a" * 40, "bob": "b" * 40, "bad": "invalid"}.get(user)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        return await client.request(method, path, headers=headers, **kwargs)


async def upload(user="alice", **fields):
    return await request("POST", "/api/v1/jobs", user, files={"file": ("x.pdf", b"%PDF-1.4\n")}, data=fields)


async def test_all_endpoints_require_credentials(monkeypatch):
    job_id = (await upload()).json()["id"]
    for user in (None, "bad"):
        assert (await upload(user)).status_code == 401
        for method, suffix, kwargs in [
            ("GET", "", {}), ("GET", "/artifacts/original.pdf", {}),
            ("POST", "/rerun", {}), ("POST", "/drawing", {"json": drawing(46)}),
        ]:
            assert (await request(method, f"/api/v1/jobs/{job_id}{suffix}", user, **kwargs)).status_code == 401
    monkeypatch.setattr(get_settings(), "auth_tokens", {})
    assert (await upload()).status_code == 503
    assert (await request("GET", "/health", None)).status_code == 200


async def test_owner_and_history_are_enforced_for_every_file_and_mutation():
    alice = (await upload(channel="vk", conversation_id="same", event_id="event", user_id="bob")).json()
    bob = (await upload("bob", channel="vk", conversation_id="same", event_id="event", user_id="alice")).json()
    assert alice["id"] != bob["id"]
    assert (await upload(channel="vk", conversation_id="same", event_id="event")).json()["id"] == alice["id"]
    async with SessionLocal() as session:
        owners = dict((await session.execute(select(Job.id, Job.owner_id))).all())
        assert owners == {alice["id"]: "alice", bob["id"]: "bob"}
    # Complete both initial runs before submitting the manual one.
    await process_run()
    await process_run()
    assert (await request("POST", f"/api/v1/jobs/{alice['id']}/drawing", json=drawing(46))).status_code == 202
    await process_run()
    first_dxf = await request("GET", f"/api/v1/jobs/{alice['id']}/artifacts/result.dxf")
    assert first_dxf.status_code == 200
    assert (await request("POST", f"/api/v1/jobs/{alice['id']}/drawing", json=drawing(47))).status_code == 202
    await process_run()
    status = (await request("GET", f"/api/v1/jobs/{alice['id']}")).json()
    for run in status["runs"]:
        for name in run["artifacts"]:
            for selector in (f"run_number={run['number']}", f"run_id={run['id']}"):
                url = f"/api/v1/jobs/{alice['id']}/artifacts/{name}?{selector}"
                assert (await request("GET", url, "bob")).status_code == 404
                assert (await request("GET", url)).status_code == 200
    for owner, intruder, job in (("alice", "bob", alice), ("bob", "alice", bob)):
        base = f"/api/v1/jobs/{job['id']}"
        assert (await request("GET", base, intruder)).status_code == 404
        assert (await request("POST", base + "/rerun", intruder)).status_code == 404
        assert (await request("POST", base + "/drawing", intruder, json=drawing(48))).status_code == 404
        assert (await request("GET", base, owner)).status_code == 200
    # An old run id cannot be transplanted into another owned job's URL.
    url = f"/api/v1/jobs/{bob['id']}/artifacts/result.dxf?run_id={status['runs'][1]['id']}"
    assert (await request("GET", url, "bob")).status_code == 404
    old = await request("GET", f"/api/v1/jobs/{alice['id']}/artifacts/result.dxf?run_number=2")
    assert old.content == first_dxf.content
    assert ezdxf.read(io.StringIO(old.text)).modelspace().query("DIMENSION")[0].get_measurement() == 46


async def test_historical_unowned_jobs_cannot_be_claimed_with_metadata():
    job = (await upload(conversation_id="legacy")).json()
    async with SessionLocal.begin() as session:
        row = await session.get(Job, job["id"])
        row.owner_id = None
    for user in ("alice", "bob"):
        base = f"/api/v1/jobs/{job['id']}"
        assert (await request("GET", base, user, params={"user_id": "alice", "conversation_id": "legacy"})).status_code == 404
        assert (await request("GET", base + "/artifacts/original.pdf", user)).status_code == 404
        assert (await request("POST", base + "/rerun", user)).status_code == 404
        assert (await request("POST", base + "/drawing", user, json=drawing(46))).status_code == 404


async def test_frontend_preflight_only_allows_configured_origin(monkeypatch):
    monkeypatch.setattr(get_settings(), "cors_origins", ["http://localhost:5173"])
    async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app()), base_url="http://test") as client:
        for origin, status in (("http://localhost:5173", 200), ("https://untrusted.example", 400)):
            response = await client.options("/api/v1/jobs", headers={
                "Origin": origin, "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            })
            assert response.status_code == status
            assert response.headers.get("access-control-allow-origin") == (origin if status == 200 else None)


@pytest.mark.parametrize("move_anchor", [False, True])
async def test_geometry_edit_cannot_hide_stale_free_dimension_by_dropping_provenance(move_anchor):
    job = (await upload()).json()
    base = f"/api/v1/jobs/{job['id']}"
    await process_run()
    graph = drawing(46)
    graph['points'][1]['refs'] = []
    assert (await request('POST', base + '/drawing', json=graph)).status_code == 202
    await process_run()
    assert (await request('GET', base)).json()['state'] == 'succeeded'
    if move_anchor:
        graph['primitives'][0]['vertices'][0][1] = 1
    graph['primitives'][0]['vertices'][1][0] = 60
    graph['primitives'][0]['vertices'][2][0] = 47
    graph.pop('provenance', None)
    assert (await request('POST', base + '/drawing', json=graph)).status_code == 202
    await process_run()
    assert (await request('GET', base)).json()['state'] == 'needs_review'
    report = (await request('GET', base + '/artifacts/report.json')).json()
    assert any(issue['code'] == 'dimension_recalculation_required' for issue in report['issues'])
    result = await request('GET', base + '/artifacts/result.dxf')
    assert not ezdxf.read(io.StringIO(result.text)).modelspace().query('DIMENSION')
