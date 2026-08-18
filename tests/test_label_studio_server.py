from __future__ import annotations

import io
import json
import subprocess
import urllib.error
from http.cookiejar import Cookie, CookieJar
from pathlib import Path

import pytest

from medliner import label_studio_server as server


class FakePodman:
    """Minimal in-memory podman CLI: one container slot keyed by name."""

    def __init__(self) -> None:
        self.containers: dict[str, dict] = {}
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        action = args[1:]
        name = args[-1]
        if action[:2] == ["container", "exists"]:
            code = 0 if name in self.containers else 1
            return subprocess.CompletedProcess(args, code, "", "")
        if action[:1] == ["inspect"]:
            if name not in self.containers:
                return subprocess.CompletedProcess(args, 1, "", "no such container")
            return subprocess.CompletedProcess(args, 0, self.containers[name]["state"] + "\n", "")
        if action[:1] == ["run"]:
            self.containers[args[args.index("--name") + 1]] = {"state": "running"}
            return subprocess.CompletedProcess(args, 0, "container-id\n", "")
        if action[:1] == ["rm"]:
            self.containers.pop(name, None)
            return subprocess.CompletedProcess(args, 0, "", "")
        if action[:1] == ["unshare"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(f"unexpected podman call: {args}")


class FakeResponse:
    def __init__(self, payload, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc) -> None:
        return None


class FakeLabelStudio:
    """In-memory Label Studio HTTP API; fakes label_studio_server._urlopen."""

    def __init__(self, *, healthy: bool = True, projects: list[dict] | None = None) -> None:
        self.healthy = healthy
        self.projects = {p["id"]: {**p, "tasks": list(p.get("tasks", []))} for p in projects or []}
        self.requests: list[tuple[str, str]] = []
        self._next_id = max(self.projects, default=0) + 1

    def __call__(self, request, timeout: float = 30.0) -> FakeResponse:
        url = request.full_url
        path = "/" + url.split("://", 1)[1].split("/", 1)[1] if "/" in url.split("://", 1)[1] else "/"
        method = request.get_method()
        self.requests.append((method, path))
        if path == "/health":
            if not self.healthy:
                raise urllib.error.URLError("connection refused")
            return FakeResponse({"status": "UP"})
        assert request.get_header("Authorization") == "Token test-token"
        if path == "/api/projects" and method == "GET":
            return FakeResponse({"results": [{"id": p["id"], "title": p["title"]} for p in self.projects.values()]})
        if path == "/api/projects" and method == "POST":
            payload = json.loads(request.data.decode())
            project = {"id": self._next_id, "title": payload["title"], "label_config": payload["label_config"]}
            self.projects[self._next_id] = {**project, "tasks": []}
            self._next_id += 1
            return FakeResponse(project)
        project_id = int(path.split("/")[3])
        project = self.projects[project_id]
        if method == "GET":
            return FakeResponse({"id": project_id, "task_number": len(project["tasks"])})
        if method == "POST" and path.endswith("/import"):
            project["tasks"] = json.loads(request.data.decode())
            return FakeResponse({"task_count": len(project["tasks"])})
        raise AssertionError(f"unexpected API call: {method} {path}")


@pytest.fixture
def podman(monkeypatch) -> FakePodman:
    fake = FakePodman()
    monkeypatch.setattr(server, "_run", fake)
    return fake


def _install_api(monkeypatch, fake: FakeLabelStudio) -> FakeLabelStudio:
    monkeypatch.setattr(server, "_urlopen", fake)
    return fake


def test_ensure_container_creates_when_absent(podman, tmp_path):
    server.ensure_container(
        name="medliner-label-studio", image="img", port=8080, data_dir=tmp_path, username="u", password="p"
    )
    (run_call,) = [call for call in podman.calls if call[1] == "run"]
    assert "-p" in run_call and "8080:8080" in run_call
    volume = run_call[run_call.index("-v") + 1]
    assert volume.endswith(":/label-studio/data:Z")
    assert "LABEL_STUDIO_USERNAME=u" in run_call and "LABEL_STUDIO_PASSWORD=p" in run_call
    assert podman.containers["medliner-label-studio"]["state"] == "running"
    # The container runs as UID 1001; the bind-mounted data dir is handed to that user.
    (chown_call,) = [call for call in podman.calls if call[1] == "unshare"]
    assert chown_call[2:5] == ["chown", "-R", "1001:0"]


def test_ensure_container_reuses_a_running_container(podman, tmp_path):
    podman.containers["medliner-label-studio"] = {"state": "running"}
    server.ensure_container(
        name="medliner-label-studio", image="img", port=8080, data_dir=tmp_path, username="u", password="p"
    )
    assert not [call for call in podman.calls if call[1] == "run"]


def test_ensure_container_replaces_a_stopped_container(podman, tmp_path):
    podman.containers["medliner-label-studio"] = {"state": "exited"}
    server.ensure_container(
        name="medliner-label-studio", image="img", port=8080, data_dir=tmp_path, username="u", password="p"
    )
    actions = [call[1] for call in podman.calls]
    assert "rm" in actions and "run" in actions


def test_stop_container_reports_whether_one_was_removed(podman):
    assert server.stop_container() is False
    podman.containers["medliner-label-studio"] = {"state": "running"}
    assert server.stop_container() is True
    assert podman.containers == {}


def test_wait_healthy_times_out(monkeypatch):
    def refused(*args, **kwargs):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(server, "_urlopen", refused)
    with pytest.raises(server.LabelStudioServerError, match="did not become healthy"):
        server.wait_healthy("http://localhost:8080", timeout_s=0.05, interval_s=0.01)


def test_client_requires_token_or_credentials():
    with pytest.raises(server.LabelStudioServerError, match="MEDLINER_LABEL_STUDIO_TOKEN"):
        server.LabelStudioClient("http://localhost:8080")


def test_session_login_fetches_a_legacy_token(monkeypatch):
    jar = CookieJar()

    def csrf_cookie() -> Cookie:
        return Cookie(
            version=0,
            name="csrftoken",
            value="csrf-value",
            port=None,
            port_specified=False,
            domain="localhost",
            domain_specified=False,
            domain_initial_dot=False,
            path="/",
            path_specified=True,
            secure=False,
            expires=None,
            discard=True,
            comment=None,
            comment_url=None,
            rest={},
            rfc2109=False,
        )

    class FakeOpener:
        def __init__(self) -> None:
            self.posts: list[bytes] = []

        def open(self, request, timeout: float = 30.0) -> FakeResponse:
            if request.data is not None:
                self.posts.append(request.data)
                return FakeResponse({})
            if request.full_url.endswith("/api/current-user/token"):
                return FakeResponse({"token": "session-token"})
            jar.set_cookie(csrf_cookie())
            return FakeResponse({})

    opener = FakeOpener()
    monkeypatch.setattr(server.LabelStudioClient, "_opener", lambda self: (jar, opener))
    client = server.LabelStudioClient("http://localhost:8080", username="u", password="p")
    assert client._token == "session-token"
    assert b"email=u" in opener.posts[0] and b"csrfmiddlewaretoken=csrf-value" in opener.posts[0]


def test_provision_creates_project_and_imports_tasks(monkeypatch, podman, tmp_path):
    fake = _install_api(monkeypatch, FakeLabelStudio())
    import_file = tmp_path / "import.json"
    import_file.write_text(json.dumps([{"id": "t1", "data": {"text": "x", "task": "indication"}}]), encoding="utf-8")
    config = tmp_path / "config.xml"
    config.write_text("<View/>", encoding="utf-8")

    result = server.provision(
        import_file=import_file,
        label_config_path=config,
        data_dir=tmp_path / "data",
        username="u",
        password="p",
        token="test-token",
    )
    assert result["url"] == f"http://localhost:{server.DEFAULT_PORT}"
    assert result["tasks_in_project"] == 1
    assert result["reimported"] is False
    project = fake.projects[result["project_id"]]
    assert project["label_config"] == "<View/>"
    assert len(project["tasks"]) == 1


def test_provision_reuses_project_and_skips_existing_tasks(monkeypatch, podman, tmp_path):
    existing = {"id": 7, "title": server.DEFAULT_PROJECT_TITLE, "tasks": [{"id": "t0", "data": {}}]}
    fake = _install_api(monkeypatch, FakeLabelStudio(projects=[existing]))
    import_file = tmp_path / "import.json"
    import_file.write_text(json.dumps([{"id": "t1", "data": {"text": "x", "task": "indication"}}]), encoding="utf-8")
    config = tmp_path / "config.xml"
    config.write_text("<View/>", encoding="utf-8")

    result = server.provision(
        import_file=import_file,
        label_config_path=config,
        data_dir=tmp_path / "data",
        username="u",
        password="p",
        token="test-token",
    )
    assert result["project_id"] == 7
    assert result["tasks_in_project"] == 1  # the pre-existing task, not a duplicate import
    assert not [call for call in fake.requests if call[0] == "POST" and call[1].endswith("/import")]

    reimported = server.provision(
        import_file=import_file,
        label_config_path=config,
        data_dir=tmp_path / "data",
        username="u",
        password="p",
        token="test-token",
        reimport=True,
    )
    assert reimported["reimported"] is True
    assert fake.projects[7]["tasks"][0]["id"] == "t1"


def test_provision_surfaces_api_errors(monkeypatch, podman, tmp_path):
    fake = _install_api(monkeypatch, FakeLabelStudio())

    def failing(request, timeout: float = 30.0):
        if request.full_url.endswith("/api/projects"):
            raise urllib.error.HTTPError(request.full_url, 500, "boom", {}, io.BytesIO(b"detail"))
        return fake(request, timeout)

    monkeypatch.setattr(server, "_urlopen", failing)
    import_file = tmp_path / "import.json"
    import_file.write_text("[]", encoding="utf-8")
    config = tmp_path / "config.xml"
    config.write_text("<View/>", encoding="utf-8")
    with pytest.raises(server.LabelStudioServerError, match="500"):
        server.provision(
            import_file=import_file,
            label_config_path=config,
            data_dir=tmp_path / "data",
            username="u",
            password="p",
            token="test-token",
        )


def test_label_config_fixture_is_real_xml():
    config = Path(__file__).resolve().parents[1] / "configs" / "label_studio_ner.xml"
    assert config.exists()
