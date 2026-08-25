from __future__ import annotations

import io
import json
import subprocess
import urllib.error
from http.cookiejar import Cookie, CookieJar
from pathlib import Path
from typing import Any

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

    def __init__(
        self, *, healthy: bool = True, projects: list[dict] | None = None, users: list[str] | None = None
    ) -> None:
        self.healthy = healthy
        self.projects = {p["id"]: {**p, "tasks": list(p.get("tasks", []))} for p in projects or []}
        self.users = list(users or [])
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
        assert request.get_header("Authorization") == "Bearer test-token"
        if path == "/api/projects" and method == "GET":
            return FakeResponse({"results": [{"id": p["id"], "title": p["title"]} for p in self.projects.values()]})
        if path == "/api/projects" and method == "POST":
            payload = json.loads(request.data.decode())
            project = {"id": self._next_id, "title": payload["title"], "label_config": payload["label_config"]}
            self.projects[self._next_id] = {**project, "tasks": []}
            self._next_id += 1
            return FakeResponse(project)
        if path == "/api/users" and method == "GET":
            return FakeResponse({"results": [{"id": index, "username": name} for index, name in enumerate(self.users)]})
        if path == "/api/users" and method == "POST":
            payload = json.loads(request.data.decode())
            self.users.append(payload["username"])
            return FakeResponse({"id": len(self.users), "username": payload["username"]})
        if method == "GET" and "/export" in path:
            project = self.projects[int(path.split("/")[3])]
            return FakeResponse(project["tasks"])
        project_id = int(path.split("/")[3])
        project = self.projects[project_id]
        if method == "GET":
            return FakeResponse({"id": project_id, "task_number": len(project["tasks"])})
        if method == "POST" and path.endswith("/import"):
            project["tasks"] = json.loads(request.data.decode())
            return FakeResponse({"task_count": len(project["tasks"])})
        if method == "PATCH":
            project.update(json.loads(request.data.decode()))
            return FakeResponse({"id": project_id})
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
    assert run_call[run_call.index("-p") + 1] == "127.0.0.1:8080:8080"
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


def test_session_login_drives_the_api_with_cookies_and_csrf(monkeypatch):
    def make_cookie(name: str, value: str) -> Cookie:
        return Cookie(
            version=0,
            name=name,
            value=value,
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
        def __init__(self, jar: CookieJar) -> None:
            self.jar = jar
            self.posts: list[bytes] = []
            self.api_headers: list[dict] = []

        def open(self, request, timeout: float = 30.0) -> FakeResponse:
            if request.full_url.endswith("/user/login/"):
                if request.data is not None:
                    self.posts.append(request.data)
                    self.jar.set_cookie(make_cookie("sessionid", "session-value"))
                else:
                    self.jar.set_cookie(make_cookie("csrftoken", "csrf-value"))
                return FakeResponse({})
            self.api_headers.append(
                {"csrf": request.get_header("X-csrftoken"), "auth": request.get_header("Authorization")}
            )
            return FakeResponse({"results": []})

    def fake_build_opener(*handlers):
        return FakeOpener(handlers[0].cookiejar)

    monkeypatch.setattr("urllib.request.build_opener", fake_build_opener)
    client = server.LabelStudioClient("http://localhost:9030", username="u", password="p")
    opener = client._opener
    assert b"email=u" in opener.posts[0] and b"csrfmiddlewaretoken=csrf-value" in opener.posts[0]

    client.api("GET", "/api/projects")
    # Session path: CSRF header, no Authorization bearer.
    assert opener.api_headers == [{"csrf": "csrf-value", "auth": None}]


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
    assert result["url"] == f"http://127.0.0.1:{server.DEFAULT_PORT}"
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


def test_label_config_labels_carry_hotkeys():
    """The single condition label keeps a number-key hotkey so live annotation stays fast."""
    import xml.etree.ElementTree as ET

    config = Path(__file__).resolve().parents[1] / "configs" / "label_studio_ner.xml"
    labels = {node.get("value"): node.get("hotkey") for node in ET.parse(config).iter("Label")}
    assert labels == {"DiseaseOrPhenotypicFeature": "1"}


def test_ensure_container_publish_host_binds_wider(podman, tmp_path):
    server.ensure_container(
        name="medliner-label-studio",
        image="img",
        port=8080,
        data_dir=tmp_path,
        username="u",
        password="p",
        publish_host="0.0.0.0",
    )
    (run_call,) = [call for call in podman.calls if call[1] == "run"]
    assert run_call[run_call.index("-p") + 1] == "0.0.0.0:8080:8080"


def test_provision_seeds_missing_annotators_idempotently(monkeypatch, podman, tmp_path):
    _install_api(monkeypatch, FakeLabelStudio(users=["medliner@localhost"]))
    import_file = tmp_path / "import.json"
    import_file.write_text(json.dumps([{"id": "t1", "data": {"text": "x", "task": "indication"}}]), encoding="utf-8")
    config = tmp_path / "config.xml"
    config.write_text("<View/>", encoding="utf-8")

    common: dict[str, Any] = dict(
        import_file=import_file,
        label_config_path=config,
        data_dir=tmp_path / "data",
        username="u",
        password="p",
        token="test-token",
    )
    result = server.provision(annotators=[("alice", "pw-a"), ("medliner@localhost", "x")], **common)
    assert result["annotators_created"] == 1  # the existing admin is skipped
    again = server.provision(annotators=[("alice", "pw-a")], **common)
    assert again["annotators_created"] == 0  # already present on the second run


def test_export_project_writes_file_and_counts_annotations(monkeypatch, podman, tmp_path):
    annotated = {
        "id": 1,
        "title": server.DEFAULT_PROJECT_TITLE,
        "tasks": [
            {"id": "t1", "annotations": [{"result": []}]},
            {"id": "t2", "annotations": []},
        ],
    }
    _install_api(monkeypatch, FakeLabelStudio(projects=[annotated]))

    output = tmp_path / "reviewed" / "export.json"
    result = server.export_project(output_path=output, username="u", password="p", token="test-token")
    assert result["project_id"] == 1
    assert result["tasks_exported"] == 2
    assert result["tasks_annotated"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))[0]["id"] == "t1"


def test_export_project_requires_the_project_to_exist(monkeypatch, podman, tmp_path):
    _install_api(monkeypatch, FakeLabelStudio())
    with pytest.raises(server.LabelStudioServerError, match="no Label Studio project titled"):
        server.export_project(output_path=tmp_path / "x.json", username="u", password="p", token="test-token")


def test_provision_turns_on_prediction_prefill_when_asked(monkeypatch, podman, tmp_path):
    # Without show_collab_predictions Label Studio stores the predictions and never shows them,
    # so the annotator would still draw every span by hand.
    fake = _install_api(monkeypatch, FakeLabelStudio())
    import_file = tmp_path / "import.json"
    import_file.write_text(json.dumps([{"id": "t1", "data": {"text": "x"}, "predictions": []}]), encoding="utf-8")
    config = tmp_path / "config.xml"
    config.write_text("<View/>", encoding="utf-8")

    result = server.provision(
        import_file=import_file,
        label_config_path=config,
        data_dir=tmp_path / "data",
        username="u",
        password="p",
        token="test-token",
        prelabel_model_version="gliner_large-v2.5@0.35",
    )
    project = fake.projects[result["project_id"]]
    assert project["show_collab_predictions"] is True
    assert project["model_version"] == "gliner_large-v2.5@0.35"
    assert result["prelabeled"] is True


def test_provision_leaves_prefill_alone_for_a_plain_text_project(monkeypatch, podman, tmp_path):
    fake = _install_api(monkeypatch, FakeLabelStudio())
    import_file = tmp_path / "import.json"
    import_file.write_text(json.dumps([{"id": "t1", "data": {"text": "x"}}]), encoding="utf-8")
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
    assert result["prelabeled"] is False
    assert not any(method == "PATCH" for method, _ in fake.requests)
