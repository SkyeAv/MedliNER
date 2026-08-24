"""Podman-hosted Label Studio server lifecycle and minimal REST client.

Label Studio stays out of the MedliNER Python environment: the ``medliner label-studio``
command launches the
stock ``heartexlabs/label-studio`` image with podman, waits for health, ensures a project
with ``configs/label_studio_ner.xml``, and imports candidate tasks. Annotation happens in
the browser; ``export_project`` downloads the result over the API. Group sessions can bind
the port mapping beyond loopback (``publish_host``) and pre-create annotator accounts.
Everything here is stdlib-only so no SDK dependency is added.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any

DEFAULT_CONTAINER = "medliner-label-studio"
DEFAULT_IMAGE = "docker.io/heartexlabs/label-studio:latest"
DEFAULT_PORT = 9030
DEFAULT_PROJECT_TITLE = "MedliNER"
ONBOARDING_PROJECT_TITLE = "Onboarding"
WARMUP_PROJECT_TITLE = "MedliNER — Warm-up"
HEALTH_TIMEOUT_S = 300.0


class LabelStudioServerError(RuntimeError):
    """Raised when the container or the Label Studio API cannot be driven."""


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Single subprocess entry point so tests can fake the podman CLI."""
    return subprocess.run(args, capture_output=True, text=True, check=False)


def _urlopen(request: urllib.request.Request, timeout: float = 30.0):
    """Single HTTP entry point so tests can fake the network."""
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 — localhost service


def container_state(name: str = DEFAULT_CONTAINER) -> str | None:
    """None when absent, else the podman-reported state (``running``, ``exited``, ...)."""
    result = _run(["podman", "container", "exists", name])
    if result.returncode != 0:
        return None
    inspected = _run(["podman", "inspect", "--format", "{{.State.Status}}", name])
    if inspected.returncode != 0:
        raise LabelStudioServerError(f"podman inspect failed for {name}: {inspected.stderr.strip()}")
    return inspected.stdout.strip()


def ensure_container(
    *,
    name: str = DEFAULT_CONTAINER,
    image: str = DEFAULT_IMAGE,
    port: int = DEFAULT_PORT,
    data_dir: str | Path,
    username: str,
    password: str,
    publish_host: str = "127.0.0.1",
) -> str:
    """Start the Label Studio container idempotently; returns the container id.

    An existing non-running container is replaced rather than started: port, volume, and
    credential env cannot change on ``podman start``, and the mounted data directory keeps
    the Label Studio database across the replacement. ``publish_host`` is the address the
    port mapping binds (``127.0.0.1`` keeps the server private; ``0.0.0.0`` exposes it on
    every interface so annotators on the same network can reach it).
    """
    state = container_state(name)
    if state == "running":
        return name
    if state is not None:
        removed = _run(["podman", "rm", "-f", name])
        if removed.returncode != 0:
            raise LabelStudioServerError(f"could not remove stale container {name}: {removed.stderr.strip()}")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    # The label-studio image runs as UID 1001. A host directory bind-mounted into a rootless
    # container appears root-owned there, so hand it to the container user through the user
    # namespace (a plain chown to 1001 when podman runs rootful).
    chown = _run(["podman", "unshare", "chown", "-R", "1001:0", str(data_dir)])
    if chown.returncode != 0:
        raise LabelStudioServerError(f"could not chown {data_dir} for the container user: {chown.stderr.strip()}")
    result = _run(
        [
            "podman",
            "run",
            "-d",
            "--name",
            name,
            "-p",
            f"{publish_host}:{port}:8080",
            "-v",
            f"{data_dir}:/label-studio/data:Z",
            "-e",
            f"LABEL_STUDIO_USERNAME={username}",
            "-e",
            f"LABEL_STUDIO_PASSWORD={password}",
            image,
        ]
    )
    if result.returncode != 0:
        raise LabelStudioServerError(f"podman run failed: {result.stderr.strip()}")
    return name


def stop_container(name: str = DEFAULT_CONTAINER) -> bool:
    """Remove the container if present; returns True when one was removed."""
    if container_state(name) is None:
        return False
    result = _run(["podman", "rm", "-f", name])
    if result.returncode != 0:
        raise LabelStudioServerError(f"podman rm failed for {name}: {result.stderr.strip()}")
    return True


def wait_healthy(base_url: str, *, timeout_s: float = HEALTH_TIMEOUT_S, interval_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with _urlopen(urllib.request.Request(f"{base_url}/health"), timeout=5.0) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(interval_s)
    raise LabelStudioServerError(f"Label Studio at {base_url} did not become healthy within {timeout_s}s")


class LabelStudioClient:
    """Authenticated subset of the Label Studio API used by the pipeline.

    Label Studio ≥1.23 disables legacy token authentication by default, so the default path
    is Django session auth: log in through the browser form, then send every API request
    with the session cookie and the CSRF header. An explicit `token` (the JWT access tokens
    from Account & Settings) is sent as a Bearer credential instead.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._opener = None
        self._csrf: str | None = None
        if self._token is None:
            if not username or not password:
                raise LabelStudioServerError(
                    "set MEDLINER_LABEL_STUDIO_TOKEN or MEDLINER_LABEL_STUDIO_USERNAME/_PASSWORD"
                )
            self._login(username, password)

    def _login(self, username: str, password: str) -> None:
        jar = CookieJar()
        self._opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        # Django's login form is CSRF-protected; fetch it first so the cookie exists.
        self._opener.open(urllib.request.Request(f"{self.base_url}/user/login/"), timeout=30.0)
        csrf = next((cookie.value for cookie in jar if cookie.name == "csrftoken"), None)
        if csrf is None:
            raise LabelStudioServerError("Label Studio login page did not set a CSRF cookie")
        form = urllib.parse.urlencode(
            {"email": username, "username": username, "password": password, "csrfmiddlewaretoken": csrf}
        ).encode()
        request = urllib.request.Request(
            f"{self.base_url}/user/login/",
            data=form,
            headers={"Referer": f"{self.base_url}/user/login/"},
        )
        self._opener.open(request, timeout=30.0)
        if not any(cookie.name == "sessionid" for cookie in jar):
            raise LabelStudioServerError(f"Label Studio login failed for {username!r}")
        self._csrf = csrf

    def api(self, method: str, path: str, payload: Any | None = None) -> Any:
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(f"{self.base_url}{path}", data=data, method=method)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            if self._opener is not None:
                # Session-authenticated mutating requests must echo the CSRF token.
                if self._csrf is not None:
                    request.add_header("X-CSRFToken", self._csrf)
                request.add_header("Referer", f"{self.base_url}/")
                with self._opener.open(request, timeout=30.0) as response:
                    body = response.read().decode()
            else:
                request.add_header("Authorization", f"Bearer {self._token}")
                with _urlopen(request) as response:
                    body = response.read().decode()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise LabelStudioServerError(f"Label Studio API {method} {path} failed: {exc.code} {detail}") from exc
        return json.loads(body) if body.strip() else {}

    def ensure_project(self, title: str, label_config: str) -> int:
        """Return the project id, creating the project with the labeling config when absent."""
        existing = self.find_project(title)
        if existing is not None:
            return existing
        created = self.api("POST", "/api/projects", {"title": title, "label_config": label_config})
        return int(created["id"])

    def enable_prelabeling(self, project_id: int, model_version: str) -> None:
        """Show imported predictions to annotators and pre-fill their draft from them.

        Without ``show_collab_predictions`` Label Studio stores the predictions but never puts
        them in front of the annotator, so the whole point of pre-labeling is lost. ``model_version``
        selects which prediction set to pre-fill from when a task carries more than one.
        """
        self.api(
            "PATCH",
            f"/api/projects/{project_id}",
            {"show_collab_predictions": True, "model_version": model_version},
        )

    def find_project(self, title: str) -> int | None:
        """Return the project id with exactly this title, or None when absent."""
        listing = self.api("GET", "/api/projects")
        projects = listing.get("results", listing) if isinstance(listing, dict) else listing
        for project in projects:
            if project.get("title") == title:
                return int(project["id"])
        return None

    def list_users(self) -> list[dict[str, Any]]:
        """All known Label Studio accounts (paginated or bare list)."""
        listing = self.api("GET", "/api/users")
        return listing.get("results", listing) if isinstance(listing, dict) else listing

    def create_user(self, *, username: str, password: str, email: str | None = None) -> dict[str, Any]:
        """Create an annotator account via ``POST /api/users``; returns the created user."""
        payload: dict[str, Any] = {"username": username, "password": password}
        if email:
            payload["email"] = email
        return self.api("POST", "/api/users", payload)

    def project_task_count(self, project_id: int) -> int:
        project = self.api("GET", f"/api/projects/{project_id}")
        return int(project.get("task_number", 0))

    def import_tasks(self, project_id: int, tasks: list[dict[str, Any]]) -> int:
        result = self.api("POST", f"/api/projects/{project_id}/import", tasks)
        return int(result.get("task_count", len(tasks)))

    def export_annotations(self, project_id: int) -> list[dict[str, Any]]:
        """Download every task with its annotations as JSON (Label Studio's JSON export)."""
        export = self.api("GET", f"/api/projects/{project_id}/export?exportType=JSON")
        if not isinstance(export, list):
            raise LabelStudioServerError(f"unexpected export payload from project {project_id}: expected a list")
        return export


def _base_url(port: int) -> str:
    # 127.0.0.1, not localhost: rootless podman's forwarder may not listen on ::1, and
    # localhost can resolve there first.
    return f"http://127.0.0.1:{port}"


def provision(
    *,
    import_file: str | Path,
    label_config_path: str | Path,
    name: str = DEFAULT_CONTAINER,
    image: str = DEFAULT_IMAGE,
    port: int = DEFAULT_PORT,
    data_dir: str | Path,
    username: str,
    password: str,
    token: str | None = None,
    project_title: str = DEFAULT_PROJECT_TITLE,
    reimport: bool = False,
    publish_host: str = "127.0.0.1",
    annotators: list[tuple[str, str]] | None = None,
    prelabel_model_version: str | None = None,
) -> dict[str, Any]:
    """Container + project + task import in one call; returns provisioning metadata.

    ``annotators`` are ``(username, password)`` pairs ensured as additional accounts via
    ``POST /api/users``; existing usernames are skipped so provisioning stays idempotent.
    ``prelabel_model_version`` turns on prediction pre-fill for the project when the import file
    carries model suggestions.
    """
    container = ensure_container(
        name=name,
        image=image,
        port=port,
        data_dir=data_dir,
        username=username,
        password=password,
        publish_host=publish_host,
    )
    base_url = _base_url(port)
    wait_healthy(base_url)
    client = LabelStudioClient(base_url, token=token, username=username, password=password)
    project_id = client.ensure_project(project_title, Path(label_config_path).read_text(encoding="utf-8"))
    if prelabel_model_version:
        client.enable_prelabeling(project_id, prelabel_model_version)
    tasks = json.loads(Path(import_file).read_text(encoding="utf-8"))
    existing = client.project_task_count(project_id)
    imported = 0
    if existing and not reimport:
        imported = existing
    else:
        imported = client.import_tasks(project_id, tasks)
    annotators_created = 0
    if annotators:
        known = {user.get("username") for user in client.list_users()}
        for annotator_username, annotator_password in annotators:
            if annotator_username in known:
                continue
            client.create_user(username=annotator_username, password=annotator_password)
            annotators_created += 1
    usernames = sorted(str(user.get("username")) for user in client.list_users() if user.get("username"))
    return {
        "url": base_url,
        "container": container,
        "project_id": project_id,
        "tasks_in_project": imported,
        "existing_tasks": existing,
        "reimported": bool(existing and reimport),
        "publish_host": publish_host,
        "annotators_created": annotators_created,
        "prelabeled": bool(prelabel_model_version),
        "usernames": usernames,
    }


def export_project(
    *,
    output_path: str | Path,
    port: int = DEFAULT_PORT,
    username: str,
    password: str,
    token: str | None = None,
    project_title: str = DEFAULT_PROJECT_TITLE,
) -> dict[str, Any]:
    """Download the project's annotations to ``output_path``; returns export metadata.

    The output lands directly at ``$MEDLINER_LABEL_STUDIO_EXPORT`` by default from the CLI,
    so the manual browser-export step is replaced end to end.
    """
    base_url = _base_url(port)
    wait_healthy(base_url)
    client = LabelStudioClient(base_url, token=token, username=username, password=password)
    project_id = client.find_project(project_title)
    if project_id is None:
        raise LabelStudioServerError(f"no Label Studio project titled {project_title!r} at {base_url}")
    tasks = client.export_annotations(project_id)
    annotated = sum(1 for task in tasks if task.get("annotations"))
    output = Path(output_path)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError as exc:
        raise LabelStudioServerError(f"cannot write {output}: {exc}") from exc
    return {
        "url": base_url,
        "project_id": project_id,
        "tasks_exported": len(tasks),
        "tasks_annotated": annotated,
        "output": output,
    }


__all__ = [
    "DEFAULT_CONTAINER",
    "DEFAULT_IMAGE",
    "DEFAULT_PORT",
    "DEFAULT_PROJECT_TITLE",
    "ONBOARDING_PROJECT_TITLE",
    "WARMUP_PROJECT_TITLE",
    "LabelStudioClient",
    "LabelStudioServerError",
    "container_state",
    "ensure_container",
    "export_project",
    "provision",
    "stop_container",
    "wait_healthy",
]
