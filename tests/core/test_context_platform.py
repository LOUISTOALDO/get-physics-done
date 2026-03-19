"""Regression tests for runtime platform detection in gpd.core.context."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import patch

import pytest

import gpd.core.context as context_module
from gpd.adapters import get_adapter, list_runtimes
from gpd.adapters.install_utils import GPD_INSTALL_DIR_NAME
from gpd.adapters.runtime_catalog import get_runtime_descriptor
from gpd.core.constants import ENV_GPD_ACTIVE_RUNTIME

_RUNTIME_NAMES = tuple(list_runtimes())
_SUPPORTED_RUNTIME_DESCRIPTORS = tuple(get_runtime_descriptor(runtime) for runtime in _RUNTIME_NAMES)
_RUNTIME_ENV_KEYS = {
    ENV_GPD_ACTIVE_RUNTIME,
    *(
        env_var
        for descriptor in _SUPPORTED_RUNTIME_DESCRIPTORS
        for env_var in (
            *descriptor.activation_env_vars,
            descriptor.global_config.env_var,
            descriptor.global_config.env_dir_var,
            descriptor.global_config.env_file_var,
            "XDG_CONFIG_HOME" if descriptor.global_config.strategy == "xdg_app" else None,
        )
        if env_var
    ),
}


def _runtime_pair() -> tuple[str, str]:
    if len(_RUNTIME_NAMES) < 2:
        raise AssertionError("Expected at least two supported runtimes")
    return _RUNTIME_NAMES[0], _RUNTIME_NAMES[1]


def _clear_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove runtime detection env vars so each test controls the signal."""
    for key in _RUNTIME_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize(
    ("env_var", "expected"),
    [(descriptor.activation_env_vars[0], descriptor.runtime_name) for descriptor in _SUPPORTED_RUNTIME_DESCRIPTORS],
)
def test_init_context_uses_active_runtime_signal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, env_var: str, expected: str
) -> None:
    with monkeypatch.context() as runtime_env:
        _clear_runtime_env(runtime_env)
        runtime_env.setenv(env_var, "active")

        module = importlib.reload(context_module)
        ctx = module.init_new_project(tmp_path)
        assert ctx["platform"] == expected

    importlib.reload(context_module)


def test_init_context_uses_runtime_detect_directory_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = _RUNTIME_NAMES[0]
    adapter = get_adapter(runtime)

    with monkeypatch.context() as runtime_env:
        _clear_runtime_env(runtime_env)
        (tmp_path / adapter.local_config_dir_name).mkdir()

        with patch("gpd.hooks.runtime_detect.Path.home", return_value=tmp_path), \
             patch("gpd.hooks.runtime_detect.Path.cwd", return_value=tmp_path):
            module = importlib.reload(context_module)
            ctx = module.init_new_project(tmp_path)
            assert ctx["platform"] == runtime

    importlib.reload(context_module)


def test_init_context_prefers_explicit_gpd_runtime_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    preferred_runtime, secondary_runtime = _runtime_pair()
    preferred_adapter = get_adapter(preferred_runtime)
    secondary_adapter = get_adapter(secondary_runtime)

    with monkeypatch.context() as runtime_env:
        _clear_runtime_env(runtime_env)
        runtime_env.setenv(ENV_GPD_ACTIVE_RUNTIME, preferred_runtime)
        (tmp_path / secondary_adapter.local_config_dir_name / GPD_INSTALL_DIR_NAME).mkdir(parents=True)
        (tmp_path / preferred_adapter.local_config_dir_name / GPD_INSTALL_DIR_NAME).mkdir(parents=True)

        with patch("gpd.hooks.runtime_detect.Path.home", return_value=tmp_path), \
             patch("gpd.hooks.runtime_detect.Path.cwd", return_value=tmp_path):
            module = importlib.reload(context_module)
            ctx = module.init_progress(tmp_path)
            assert ctx["platform"] == preferred_runtime

    importlib.reload(context_module)


def test_resolve_model_delegates_runtime_specific_lookup_to_config_helper(tmp_path: Path) -> None:
    runtime = _RUNTIME_NAMES[0]
    calls: dict[str, object] = {}

    def _fake_resolve_model(project_dir: Path, agent_name: str, runtime: str | None = None) -> str | None:
        calls["project_dir"] = project_dir
        calls["agent_name"] = agent_name
        calls["runtime"] = runtime
        return "delegated-model"

    with patch.object(context_module, "_resolve_model_canonical", side_effect=_fake_resolve_model):
        result = context_module._resolve_model(
            tmp_path,
            "gpd-planner",
            {"model_profile": "review", "model_overrides": {runtime: {"tier-1": "do-not-read-directly"}}},
            runtime=runtime,
        )

    assert result == "delegated-model"
    assert calls == {"project_dir": tmp_path, "agent_name": "gpd-planner", "runtime": runtime}


def test_resolve_model_falls_back_to_platform_detection_when_runtime_detector_returns_unknown(tmp_path: Path) -> None:
    runtime = _RUNTIME_NAMES[0]
    calls: dict[str, object] = {}

    def _fake_resolve_model(project_dir: Path, agent_name: str, runtime: str | None = None) -> str | None:
        calls["project_dir"] = project_dir
        calls["agent_name"] = agent_name
        calls["runtime"] = runtime
        return "fallback-model"

    with (
        patch("gpd.hooks.runtime_detect.detect_runtime_for_gpd_use", return_value="unknown"),
        patch.object(context_module, "_detect_platform", return_value=runtime),
        patch.object(context_module, "_resolve_model_canonical", side_effect=_fake_resolve_model),
    ):
        result = context_module._resolve_model(tmp_path, "gpd-executor")

    assert result == "fallback-model"
    assert calls == {"project_dir": tmp_path, "agent_name": "gpd-executor", "runtime": runtime}
