"""Validated provider-owned routes for provider-neutral generation tasks."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib  # type: ignore[import-not-found]
else:
    import tomli as tomllib  # type: ignore[import-not-found]

GENERATION_CONFIG_VERSION = 1


@dataclass(frozen=True)
class GenerationTaskRoute:
    model: str
    reasoning_effort: str


@dataclass(frozen=True)
class ProviderTaskRoutes:
    provider: str
    tasks: Mapping[str, GenerationTaskRoute]

    def route(self, task: str) -> GenerationTaskRoute:
        try:
            return self.tasks[task]
        except KeyError as error:
            raise ValueError(
                f"generation provider {self.provider!r} has no route for task {task!r}"
            ) from error


def load_provider_task_routes(
    path: Path,
    *,
    provider: str,
    required_tasks: Sequence[str],
) -> ProviderTaskRoutes:
    """Load one provider's complete task routing snapshot from the tracked TOML."""

    provider = provider.strip()
    if not provider:
        raise ValueError("generation provider must not be blank")
    tasks = tuple(required_tasks)
    if not tasks or any(not task.strip() for task in tasks) or len(tasks) != len(set(tasks)):
        raise ValueError("required generation tasks must be unique nonblank names")

    try:
        with path.open("rb") as config_file:
            config: dict[str, Any] = tomllib.load(config_file)
    except FileNotFoundError as error:
        raise ValueError(f"generation config not found: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"generation config is invalid TOML: {path}") from error

    version = config.get("version")
    if type(version) is not int or version != GENERATION_CONFIG_VERSION:
        raise ValueError(
            f"generation config version must be {GENERATION_CONFIG_VERSION}, got {version!r}"
        )
    providers = config.get("providers")
    if not isinstance(providers, dict):
        raise ValueError("generation config providers must be a table")
    provider_config = providers.get(provider)
    if not isinstance(provider_config, dict):
        raise ValueError(f"generation config provider is missing or invalid: {provider}")
    raw_tasks = provider_config.get("tasks")
    if not isinstance(raw_tasks, dict):
        raise ValueError(f"generation provider {provider!r} tasks must be a table")

    routes: dict[str, GenerationTaskRoute] = {}
    for task in tasks:
        raw_route = raw_tasks.get(task)
        if not isinstance(raw_route, dict):
            raise ValueError(
                f"generation provider {provider!r} task route is missing or invalid: {task}"
            )
        model = raw_route.get("model")
        effort = raw_route.get("reasoning_effort")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"generation provider {provider!r} task model is invalid: {task}")
        if not isinstance(effort, str) or not effort.strip():
            raise ValueError(
                f"generation provider {provider!r} task reasoning_effort is invalid: {task}"
            )
        routes[task] = GenerationTaskRoute(
            model=model.strip(),
            reasoning_effort=effort.strip(),
        )
    return ProviderTaskRoutes(provider=provider, tasks=MappingProxyType(routes))
