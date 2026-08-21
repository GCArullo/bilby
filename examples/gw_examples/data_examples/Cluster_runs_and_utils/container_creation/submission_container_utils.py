from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def add_container_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_image_file: Path,
) -> None:
    parser.add_argument(
        "--container",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the Bilby container for Condor jobs (default). Pass "
            "--no-container to use the existing node environment."
        ),
    )
    parser.add_argument(
        "--container-image",
        default=None,
        help=(
            "Container image path or URL. By default this is read from "
            f"the current Git branch entry in {default_image_file}, written "
            "by `make publish` in the container_creation directory."
        ),
    )


def environment_variables_for_container(
    environment_variables: dict,
    container_image: str | None,
) -> dict:
    """Let the container supply its packaged LAL data path."""
    environment_variables = environment_variables.copy()
    if container_image is not None:
        environment_variables.pop("LAL_DATA_PATH", None)
    return environment_variables


def resolve_container_image(
    *,
    use_container: bool,
    container_image: str | None,
    default_image_file: Path,
) -> str | None:
    if not use_container:
        return None

    if container_image is None:
        if not default_image_file.is_file():
            raise FileNotFoundError(
                f"Missing container image registry: {default_image_file}. "
                "Run `make publish` in the container_creation directory, pass "
                "--container-image, or pass --no-container."
            )
        try:
            branch = subprocess.run(
                [
                    "git",
                    "-C",
                    str(default_image_file.parent),
                    "branch",
                    "--show-current",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except subprocess.CalledProcessError as exc:
            raise ValueError("Unable to determine the current Git branch") from exc
        if not branch:
            raise ValueError("Cannot select a container image from a detached HEAD")

        images = json.loads(default_image_file.read_text(encoding="utf-8"))
        try:
            container_image = images[branch]
        except KeyError as exc:
            available = ", ".join(sorted(images))
            raise ValueError(
                f"No container image configured for Git branch {branch!r} in "
                f"{default_image_file}. Available branches: {available}"
            ) from exc

    container_image = container_image.strip()
    if not container_image:
        raise ValueError("Container image must not be empty")
    if "://" not in container_image:
        return str(Path(container_image).expanduser().resolve())
    return container_image
