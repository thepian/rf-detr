# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for optional dependency declarations in pyproject.toml."""

import pathlib

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10 fallback
    import tomli as tomllib


class TestOptionalDependencies:
    """Validate selected extras constraints in pyproject.toml."""

    @staticmethod
    def read_loggers_extra() -> list[str]:
        """Return the loggers optional-dependency list from pyproject.toml."""
        root = pathlib.Path(__file__).parent.parent.parent
        pyproject = tomllib.loads((root / "pyproject.toml").read_text())
        loggers = pyproject["project"]["optional-dependencies"].get("loggers")
        assert loggers, "loggers extra not found in [project.optional-dependencies]"
        return loggers

    @staticmethod
    def has_upper_bound_below_4(requirement: Requirement) -> bool:
        """Return True if the requirement specifier caps the version below 4.0.0."""
        max_allowed_version = Version("4.0.0")
        for spec in requirement.specifier:
            spec_version = Version(spec.version)
            if spec.operator == "<" and spec_version <= max_allowed_version:
                return True
            if spec.operator == "<=" and spec_version < max_allowed_version:
                return True
        return False

    def test_loggers_extra_does_not_pin_protobuf_below_4(self):
        """loggers extra must not pin protobuf <4 — would block wandb 0.26+ which requires protobuf>4.21."""
        requirements = [Requirement(dep) for dep in self.read_loggers_extra()]
        protobuf_requirements = [req for req in requirements if req.name == "protobuf"]
        assert not any(self.has_upper_bound_below_4(req) for req in protobuf_requirements), (
            "loggers extra must not pin protobuf below 4.0.0; doing so makes [loggers] + wandb>=0.26 unsatisfiable. "
            "Modern tensorboard (>=2.13) works with protobuf 4-7."
        )

    @pytest.mark.parametrize(
        "dep_str,expected",
        [
            pytest.param("protobuf>=3.20.0,<4.0.0", True, id="strict-upper-bound"),
            pytest.param("protobuf>=3.20.0", False, id="lower-bound-only"),
            pytest.param("protobuf>=3.20.0,<5.0.0", False, id="too-permissive-upper"),
            pytest.param("protobuf<=3.9.0", True, id="le-bound-below-4"),
        ],
    )
    def test_has_upper_bound_below_4(self, dep_str: str, expected: bool) -> None:
        """has_upper_bound_below_4 correctly identifies specifiers bounded below 4.0.0."""
        assert self.has_upper_bound_below_4(Requirement(dep_str)) == expected
