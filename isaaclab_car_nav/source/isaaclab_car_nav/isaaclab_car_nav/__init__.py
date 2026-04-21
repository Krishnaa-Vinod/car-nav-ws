# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

def register_tasks() -> None:
	"""Import task modules to register Gym environments."""
	from . import tasks  # noqa: F401


def register_ui_extensions() -> None:
	"""Import UI extension modules."""
	from . import ui_extension_example  # noqa: F401


__all__ = ["register_tasks", "register_ui_extensions"]
