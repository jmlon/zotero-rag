"""Putting text on the system clipboard from a terminal app.

Textual's `copy_to_clipboard` sends the OSC 52 escape sequence, which works over
SSH but which many terminals ignore, GNOME Terminal and other VTE-based ones among
them. So a local clipboard tool is tried first, and OSC 52 is still sent by the
caller in any case. Inside an SSH session a local tool would fill the *remote*
machine's clipboard, so there OSC 52 is the only route.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)

TOOL_TIMEOUT_S = 2.0


def _candidates() -> list[list[str]]:
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return []
    if sys.platform == "darwin":
        return [["pbcopy"]]
    commands: list[list[str]] = []
    if os.environ.get("WAYLAND_DISPLAY"):
        commands.append(["wl-copy"])
    if os.environ.get("DISPLAY"):
        commands += [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]
    return commands


def copy_with_system_tool(text: str) -> str | None:
    """Copy with the first clipboard tool that works; returns its name, or `None`.

    The tools' output goes to /dev/null. xclip and wl-copy stay running to serve
    the selection, and a pipe left open to them would make this call hang.
    """
    for command in _candidates():
        if shutil.which(command[0]) is None:
            continue
        try:
            subprocess.run(
                command,
                input=text.encode("utf-8"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=TOOL_TIMEOUT_S,
                check=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("clipboard tool %s failed: %s", command[0], exc)
            continue
        return command[0]
    return None
