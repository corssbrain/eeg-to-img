import os
import platform
import re


def clear_screen():
    """Clear the console screen.

    Works on Windows, macOS, and Linux.
    """
    cmd = 'cls' if platform.system() == 'Windows' else 'clear'
    os.system(cmd)


def extract_id_from_string(s):
    """Extract trailing integer ID from a string.

    Returns the integer if found, otherwise None.
    """
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None
