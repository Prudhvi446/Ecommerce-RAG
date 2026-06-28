"""
DLL Helper for Windows Store Python installations.
Dynamically locates and adds the local-packages Library/bin directory to the DLL search path.
This resolves missing dependency issues for compiled packages like PyTorch and MKL.
"""

import glob
import os
import sys

if sys.platform == "win32":
    # List to hold possible DLL search directories
    dll_paths = []

    # 1. Standard python prefix Library bin (e.g. for conda or standard installs)
    dll_paths.append(os.path.join(sys.prefix, "Library", "bin"))

    # 2. General user site Library/bin
    user_base = os.environ.get("APPDATA")
    if user_base:
        dll_paths.append(os.path.join(user_base, "Python", "Library", "bin"))

    # 3. Windows Store Python package paths
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        store_pattern = os.path.join(
            local_appdata,
            "Packages",
            "PythonSoftwareFoundation.Python.*",
            "LocalCache",
            "local-packages",
            "Library",
            "bin",
        )
        dll_paths.extend(glob.glob(store_pattern))

    # Add all unique existing directories to the DLL search path
    added_paths = []
    for path in set(dll_paths):
        if os.path.isdir(path):
            try:
                os.add_dll_directory(path)
                added_paths.append(path)
            except Exception:
                pass

    if added_paths:
        # Subtle print so developers know DLLs were configured
        print(f"[DLL Loader] Configured Windows DLL search path: {added_paths}")
