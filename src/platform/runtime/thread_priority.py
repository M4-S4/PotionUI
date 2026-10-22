import enum
import logging
import os
import sys
import threading
from functools import lru_cache

logger = logging.getLogger(__name__)


class ThreadPriority(enum.IntEnum):
    NORMAL = 0
    BELOW_NORMAL = 1
    BACKGROUND = 2


_LINUX_NICE = {ThreadPriority.BELOW_NORMAL: 10, ThreadPriority.BACKGROUND: 19}
_LINUX_IO_LEVEL = {ThreadPriority.BELOW_NORMAL: 6, ThreadPriority.BACKGROUND: 7}

_WINDOWS_BELOW_NORMAL = -1
_WINDOWS_BACKGROUND_BEGIN = 0x00010000


def _platform_name() -> str:
    return sys.platform


@lru_cache(maxsize=1)
def _windows_kernel32():
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentThread.restype = wintypes.HANDLE
    kernel32.GetCurrentThread.argtypes = ()
    kernel32.SetThreadPriority.restype = wintypes.BOOL
    kernel32.SetThreadPriority.argtypes = (wintypes.HANDLE, ctypes.c_int)
    return kernel32


def _apply_linux(priority: ThreadPriority) -> bool:
    thread_id = threading.get_native_id()
    os.setpriority(os.PRIO_PROCESS, thread_id, _LINUX_NICE[priority])
    try:
        import psutil

        psutil.Process(thread_id).ionice(psutil.IOPRIO_CLASS_BE, value=_LINUX_IO_LEVEL[priority])
    except Exception:
        logger.debug("Could not lower the IO priority of thread %s", thread_id, exc_info=True)
    return True


def _apply_windows(priority: ThreadPriority) -> bool:
    kernel32 = _windows_kernel32()
    handle = kernel32.GetCurrentThread()
    value = _WINDOWS_BACKGROUND_BEGIN if priority == ThreadPriority.BACKGROUND else _WINDOWS_BELOW_NORMAL
    return bool(kernel32.SetThreadPriority(handle, value))


def set_current_thread_priority(priority: ThreadPriority) -> bool:
    if priority == ThreadPriority.NORMAL:
        return False
    if threading.current_thread() is threading.main_thread():
        return False
    platform = _platform_name()
    try:
        if platform.startswith("linux"):
            return _apply_linux(priority)
        if platform.startswith("win"):
            return _apply_windows(priority)
    except Exception:
        logger.debug("Could not change the priority of the current thread", exc_info=True)
    return False
