import os
import time
from typing import Callable


def temporary_message(message_fn: Callable, message: str, seconds: int = 1) -> None:
    """Temporarily shows a success message.

    message_fn can be, e.g. st.success, col[2].info, grid.warning, position.error, ...
    """
    message = message_fn(message)
    time.sleep(seconds)
    message.empty()  # type: ignore
