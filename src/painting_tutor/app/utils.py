import os
import time
from typing import Callable

import dotenv
import streamlit as st

dotenv.load_dotenv()


def temporary_message(message_fn: Callable, message: str, seconds: int = 1) -> None:
    """Temporarily shows a success message.

    message_fn can be, e.g. st.success, col[2].info, grid.warning, position.error, ...
    """
    message = message_fn(message)
    time.sleep(seconds)
    message.empty()


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == os.environ["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
            temporary_message(st.success, "🎉 Password correct", seconds=2)
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        temporary_message(st.error, "😭 Password incorrect", seconds=2)
        return False
    else:
        # Password correct.
        return True
