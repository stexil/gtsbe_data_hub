# auth.py
import streamlit as st

PROVIDER = "auth0"  # must match [auth.auth0] block name in secrets

def _login_button():
    st.markdown("## 🔐 Please sign in")
    # Use a named provider to match your secrets
    st.button("Sign in", on_click=lambda: st.login(PROVIDER))

def auth_gate():
    # If auth isn't configured, st.user has no attrs → treat as not logged in
    is_logged_in = getattr(st.user, "is_logged_in", False)
    if not is_logged_in:
        _login_button()
        st.stop()

    # Allowlist (comma-separated string or list)
    raw = st.secrets.get("ALLOWED_EMAILS", "")
    allowed = [e.strip().lower() for e in (raw if isinstance(raw, list) else raw.split(",")) if e.strip()]
    email = getattr(st.user, "email", "").lower()

    if allowed and email not in allowed:
        st.error("Access denied: Your email is not authorized.")
        st.button("Log out", on_click=st.logout)
        st.stop()

    return {
        "email": getattr(st.user, "email", ""),
        "name": getattr(st.user, "name", ""),
        "sub": getattr(st.user, "sub", getattr(st.user, "email", "")),
    }

def logout():
    st.logout()
