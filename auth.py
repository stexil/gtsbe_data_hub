# auth.py
import streamlit as st
from typing import Dict, Any

PROVIDER = "google"

# Optional: put your logo in .streamlit (or any public URL)
APP_LOGO_URL = st.secrets.get("APP_LOGO_URL", "")
APP_NAME = st.secrets.get("APP_NAME", "GTSBE Analytics Hub")
SUPPORT_EMAIL = st.secrets.get("SUPPORT_EMAIL", "gtsbe.cpcchair@gmail.com")



def _render_signin_card(denied_msg: str | None = None) -> None:
    
    # Center the content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        
        
        # Title and subtitle
        st.markdown(f'<h1 class="auth-title">Welcome to {APP_NAME}</h1>', unsafe_allow_html=True)
        st.markdown('<p class="auth-subtitle">Sign in with your Google account to continue</p>', unsafe_allow_html=True)
        
        # Error message if access denied
        if denied_msg:
            st.error(f"🚫 {denied_msg}")
        
        # Google Sign In button
        if st.button("🚀 Sign in with Google", type="primary", use_container_width=True):
            st.session_state["_auth_click"] = True
        
        # Divider
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Footer text
        st.markdown(
            f'<div class="footer-text">By continuing, you agree to abide by GTSBE community guidelines.<br>Need access? Contact <b>{SUPPORT_EMAIL}</b></div>',
            unsafe_allow_html=True,
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Trigger login after layout is drawn
    if st.session_state.get("_auth_click"):
        st.session_state.pop("_auth_click", None)
        st.login(PROVIDER)

# ---- Public API (unchanged) --------------------------------------------------
def auth_gate() -> Dict[str, Any]:
    """
    Blocks until:
      - user is logged in (via st.login(PROVIDER)), and
      - user is in ALLOWED_EMAILS (server-side allowlist)
    Returns {email, name, sub}
    """
    # Not logged in -> show signin card
    if not getattr(st.user, "is_logged_in", False):
        _render_signin_card()
        st.stop()

    # Server-side allowlist check
    allowed_raw = st.secrets.get("ALLOWED_EMAILS", "")
    allowed = [
        e.strip().lower()
        for e in (allowed_raw if isinstance(allowed_raw, list) else str(allowed_raw).split(","))
        if e.strip()
    ]
    email = (getattr(st.user, "email", "") or "").lower()

    if allowed and email not in allowed:
        # Show denied state with logout option
        _render_signin_card(denied_msg="Access denied: your email is not on the allowlist.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Sign Out", type="secondary", use_container_width=True):
                st.logout()
        st.stop()

    # Success - return user info
    return {
        "email": getattr(st.user, "email", ""),
        "name": getattr(st.user, "name", ""),
        "sub": getattr(st.user, "sub", getattr(st.user, "email", "")),
    }

def logout() -> None:
    st.logout()