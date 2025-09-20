import streamlit as st

PROVIDER = "google"

def auth_gate():
    if not getattr(st.user, "is_logged_in", False):
        st.markdown("## 🔐 Please sign in")
        st.button("Sign in with Google",
                  key="btn_signin_google",
                  on_click=lambda: st.login(PROVIDER))
        st.stop()

    # allowlist case (if you show a logout here, give it a different key)
    allowed_raw = st.secrets.get("ALLOWED_EMAILS", "")
    allowed = [e.strip().lower() for e in (allowed_raw if isinstance(allowed_raw, list) else allowed_raw.split(",")) if e.strip()]
    email = getattr(st.user, "email", "").lower()
    if allowed and email not in allowed:
        st.error("Access denied: Your email is not authorized.")
        st.button("Log out", key="btn_logout_denied", on_click=st.logout)
        st.stop()

    return {
        "email": getattr(st.user, "email", ""),
        "name": getattr(st.user, "name", ""),
        "sub": getattr(st.user, "sub", getattr(st.user, "email", "")),
    }

def logout():
    st.logout()
