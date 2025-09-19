import streamlit as st

def auth_gate():
    """Super simple Google login"""
    if not st.user.is_logged_in:
        st.markdown("## 🔐 Please sign in")
        st.button("Sign in with Google", on_click=st.login)
        st.stop()
    
    # Check email allowlist
    allowed_emails = st.secrets.get("ALLOWED_EMAILS", "").split(",")
    if st.user.email not in [email.strip() for email in allowed_emails]:
        st.error("Access denied: Your email is not authorized.")
        st.button("Log out", on_click=st.logout)
        st.stop()
    
    return {
        'email': st.user.email,
        'name': st.user.name,
        'sub': st.user.email
    }

def logout():
    st.logout()
