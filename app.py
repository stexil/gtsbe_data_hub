"""
Prop Points Hub — Streamlit app (Auth0 login + server-side allowlist)

How to run locally:
1) pip install -r requirements.txt
2) Create .streamlit/secrets.toml (Auth0, ALLOWED_EMAILS, GOOGLE_SERVICE_ACCOUNT, SHEETS)
3) streamlit run app.py

In production (Render):
- Set env vars matching secrets (AUTH0__..., ALLOWED_EMAILS, GOOGLE_SERVICE_ACCOUNT__..., SHEETS__...)
- Point hub.gtsbe.org CNAME to your Render hostname and add the custom domain in Render
"""

from __future__ import annotations
import io
import json
import traceback
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from auth import auth_gate, logout
import streamlit as st



st.set_page_config(page_title="GTSBE Analytics Hub", layout="wide")
user = auth_gate()



# ---------- Google auth helpers (Service Account) ----------
SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"
DRIVE_RO_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"  # if you ever need write

def get_google_creds() -> Credentials:
    sa = dict(st.secrets["GOOGLE_SERVICE_ACCOUNT"])

    # Normalize private_key: convert literal "\n" into real newlines if needed
    pk = sa.get("private_key", "")
    if "\\n" in pk and "\n" not in pk.split("\\n")[0]:  # crude but effective
        sa["private_key"] = pk.replace("\\n", "\n")

    # Optional sanity checks (won't leak contents)
    if not sa["private_key"].startswith("-----BEGIN PRIVATE KEY-----"):
        raise RuntimeError("Service account private_key is not a valid PEM (missing BEGIN header).")
    if not sa["private_key"].strip().endswith("END PRIVATE KEY-----"):
        raise RuntimeError("Service account private_key is not a valid PEM (missing END footer).")

    scopes = [SHEETS_SCOPE, DRIVE_RO_SCOPE]
    return Credentials.from_service_account_info(sa, scopes=scopes)

def _find_email_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        cl = c.strip().lower()
        if "email" in cl:
            return c
    return None

def go(page_name: str):
    st.session_state.page = page_name

def get_excluded_gtids(key: str = "PROP_EXCLUDED_GTIDS") -> set[str]:
    val = st.secrets.get(key)
    if val is None:
        return set()
    if isinstance(val, list):
        vals = [str(x).strip() for x in val if str(x).strip()]
    else:
        vals = [s.strip() for s in str(val).split(",") if s.strip()]
    return set(vals)

def find_email_col(df: pd.DataFrame) -> str | None:
    candidates = {"email", "email address", "gt email", "gtmail"}
    lowmap = {c.lower(): c for c in df.columns}
    for c in lowmap:
        if c.strip().lower() in candidates or "email" in c.strip().lower():
            return lowmap[c]
    return None


def get_gspread_client(creds: Credentials) -> gspread.Client:
    return gspread.authorize(creds)

def open_spreadsheet(gc: gspread.Client, ref: str):
    """Open by URL or by key."""
    return gc.open_by_url(ref) if str(ref).startswith("http") else gc.open_by_key(ref)

def _read_csv_or_list(val) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    # CSV string
    return [s.strip() for s in str(val).split(",") if s.strip()]

def get_allowlist(key: str) -> set[str]:
    """
    Reads an allowlist from secrets. Accepts either:
      - Top-level: PROP_ALLOWED_EMAILS = "a@x,b@y"  (or a TOML array)
      - Nested:   [access] PROP_ALLOWED_EMAILS = [...]
    Returns all entries lowercased.
    """
    import streamlit as st
    val = st.secrets.get(key)
    if val is None:
        val = st.secrets.get("access", {}).get(key) or st.secrets.get("ACCESS", {}).get(key)
    items = _read_csv_or_list(val)
    return {s.lower() for s in items}

def email_is_allowed(email: str, allowlist: set[str]) -> bool:
    """
    Rules:
      - '*' allows everyone
      - exact email match
      - domain rule: entries that start with '@' allow any address ending with that domain
    """
    if not allowlist:
        return False
    em = (email or "").lower().strip()
    if "*" in allowlist:
        return True
    if em in allowlist:
        return True
    for entry in allowlist:
        if entry.startswith("@") and em.endswith(entry):
            return True
    return False


# ---------- Auth gate (blocks until logged in & allowed) ----------
user = auth_gate()  # -> {"email","name","sub"}
st.sidebar.markdown(f"**Signed in as:** {user['email']}")
st.sidebar.button("Logout", on_click=logout)

# ---------- Global nav ----------
if "page" not in st.session_state:
    st.session_state.page = "menu"

def go(page_name: str):
    st.session_state.page = page_name
    #st.rerun()

# ---------- MENU ----------
if st.session_state.page == "menu":
    st.markdown("## 🎉 Welcome to the GTSBE Prop Points Hub")
    st.caption("Choose a tool to continue.")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("### Prop Points Calculator")
            st.write("Calculate prop points for each **unique GTID and name**.")
            st.button("Open", key="tile_calc", type="primary", on_click=lambda: go("prop_points"))
    with col2:
        with st.container(border=True):
            st.markdown("### Event Point Values")
            st.write("Adjust event point worths and save a new configuration.")
            st.button("Open", key="tile_events", on_click=lambda: go("event_points"))

    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            st.markdown("### Activity Report Attendance Generator")
            st.write("Generate report rows for a date + time window (values untouched).")
            st.button("Open", key="tile_blank1", on_click=lambda: go("blank1"))
    with col4:
        with st.container(border=True):
            st.markdown("### Member Attendance Lookup")
            st.write("Find Attendance History for any member that has signed in")
            st.button("Open", key="tile_blank2", on_click=lambda: go("blank2"))

# ---------- PROP POINTS ----------
elif st.session_state.page == "prop_points":
    st.markdown("### Prop Points Calculator")

    # === PAGE-LEVEL ACCESS CONTROL ===
    prop_allow = get_allowlist("PROP_ALLOWED_EMAILS")  # reads top-level or [access] block
    user_email = (user.get("email") or "").lower()
    if not email_is_allowed(user_email, prop_allow):
        st.error("Access denied: this tool is restricted to the Prop Points team.")
        st.caption("If you believe this is an error, ask an admin to add your email to PROP_ALLOWED_EMAILS in secrets.")
        st.button("⬅︎ Back to menu", key="back_prop_denied", on_click=lambda: go("menu"))
        st.stop()
    # =================================

    # Helpers local to this page
    def pick(df: pd.DataFrame, *cands):
        lowmap = {c.lower(): c for c in df.columns}
        for c in cands:
            if c in df.columns: return c
            if c.lower() in lowmap: return lowmap[c.lower()]
        return None

    def attach_names(df: pd.DataFrame):
        gtid_col = pick(df, "GTID", "Id", "Student Id", "Gtid")
        first_col = pick(df, "First Name", "First", "Given Name")
        last_col  = pick(df, "Last Name", "Last", "Surname", "Family Name")
        name_col  = pick(df, "Name", "Full Name", "Student Name")

        out = df.copy()
        if first_col and last_col:
            out["__first"] = out[first_col].astype(str).str.strip()
            out["__last"]  = out[last_col].astype(str).str.strip()
        elif name_col:
            parts = out[name_col].astype(str).str.strip().str.split()
            out["__first"] = parts.str[0]
            out["__last"]  = parts.str[1].fillna("")
        else:
            out["__first"] = ""
            out["__last"]  = ""
        return gtid_col, out[["__first","__last"]]

    def load_event_weights_from_drive(creds: Credentials) -> dict:
        """Load event_points.json from Drive (by file ID or search a folder)."""
        drive = build("drive", "v3", credentials=creds)

        file_id = st.secrets.get("EVENT_WEIGHTS_DRIVE_FILE_ID")
        if not file_id:
            folder_id = st.secrets.get("EVENT_WEIGHTS_FOLDER_ID")
            filename = st.secrets.get("EVENT_WEIGHTS_FILENAME", "event_points.json")
            if not folder_id:
                raise RuntimeError("Missing EVENT_WEIGHTS_DRIVE_FILE_ID or EVENT_WEIGHTS_FOLDER_ID in secrets.")

            q = f"'{folder_id}' in parents and name = '{filename}' and trashed = false"
            res = drive.files().list(
                q=q,
                fields="files(id,name,modifiedTime)",
                orderBy="modifiedTime desc",
                pageSize=1
            ).execute()
            files = res.get("files", [])
            if not files:
                raise FileNotFoundError(f"'{filename}' not found in the provided folder.")
            file_id = files[0]["id"]

        req = drive.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        buf.seek(0)
        try:
            return json.load(io.TextIOWrapper(buf, encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"Invalid JSON in weights file: {e}")

    # Connect to Sheets
    with st.spinner("Connecting to Google Sheets..."):
        try:
            creds = get_google_creds()
            gc = get_gspread_client(creds)

            sheet_ref = st.secrets.get("GOOGLE_SHEETS_ID") or st.secrets.get("GOOGLE_SHEETS_URL")
            if not sheet_ref:
                raise RuntimeError("Missing GOOGLE_SHEETS_ID or GOOGLE_SHEETS_URL in secrets.")

            sh = open_spreadsheet(gc, sheet_ref)
            ws_name = st.secrets.get("GOOGLE_SHEETS_WORKSHEET")
            ws = sh.worksheet(ws_name) if ws_name else sh.get_worksheet(0)

            records = ws.get_all_records()
            df = pd.DataFrame(records)

            weights = load_event_weights_from_drive(creds)

            st.success("Connected to attendance sheet and weights file.")
            if df.empty:
                st.info("Attendance sheet is connected but returned no rows.")
        except Exception:
            st.error("Could not connect to Google Sheets / Drive.")
            st.code(traceback.format_exc())
            st.button("⬅︎ Back to menu", on_click=lambda: go("menu"))
            st.stop()

    # Compute points
    if not df.empty:
        meeting_col = [c for c in df.columns if c.strip().lower() == "what type of meeting is this?"]
        meeting_col = meeting_col[0] if meeting_col else None
        if not meeting_col:
            st.warning("Column 'What type of meeting is this?' not found in the sheet.")
        else:
            gtid_col, names_frame = attach_names(df)
            if not gtid_col:
                st.warning("No GTID column found.")
            else:
                tmp = df[[gtid_col, meeting_col]].copy()
                tmp["__meet"] = tmp[meeting_col].astype(str).str.strip()
                tmp["__w"] = tmp["__meet"].map(lambda m: float(weights.get(m, 0)))

                points = tmp.groupby(gtid_col, dropna=True)["__w"].sum().reset_index()
                points.rename(columns={gtid_col: "GTID", "__w": "Points"}, inplace=True)

                first_last = (
                    df[[gtid_col]].join(names_frame)
                    .drop_duplicates(subset=[gtid_col])
                    .rename(columns={gtid_col: "GTID"})
                )
                out = points.merge(first_last, on="GTID", how="left")
                out.rename(columns={"__first": "First Name", "__last": "Last Name"}, inplace=True)

                st.session_state["points_df"] = out

                st.subheader("Preview")

    # Search + export
    st.markdown("#### Search")
    q = st.text_input("Search by GTID or name")
    res = st.session_state.get("points_df")
    if isinstance(res, pd.DataFrame) and not res.empty:
        if q:
            ql = q.strip().lower()
            mask = (
                res["GTID"].astype(str).str.contains(ql, case=False, na=False) |
                (res["First Name"].astype(str).str.lower().str.contains(ql, na=False)) |
                (res["Last Name"].astype(str).str.lower().str.contains(ql, na=False)) |
                ((res["First Name"].astype(str) + " " + res["Last Name"].astype(str)).str.lower().str.contains(ql, na=False))
            )
            view = res[mask]
        else:
            view = res

        st.dataframe(view.sort_values(["Points","GTID"], ascending=[False, True]), use_container_width=True)
        st.download_button(
            "⬇️ Download points CSV",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="prop_points.csv",
            mime="text/csv",
        )
    else:
        st.info("No calculated results to show yet.")


        # ---------- Pricing (Lodging & Housing) ----------
    st.markdown("### Pricing (Lodging & Housing)")

    def get_excluded_gtids(key: str = "PROP_EXCLUDED_GTIDS") -> set[str]:
        val = st.secrets.get(key)
        if val is None:
            return set()
        if isinstance(val, list):
            vals = [str(x).strip() for x in val if str(x).strip()]
        else:
            vals = [s.strip() for s in str(val).split(",") if s.strip()]
        return set(vals)

    def find_email_col(df: pd.DataFrame) -> str | None:
        candidates = {"email", "email address", "gt email", "gtmail"}
        lowmap = {c.lower(): c for c in df.columns}
        for c in lowmap:
            if c.strip().lower() in candidates or "email" in c.strip().lower():
                return lowmap[c]
        return None

    if isinstance(res, pd.DataFrame) and not res.empty:
        # Inputs
        cA, cB, cC = st.columns([1,1,1])
        with cA:
            min_price = st.number_input("Min price (X)", min_value=0.0, value=50.0, step=5.0)
        with cB:
            max_price = st.number_input("Max price (Y)", min_value=0.0, value=170.0, step=5.0)
        with cC:
            method = st.selectbox(
                "Mapping",
                ["Linear", "Normal-like (tanh)", "Quantile (rank-linear)"],
                index=2,
                help=(
                    "All are monotonic (higher points ⇒ lower price). "
                    "Quantile uses ECDF/ranks (robust to skew; median → (X+Y)/2)."
                ),
            )

        # Pull email column from the original attendance df (not the aggregated one)
        email_col = find_email_col(df)

        # Merge emails into the aggregated points result
        email_frame = pd.DataFrame()
        if email_col:
            keep_cols = [c for c in (gtid_col, email_col) if c]
            email_frame = df[keep_cols].dropna(subset=[gtid_col]).drop_duplicates(subset=[gtid_col])
            email_frame = email_frame.rename(columns={gtid_col: "GTID", email_col: "Email"})
        else:
            email_frame["GTID"] = res["GTID"]
            email_frame["Email"] = ""

        merged = (
            res.merge(email_frame, on="GTID", how="left")
            .loc[:, ["GTID", "First Name", "Last Name", "Email", "Points"]]
        )

        # Exclusions from distribution
        excluded_gtids = get_excluded_gtids("PROP_EXCLUDED_GTIDS")
        merged["Excluded"] = merged["GTID"].astype(str).isin(excluded_gtids)

        # Compute mapping only on non-excluded
        working = merged.loc[~merged["Excluded"]].copy()

        def price_from_weight(w: np.ndarray) -> np.ndarray:
            # Price = Y - w*(Y - X).  w in [0,1], higher w -> lower price.
            return (max_price - w * (max_price - min_price)).astype(float)

        # ---- Compute weights w in [0,1] for each row in 'working'
        if working.empty:
            st.info("All rows are excluded by PROP_EXCLUDED_GTIDS. Nothing to price.")
            out_priced = merged.copy()
            out_priced["Expected Price"] = np.nan
        else:
            n = len(working)
            pts = working["Points"].to_numpy(dtype=float)

            if method == "Linear" or method.startswith("Normal-like"):
                pmin, pmax = float(np.min(pts)), float(np.max(pts))
                if pmax == pmin:
                    norm = np.ones(n)
                else:
                    norm = (pts - pmin) / (pmax - pmin)
                w = norm.copy()
                if method.startswith("Normal-like"):
                    alpha = 2.0  # steepness
                    w = 0.5 + 0.5 * np.tanh(alpha * (norm - 0.5))

            elif method == "Quantile (rank-linear)":
                # Rank-based weights: w = (rank-1)/(n-1), average ranks for ties.
                # Guarantees median maps to (X+Y)/2; robust to skew & outliers.
                if n == 1:
                    w = np.ones(1)
                else:
                    import pandas as _pd
                    ranks = _pd.Series(pts).rank(method="average", ascending=True)
                    w = (ranks - 1) / (n - 1)
                    w = w.to_numpy(dtype=float)

            prices = np.round(price_from_weight(w), 2)
            working["Expected Price"] = prices

            # Stitch back (excluded get blank price)
            out_priced = merged.copy()
            out_priced = out_priced.merge(
                working[["GTID", "Expected Price"]],
                on="GTID", how="left"
            )

        # Show / export
        show_excluded = st.checkbox("Show excluded rows", value=True)
        show_df = out_priced if show_excluded else out_priced.loc[~out_priced["Excluded"]]

        st.markdown("#### Pricing Preview")
        st.dataframe(
            show_df.sort_values(["Excluded","Expected Price","Points","GTID"],
                                ascending=[True, True, False, True]),
            use_container_width=True
        )

        export_df = out_priced.loc[~out_priced["Excluded"], ["GTID","First Name","Last Name","Email","Points","Expected Price"]]
        st.download_button(
            "⬇️ Download pricing CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="prop_points_pricing.csv",
            mime="text/csv",
            key="dl_pricing_csv",
        )

        st.caption(
            "Mapping notes: Linear and Quantile both send the median to (X+Y)/2. "
            "Quantile uses ranks (ECDF), so it’s less sensitive to outliers."
        )

        # ---------- Individual price checker ----------
        st.markdown("#### Individual price checker")

        colL, colR = st.columns([1,1])
        with colL:
            gtid_lookup = st.text_input("GTID (optional)", help="Enter a GTID from the table to fetch their points.")
        with colR:
            manual_points = st.number_input("Or enter points directly (optional)", min_value=0.0, value=0.0, step=1.0)

        # Resolve the points to test
        test_points = None
        test_excluded = False
        if gtid_lookup.strip():
            row = merged.loc[merged["GTID"].astype(str) == gtid_lookup.strip()]
            if not row.empty:
                test_points = float(row.iloc[0]["Points"])
                test_excluded = bool(row.iloc[0]["Excluded"])
            else:
                st.info("GTID not found in current results; using manual points if provided.")

        if test_points is None and manual_points > 0:
            test_points = float(manual_points)

        # Compute expected price for the test point using the current mapping
        if test_points is not None:
            if test_excluded:
                st.warning("This GTID is excluded by PROP_EXCLUDED_GTIDS, so no price is computed.")
            else:
                # We need the current working distribution for ranks / normalization
                vals = working["Points"].to_numpy(dtype=float)
                m = len(vals)

                if method == "Linear" or method.startswith("Normal-like"):
                    pmin, pmax = float(np.min(vals)), float(np.max(vals))
                    if pmax == pmin:
                        norm_tp = 1.0
                    else:
                        norm_tp = (test_points - pmin) / (pmax - pmin)
                    norm_tp = float(np.clip(norm_tp, 0.0, 1.0))
                    w_tp = norm_tp
                    if method.startswith("Normal-like"):
                        alpha = 2.0
                        w_tp = 0.5 + 0.5 * float(np.tanh(alpha * (norm_tp - 0.5)))

                elif method == "Quantile (rank-linear)":
                    # ECDF with average for ties: use left/right indices
                    s = np.sort(vals)
                    if m == 1:
                        r = 1.0
                    else:
                        left = int(np.searchsorted(s, test_points, side="left"))
                        right = int(np.searchsorted(s, test_points, side="right"))
                        avg_pos = 0.5 * (left + right)  # average tie position
                        # Convert 0..(m-1) scale:
                        r = (avg_pos - 1) / (m - 1)
                        r = float(np.clip(r, 0.0, 1.0))
                    w_tp = r

                expected_price = round(float(max_price - w_tp * (max_price - min_price)), 2)
                st.success(f"Expected price for {test_points:g} points ({method}): **${expected_price}**")

    else:
        st.info("Compute points first to enable pricing.")



    st.button("⬅︎ Back to menu", on_click=lambda: go("menu"))

# ---------- EVENT POINTS (edit/JSON) ----------
elif st.session_state.page == "event_points":
    st.markdown("### Event Point Values (Download/Upload JSON)")
    st.info(
        "Pulls unique events from the MAIN FORM sheet. Upload a JSON to prefill, "
        "edit weights here, then download the updated file and place it in Drive as **event_points.json**."
    )

    EVENT_COL = "What type of meeting is this?"
    DEFAULT_JSON_FILENAME = st.secrets.get("EVENT_WEIGHTS_FILENAME", "event_points.json")

    def open_main_form():
        creds = get_google_creds()
        gc = get_gspread_client(creds)
        sheet_ref = st.secrets.get("GOOGLE_SHEETS_ID") or st.secrets.get("GOOGLE_SHEETS_URL")
        if not sheet_ref:
            raise RuntimeError("Missing GOOGLE_SHEETS_ID or GOOGLE_SHEETS_URL in secrets.")
        return open_spreadsheet(gc, sheet_ref)

    with st.spinner("Connecting to Google Sheets..."):
        try:
            main_sh = open_main_form()
            ws_name = st.secrets.get("GOOGLE_SHEETS_WORKSHEET")
            form_ws = main_sh.worksheet(ws_name) if ws_name else main_sh.get_worksheet(0)
            records = form_ws.get_all_records()
            df = pd.DataFrame(records)
            if df.empty:
                st.warning("The MAIN FORM sheet is connected but returned no rows.")
                st.button("⬅︎ Back to menu", on_click=lambda: go("menu"))
                st.stop()
            if EVENT_COL not in df.columns:
                st.error(f"Column '{EVENT_COL}' not found in the MAIN FORM sheet.")
                st.button("⬅︎ Back to menu", on_click=lambda: go("menu"))
                st.stop()
            unique_events = sorted([e for e in df[EVENT_COL].dropna().astype(str).unique() if e.strip()])
            st.success("Connected. Loaded unique events from MAIN FORM.")
        except Exception:
            st.error("Could not initialize connections.")
            st.code(traceback.format_exc())
            st.button("⬅︎ Back to menu", on_click=lambda: go("menu"))
            st.stop()

    # Optional upload to prefill
    st.subheader("1) (Optional) Upload existing weights JSON to prefill")
    uploaded = st.file_uploader("Upload existing JSON", type=["json"], label_visibility="visible")

    uploaded_weights = {}
    if uploaded is not None:
        try:
            data = json.load(uploaded)
            if isinstance(data, dict):
                normalized = {}
                for k, v in data.items():
                    try:
                        x = float(v)
                        normalized[k] = int(x) if x.is_integer() else x
                    except:
                        pass
                uploaded_weights = normalized
                st.success(f"Loaded {len(uploaded_weights)} weights from uploaded JSON.")
                st.code(json.dumps(uploaded_weights, indent=2), language="json")
            else:
                st.warning("Uploaded JSON must be an object mapping event -> weight.")
        except Exception:
            st.error("Failed to parse uploaded JSON.")
            st.code(traceback.format_exc())

    # Editor
    st.subheader("2) Edit weights")
    table = pd.DataFrame({
        "Event": unique_events,
        "Weight": [uploaded_weights.get(ev, None) for ev in unique_events],
    })
    edited = st.data_editor(
        table,
        key="event_weights_editor",
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Event": st.column_config.TextColumn(disabled=True),
            "Weight": st.column_config.NumberColumn(
                "Weight",
                help="Leave blank for unweighted events. Use integers or decimals.",
                min_value=0.0,
                step=0.5,
            ),
        },
        hide_index=True,
    )

    current_json = {}
    for _, row in edited.iterrows():
        ev = str(row["Event"]).strip()
        wt = row["Weight"]
        if ev and pd.notna(wt):
            try:
                x = float(wt)
                current_json[ev] = int(x) if x.is_integer() else x
            except:
                pass

    st.subheader("3) Download JSON")
    st.download_button(
        "⬇️ Download Event Weights JSON",
        data=json.dumps(current_json, indent=2),
        file_name=DEFAULT_JSON_FILENAME,
        mime="application/json",
    )

    st.caption(
        f"After downloading **{DEFAULT_JSON_FILENAME}**, upload it to the Drive folder in the CPC folder \n"
        "after downloading reupload the JSON file and ensure that the name is event_points.json."
    )

    st.button("⬅︎ Back to menu", on_click=lambda: go("menu"))

# ---------- ACTIVITY REPORT (date/time + optional event type filter) ----------
elif st.session_state.page == "blank1":
    st.markdown("### Activity Report Attendance")

    EVENT_COL = "What type of meeting is this?"

    @st.cache_data(ttl=600)
    def _load_attendance_df():
        creds = get_google_creds()
        gc = get_gspread_client(creds)
        sheet_ref = st.secrets.get("GOOGLE_SHEETS_ID") or st.secrets.get("GOOGLE_SHEETS_URL")
        if not sheet_ref:
            raise RuntimeError("Missing GOOGLE_SHEETS_ID or GOOGLE_SHEETS_URL in secrets.")
        sh = open_spreadsheet(gc, sheet_ref)
        ws_name = st.secrets.get("GOOGLE_SHEETS_WORKSHEET")
        ws = sh.worksheet(ws_name) if ws_name else sh.get_worksheet(0)
        records = ws.get_all_records()
        return pd.DataFrame(records)

    # Preload unique event types for the selector (if the column exists)
    try:
        _df_for_options = _load_attendance_df()
        if not _df_for_options.empty and EVENT_COL in _df_for_options.columns:
            unique_event_types = sorted(
                [s for s in _df_for_options[EVENT_COL].astype(str).str.strip().unique() if s]
            )
        else:
            unique_event_types = []
    except Exception:
        unique_event_types = []

    st.caption("Filter attendance by date, time window, and (optionally) event type.")
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        the_date = st.date_input("Date")
    with colB:
        start_t = st.time_input("Start time", value=dtime(0, 0))
    with colC:
        end_t   = st.time_input("End time", value=dtime(23, 59))

    # Event Type filter (default = All)
    if unique_event_types:
        event_type_choice = st.selectbox(
            "Event type (optional)", 
            options=["All"] + unique_event_types, 
            index=0,
            help=f"Values come from the '{EVENT_COL}' column."
        )
    else:
        event_type_choice = "All"
        st.caption(f"*No event type selector shown: '{EVENT_COL}' not found or sheet empty.*")

    run = st.button("Run report", type="primary")
    st.button("⬅︎ Back to menu", on_click=lambda: go("menu"))

    if run:
        with st.spinner("Loading attendance…"):
            try:
                df = _load_attendance_df()
                if df.empty:
                    st.info("Sheet connected, but no rows found.")
                    st.stop()

                # find a timestamp column
                possible_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
                if not possible_cols:
                    st.error("Could not find a timestamp column. Please check your sheet headers.")
                    st.code(f"Available columns: {list(df.columns)}")
                    st.stop()
                ts_col = possible_cols[0]

                ts = pd.to_datetime(df[ts_col], errors="coerce")  # no infer_datetime_format (deprecated)
                start_dt = datetime.combine(the_date, start_t)
                end_dt   = datetime.combine(the_date, end_t)

                mask = (ts >= start_dt) & (ts <= end_dt)

                # Apply event type filter if selected and column exists
                if event_type_choice != "All" and EVENT_COL in df.columns:
                    mask &= (df[EVENT_COL].astype(str).str.strip() == event_type_choice)

                view = df.loc[mask].copy()

                # Status + table
                extra = "" if event_type_choice == "All" else f" | Event type: {event_type_choice}"
                st.success(
                    f"Found {len(view)} records between {start_dt:%Y-%m-%d %H:%M} and {end_dt:%Y-%m-%d %H:%M}{extra}."
                )
                st.dataframe(view, use_container_width=True)

                # Download CSV
                csv_bytes = view.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="activity_report.csv", mime="text/csv")

                # Email list (if an email column exists)
                email_col = _find_email_col(view)
                if email_col:
                    emails_series = (
                        view[email_col].astype(str).str.strip().str.lower()
                    )
                    emails_series = emails_series[emails_series.str.contains("@")]
                    unique_emails = sorted({e for e in emails_series.tolist() if e})
                    email_blob = ", ".join(unique_emails)

                    st.subheader("Email list for this window")
                    st.caption("Use the copy button or download as a .txt file.")
                    st.text_area("Comma-separated emails", value=email_blob, height=120, key="emails_blob")
                    st.download_button(
                        "⬇️ Download emails.txt",
                        data=email_blob.encode("utf-8"),
                        file_name="emails.txt",
                        mime="text/plain",
                        key="dl_emails",
                    )

                # NSBE members count (if present)
                nsbe_cols = [c for c in df.columns if c.strip().lower() == "nsbe id"]
                if not nsbe_cols:
                    st.info("No **NSBE ID** column found. Skipping NSBE member summary.")
                else:
                    nsbe_col = nsbe_cols[0]
                    nsbe_numeric = pd.to_numeric(view[nsbe_col], errors="coerce")
                    nsbe_count = nsbe_numeric.notna().sum()
                    st.markdown(f"**Registered NSBE Members (numeric NSBE ID):** {nsbe_count}")

                # Pie helpers (unchanged)
                def pie_counts_from_column(view_df, colname, title, key, top_n=None, outside_if_pct_below=0.08):
                    if colname not in view_df.columns:
                        st.info(f"No '{colname}' column found for breakdown.")
                        return
                    s = view_df[colname].astype(str).str.strip()
                    counts = s[s != ""].value_counts().rename_axis("label").reset_index(name="count")
                    if counts.empty:
                        st.info(f"No non-empty values in '{colname}'.")
                        return
                    if top_n and len(counts) > top_n:
                        head = counts.iloc[:top_n].copy()
                        other_val = counts.iloc[top_n:]["count"].sum()
                        counts = pd.concat(
                            [head, pd.DataFrame([{"label": "Other", "count": other_val}])],
                            ignore_index=True
                        )
                    total = counts["count"].sum()
                    percents = counts["count"] / total
                    textpos = ["inside" if p >= outside_if_pct_below else "outside" for p in percents]
                    fig = px.pie(counts, names="label", values="count", title=title)
                    fig.update_traces(
                        textposition=textpos,
                        texttemplate="%{label}: %{value} (%{percent:.1%})",
                        hovertemplate="%{label}<br>Count=%{value}<br>%{percent}",
                        sort=False,
                        pull=[0.02 if tp == "outside" else 0 for tp in textpos],
                    )
                    fig.update_layout(uniformtext_minsize=10, uniformtext_mode="hide",
                                      margin=dict(t=60, b=20, l=20, r=20), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True, key=key)

                pie_counts_from_column(view, "Major", "Attendance by Major", key="pie_major")
                pie_counts_from_column(view, "Year", "Attendance by Year", key="pie_year")

                dues_cols = [c for c in view.columns if "due" in c.lower()]
                if dues_cols:
                    dues_col = dues_cols[0]
                    def map_dues(v):
                        if pd.isna(v): return None
                        s = str(v).strip().lower()
                        if s in {"yes","y","true","1","paid"}: return "Paid"
                        if s in {"no","n","false","0","unpaid"} or "not paid" in s: return "Not Paid"
                        if "paid" in s and "not" not in s and "un" not in s: return "Paid"
                        return None
                    status = view[dues_col].apply(map_dues).dropna()
                    dues_counts = status.value_counts().rename_axis("label").reset_index(name="count")
                    if not dues_counts.empty:
                        fig = px.pie(dues_counts, names="label", values="count", title="Dues Paid Status")
                        fig.update_traces(
                            textposition="inside",
                            texttemplate="%{label}: %{value} (%{percent:.1%})",
                            hovertemplate="%{label}<br>Count=%{value}<br>%{percent}"
                        )
                        st.plotly_chart(fig, use_container_width=True, key="pie_dues")
                    else:
                        st.info("No recognizable dues responses in this window.")
                else:
                    st.info("No 'Dues' column found for breakdown.")

            except Exception:
                st.error("Failed to generate report.")
                st.code(traceback.format_exc())


# ---------- BLANK ----------
# ---------- MEMBER ATTENDANCE LOOKUP ----------
elif st.session_state.page == "blank2":
    st.markdown("### Member Attendance Lookup")
    st.info("Find all events a member attended (by GTID or name), shown in chronological order.")
    st.button("⬅︎ Back to menu", on_click=lambda: go("menu"))

    import pandas as pd
    import numpy as np
    from datetime import datetime, time as dtime

    @st.cache_data(ttl=600)
    def load_attendance() -> pd.DataFrame:
        """Load the same attendance sheet used elsewhere."""
        creds = get_google_creds()
        gc = get_gspread_client(creds)

        sheet_ref = st.secrets.get("GOOGLE_SHEETS_ID") or st.secrets.get("GOOGLE_SHEETS_URL")
        if not sheet_ref:
            raise RuntimeError("Missing GOOGLE_SHEETS_ID or GOOGLE_SHEETS_URL in secrets.")
        sh = open_spreadsheet(gc, sheet_ref)

        ws_name = st.secrets.get("GOOGLE_SHEETS_WORKSHEET")
        ws = sh.worksheet(ws_name) if ws_name else sh.get_worksheet(0)

        records = ws.get_all_records()
        return pd.DataFrame(records)

    def _pick(df: pd.DataFrame, *cands: str) -> str | None:
        lowmap = {c.lower(): c for c in df.columns}
        for name in cands:
            if name in df.columns:
                return name
            if name.lower() in lowmap:
                return lowmap[name.lower()]
        return None

    def _normalize_name(s: str) -> str:
        return " ".join(str(s).strip().lower().split())

    def _coerce_ts(s):
        try:
            return pd.to_datetime(s, errors="coerce")
        except Exception:
            return pd.NaT

    # ------- UI -------
    with st.form("member_lookup"):
        mode = st.radio("Search by", ["GTID", "First & Last Name"], horizontal=True)

        c1, c2, c3 = st.columns([1,1,1])
        gtid, first, last = "", "", ""
        if mode == "GTID":
            with c1:
                gtid = st.text_input("GTID", placeholder="e.g., 903123456")
        else:
            with c1:
                first = st.text_input("First name", placeholder="e.g., Jordan")
            with c2:
                last  = st.text_input("Last name",  placeholder="e.g., Davis")

        with c3:
            filter_dates = st.checkbox("Filter by date range?", value=False)

        d1, d2 = st.columns([1,1])
        start_date = end_date = None
        if filter_dates:
            with d1:
                start_date = st.date_input("Start date", value=None)
            with d2:
                end_date   = st.date_input("End date", value=None)

        submitted = st.form_submit_button("Search", type="primary")

    if submitted:
        with st.spinner("Loading attendance…"):
            try:
                df = load_attendance()
            except Exception as e:
                st.error("Could not connect to Google Sheets.")
                st.code(str(e))
                st.stop()

        if df.empty:
            st.info("Sheet connected, but no rows found.")
            st.stop()

        # ---- Identify key columns (robust to header differences)
        ts_col = None
        for c in df.columns:
            cl = c.lower().strip()
            if "time" in cl or "date" in cl:
                ts_col = c
                break
        if not ts_col:
            st.error("Could not find a timestamp/date column in the sheet.")
            st.code(f"Columns: {list(df.columns)}")
            st.stop()

        gtid_col = _pick(df, "GTID", "Id", "Student Id", "Gtid")
        first_col = _pick(df, "First Name", "First", "Given Name")
        last_col  = _pick(df, "Last Name", "Last", "Surname", "Family Name")
        event_col = _pick(df, "Event", "What type of meeting is this?", "Meeting Type", "Event Name")

        if not event_col:
            st.error("Could not find an event/meeting column (e.g., 'Event' or 'What type of meeting is this?').")
            st.code(f"Columns: {list(df.columns)}")
            st.stop()

        # ---- Clean types
        view = df.copy()
        view["__ts"] = view[ts_col].apply(_coerce_ts)

        # Name normalization for matching
        if first_col and last_col:
            view["__first_n"] = view[first_col].map(_normalize_name)
            view["__last_n"]  = view[last_col].map(_normalize_name)
            view["__full_n"]  = (view["__first_n"] + " " + view["__last_n"]).str.strip()
        else:
            view["__first_n"] = ""
            view["__last_n"]  = ""
            view["__full_n"]  = ""

        # ---- Build filter
        mask = pd.Series(True, index=view.index)

        if mode == "GTID":
            q = (gtid or "").strip()
            if not q:
                st.error("Please enter a GTID.")
                st.stop()
            if not gtid_col:
                st.error("No GTID-like column found in the sheet.")
                st.stop()
            mask &= (view[gtid_col].astype(str).str.strip() == q)
        else:
            fn = _normalize_name(first)
            ln = _normalize_name(last)
            if not fn or not ln:
                st.error("Please enter both first and last name.")
                st.stop()
            mask &= (view["__first_n"] == fn) & (view["__last_n"] == ln)

        if filter_dates and (start_date or end_date):
            if start_date:
                mask &= (view["__ts"] >= pd.Timestamp(datetime.combine(start_date, dtime.min)))
            if end_date:
                mask &= (view["__ts"] <= pd.Timestamp(datetime.combine(end_date, dtime.max)))

        result = view.loc[mask].copy()

        # ---- Shape output (chronological)
        if result.empty:
            who = gtid if mode == "GTID" else f"{first} {last}"
            st.info(f"No attendance records found for **{who}**.")
        else:
            # Choose nice display columns if present
            display_cols = []
            if ts_col: display_cols.append(("__ts", "Date/Time"))
            if event_col: display_cols.append((event_col, "Event"))
            if gtid_col: display_cols.append((gtid_col, "GTID"))
            if first_col: display_cols.append((first_col, "First Name"))
            if last_col:  display_cols.append((last_col, "Last Name"))

            tidy = result[[c for c, _ in display_cols]].copy()
            tidy = tidy.sort_values("__ts", ascending=True)
            tidy.rename(columns={c: label for c, label in display_cols}, inplace=True)

            who = ""
            if mode == "GTID":
                who = gtid
            else:
                # Use the cleaned name if available, else the typed name
                if first_col and last_col and not tidy.empty:
                    who = f"{tidy['First Name'].iloc[0]} {tidy['Last Name'].iloc[0]}".strip()
                else:
                    who = f"{first} {last}".strip()

            st.success(f"Found **{len(tidy)}** record(s) for **{who}**.")
            st.dataframe(tidy, use_container_width=True, hide_index=True)

            # Quick summary: counts per event
            with st.expander("Summary by event"):
                summary = (
                    tidy.groupby("Event", as_index=False)
                        .agg(Occurrences=("Event", "count"))
                        .sort_values(["Occurrences", "Event"], ascending=[False, True])
                )
                st.dataframe(summary, use_container_width=True, hide_index=True)

            # Download CSV
            st.download_button(
                "⬇️ Download results as CSV",
                data=tidy.to_csv(index=False).encode("utf-8"),
                file_name=f"attendance_{who.replace(' ','_')}.csv",
                mime="text/csv",
                key="dl_member_attendance_csv",
            )


# ---------- Footer ----------
st.markdown("---")
col_left, col_center, col_right = st.columns([3, 2, 3])
with col_center:
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if st.button("🔁 Refresh page", use_container_width=True):
        go()
    if st.button("🚪 Log out", use_container_width=True):
        logout()
    st.markdown("</div>", unsafe_allow_html=True)
