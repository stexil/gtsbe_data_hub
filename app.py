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



st.set_page_config(page_title="GTSBE Prop Points Hub", layout="wide")
user = auth_gate()
st.sidebar.markdown(f"**Signed in as:** {user['email']}")
st.sidebar.button("Logout", key="btn_sidebar_logout", on_click=logout)



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
    st.rerun()

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
            st.markdown("### (Blank)")
            st.write("Reserved for a future tool.")
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

# ---------- ACTIVITY REPORT (date/time filter, untouched rows) ----------
elif st.session_state.page == "blank1":
    st.markdown("### Activity Report Attendance")

    st.caption("Filter attendance by date and time window.")
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        the_date = st.date_input("Date")
    with colB:
        start_t = st.time_input("Start time", value=dtime(0, 0))
    with colC:
        end_t   = st.time_input("End time", value=dtime(23, 59))

    run = st.button("Run report", type="primary")
    st.button("⬅︎ Back to menu", on_click=lambda: go("menu"))

    if run:
        with st.spinner("Loading attendance…"):
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

                ts = pd.to_datetime(df[ts_col], errors="coerce", infer_datetime_format=True)
                start_dt = datetime.combine(the_date, start_t)
                end_dt   = datetime.combine(the_date, end_t)

                mask = (ts >= start_dt) & (ts <= end_dt)
                view = df.loc[mask].copy()

                st.success(f"Found {len(view)} records between {start_dt:%Y-%m-%d %H:%M} and {end_dt:%Y-%m-%d %H:%M}.")
                st.dataframe(view, use_container_width=True)

                csv_bytes = view.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="activity_report.csv", mime="text/csv")

                # NSBE members count (if present)
                nsbe_cols = [c for c in df.columns if c.strip().lower() == "nsbe id"]
                if not nsbe_cols:
                    st.info("No **NSBE ID** column found. Skipping NSBE member summary.")
                else:
                    nsbe_col = nsbe_cols[0]
                    nsbe_numeric = pd.to_numeric(view[nsbe_col], errors="coerce")
                    nsbe_count = nsbe_numeric.notna().sum()
                    st.markdown(f"**Registered NSBE Members (numeric NSBE ID):** {nsbe_count}")

                # Pies
                def pie_counts_from_column(view_df, colname, title, key, top_n=None, outside_if_pct_below=0.08):
                    if colname not in view_df.columns:
                        st.info(f"No '{colname}' column found for breakdown.")
                        return
                    s = view_df[colname].astype(str).str.strip()
                    counts = (
                        s[s != ""].value_counts()
                         .rename_axis("label")
                         .reset_index(name="count")
                    )
                    if counts.empty:
                        st.info(f"No non-empty values in '{colname}'.")
                        return
                    if top_n and len(counts) > top_n:
                        head = counts.iloc[:top_n].copy()
                        other_val = counts.iloc[top_n:]["count"].sum()
                        counts = pd.concat([head, pd.DataFrame([{"label": "Other", "count": other_val}])],
                                           ignore_index=True)
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
                    dues_counts = (
                        status.value_counts()
                              .rename_axis("label").reset_index(name="count")
                    )
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
elif st.session_state.page == "blank2":
    st.markdown("### Activity Report Attendance Generator.")
    st.info("Use this to generate your attendance for your specific event. Just insert the day and time range.")
    st.button("⬅︎ Back to menu", on_click=lambda: go("menu"))

# ---------- Footer ----------
st.markdown("---")
col_left, col_center, col_right = st.columns([3, 2, 3])
with col_center:
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if st.button("🔁 Refresh page", use_container_width=True):
        st.rerun()
    if st.button("🚪 Log out", use_container_width=True):
        logout()
    st.markdown("</div>", unsafe_allow_html=True)
