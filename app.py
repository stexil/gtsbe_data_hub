"""
Minimal Streamlit template for Google OAuth (Desktop App flow - 2A).
- Opens the system browser for sign-in
- Caches the token locally (token.json)
- After login, fetches the user's email
- If email is "sguexil@gmail.com" -> show Welcome
else -> prompt to use GTSBE email
How to run:
1) pip install -r requirements.txt
2) Create .streamlit/secrets.toml (see bottom of this file for example)
3) streamlit run oauth_template.py
"""
from __future__ import annotations
import io
import json
import os
import requests
import streamlit as st
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
import traceback
import plotly.express as px
import numpy



ALLOWED = {
    "gtsbe.publications@gmail.com",
    "gtsbe.pr@gmail.com",
    "gtsbe.pci@gmail.com",
    "gtsbe.treasurer@gmail.com",
    "gtsbe.parliamentarian@gmail.com",
    "gtsbe.tribe@gmail.com",
    "gtsbe.vicepresident@gmail.com",
    "gtsbe.membership@gmail.com",
    "gtsbe.historian@gmail.com",
    "gtsbe.secretary@gmail.com",
    "gtsbe.ldr@gmail.com",
    "gtsbe.torch@gmail.com",
    "gtsbe.academic@gmail.com",
    "gtsbe.international@gmail.com",
    "gtsbe.programs@gmail.com",
    "gtsbe.tcomm@gmail.com",
    "gtsbe.president@gmail.com",
    "gtsbe.finance@gmail.com",
    "gtsbe.cpcchair@gmail.com",
    "sguexil@gmail.com",
}
def norm(s: str) -> str:
    # strip whitespace, lowercase, collapse weird unicode spaces
    return " ".join(s.split()).lower()


# --- Config ---
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive"
]
TOKEN_PATH = "token.json"  # stored next to this script; delete to force re-login



# --- Helpers ---

def load_client_config() -> dict:
    cfg_str = st.secrets.get("GOOGLE_OAUTH_CLIENT_CONFIG")
    if not cfg_str:
        st.stop()
    return json.loads(cfg_str)


def get_stored_creds() -> Credentials | None:
    if os.path.exists(TOKEN_PATH):
        try:
            return Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        except Exception:
            return None
    return None


def save_creds(creds: Credentials) -> None:
    with open(TOKEN_PATH, "w") as f:
        f.write(creds.to_json())


def do_login() -> Credentials:
    cfg = load_client_config()
    flow = InstalledAppFlow.from_client_config(cfg, scopes=SCOPES)
    # Runs a temporary localhost server and opens the system browser
    creds = flow.run_local_server(port=0, prompt="consent")
    save_creds(creds)
    return creds

# ---- Global footer controls (always visible) ----
def render_footer():
    st.markdown("---")
    # Use three columns, buttons in the center one
    col_left, col_center, col_right = st.columns([3, 2, 3])

    with col_center:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

        if st.button("🔁 Refresh page", use_container_width=True):
            st.rerun()

        if st.button("🚪 Log out", use_container_width=True):
            try:
                os.remove(TOKEN_PATH)   # delete cached OAuth token
            except FileNotFoundError:
                pass
            # clear UI/session + caches
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            try:
                st.cache_data.clear()
                st.cache_resource.clear()
            except Exception:
                pass
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def fetch_email(creds: Credentials) -> str | None:
    # Use the OAuth2 userinfo endpoint to get email
    resp = requests.get(
        "https://www.googleapis.com/oauth2/v3/userinfo",
        headers={"Authorization": f"Bearer {creds.token}"},
        timeout=15,
    )
    if resp.ok:
        return resp.json().get("email")
    return None


# --- UI Flow ---
# Only show the Desktop App caption before login
creds = get_stored_creds()
if not creds or not creds.valid:
    st.title("🔐 Google Login")
    st.caption("This demo uses the **Desktop App** OAuth client. No redirect URIs needed.")

    with st.container(border=True):
        st.subheader("Step 1: Sign in with Google")
        st.write("Click below to open a browser window for Google sign‑in.")
        if st.button("Sign in with Google", type="primary"):
            try:
                creds = do_login()
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")
                st.stop()
    st.stop()

# If we reach here, we have valid creds
email = fetch_email(creds)
# Show success only once after a fresh login
if "just_logged_in" not in st.session_state:
    st.session_state.just_logged_in = True
if st.session_state.just_logged_in:
    st.success("✅ Login successful. User has been authenticated.")
    st.session_state.just_logged_in = False

# --- Authz check ---
if email is None:
    st.warning("Could not fetch your email. Make sure the 'userinfo.email' scope is present.")
    ok = False
else:
    ok = norm(email) in {norm(x) for x in ALLOWED}

# ---------------- MENU / NAV ----------------
# Keep a simple, persistent header on every logged-in page:
st.write(f"Signed in as: **{email or 'unknown'}**")

if ok:
    # page state
    if "page" not in st.session_state:
        st.session_state.page = "menu"

    def go(page_name: str):
        st.session_state.page = page_name
        st.rerun()

    if st.session_state.page == "menu":
        # Show the welcome/title ONLY on the menu screen
        st.markdown(f"## 🎉 Welcome  GTSBE Prop Points Calculator")
        st.caption("Choose a tool to continue.")

        # Tile grid (2x2)
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
                st.write("Use this to generate your activity report attendance with the date and time of your event.")
                st.button("Open", key="tile_blank1", on_click=lambda: go("blank1"))
        with col4:
            with st.container(border=True):
                st.markdown("### (Blank)")
                st.write("Reserved for a future tool.")
                st.button("Open", key="tile_blank2", on_click=lambda: go("blank2"))

    elif st.session_state.page == "prop_points":
        st.markdown("### Prop Points Calculator (Template)")

        import traceback
        import gspread
        import pandas as pd
        import io, json
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload

        # ---------- helpers ----------
        def load_event_weights_from_drive(creds) -> dict:
            """Load event_points_template.json from Drive using either a file ID or by searching a folder."""
            drive = build("drive", "v3", credentials=creds)

            file_id = st.secrets.get("EVENT_WEIGHTS_DRIVE_FILE_ID")
            if not file_id:
                folder_id = st.secrets.get("EVENT_WEIGHTS_FOLDER_ID")
                filename = st.secrets.get("EVENT_WEIGHTS_FILENAME", "event_points.json")
                if not folder_id:
                    raise RuntimeError("Missing EVENT_WEIGHTS_DRIVE_FILE_ID or EVENT_WEIGHTS_FOLDER_ID in secrets.")

                q = f"'{folder_id}' in parents and name = '{filename}' and trashed = false"
                res = drive.files().list(q=q,
                                        fields="files(id,name,modifiedTime)",
                                        orderBy="modifiedTime desc",
                                        pageSize=1).execute()
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

        def pick(df, *cands):
            """Return first existing column name from candidates (case-insensitive)."""
            lowmap = {c.lower(): c for c in df.columns}
            for c in cands:
                if c in df.columns: return c
                if c.lower() in lowmap: return lowmap[c.lower()]
            return None

        def attach_names(df):
            """Return a small frame with GTID + First/Last picked from common patterns."""
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

        # ---------- load attendance + weights ----------
        with st.spinner("Connecting to Google Sheets..."):
            try:
                gc = gspread.authorize(creds)

                sheet_ref = st.secrets.get("GOOGLE_SHEETS_ID") or st.secrets.get("GOOGLE_SHEETS_URL")
                if not sheet_ref:
                    raise RuntimeError("Missing GOOGLE_SHEETS_ID or GOOGLE_SHEETS_URL in .streamlit/secrets.toml")

                sh = gc.open_by_url(sheet_ref) if str(sheet_ref).startswith("http") else gc.open_by_key(sheet_ref)
                ws_name = st.secrets.get("GOOGLE_SHEETS_WORKSHEET")
                ws = sh.worksheet(ws_name) if ws_name else sh.get_worksheet(0)

                records = ws.get_all_records()
                df = pd.DataFrame(records)

                weights = load_event_weights_from_drive(creds)

                st.success("Successfully connected with the attendance form and weights file.")
                if df.empty:
                    st.info("Attendance sheet is connected but returned no rows.")
            except Exception as e:
                st.error("Could not connect to Google Sheets / Drive.")
                st.code(traceback.format_exc())
                st.button("⬅︎ Back to menu", on_click=lambda: go("menu"))
                render_footer()
                st.stop()

        # ---------- calculate points ----------
        if not df.empty:
            meeting_col = pick(df, "What type of meeting is this?")
            if not meeting_col:
                st.warning("Column 'What type of meeting is this?' not found in the sheet.")
            else:
                gtid_col, names_frame = attach_names(df)
                if not gtid_col:
                    st.warning("No GTID column found.")
                else:
                    # normalize meeting type and map to weights (default 0 if missing)
                    tmp = df[[gtid_col, meeting_col]].copy()
                    tmp["__meet"] = tmp[meeting_col].astype(str).str.strip()
                    tmp["__w"] = tmp["__meet"].map(lambda m: float(weights.get(m, 0)))

                    # sum weights per GTID
                    points = tmp.groupby(gtid_col, dropna=True)["__w"].sum().reset_index()
                    points.rename(columns={gtid_col: "GTID", "__w": "Points"}, inplace=True)

                    # attach first/last (first occurrence)
                    first_last = (
                        df[[gtid_col]].join(names_frame)
                        .drop_duplicates(subset=[gtid_col])
                        .rename(columns={gtid_col: "GTID"})
                    )
                    out = points.merge(first_last, on="GTID", how="left")
                    out.rename(columns={"__first": "First Name", "__last": "Last Name"}, inplace=True)

                    # cache results for search/export
                    st.session_state["points_df"] = out

                    st.subheader("Preview")
                    #st.dataframe(out.sort_values("Points", ascending=False).head(15), use_container_width=True)

        # ---------- search ----------
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


    elif st.session_state.page == "event_points":
        st.markdown("### Event Point Values (Download/Upload JSON)")
        st.info(
        "Pulls unique events from the MAIN FORM sheet. You can upload a JSON to prefill, edit weights here, "
        "then download the updated file. Finally, place it back in the CPC Drive folder and rename it **event_points.json**."
        )

        import json
        import traceback
        import gspread
        import pandas as pd

        EVENT_COL = "What type of meeting is this?"
        DEFAULT_JSON_FILENAME = st.secrets.get("EVENT_WEIGHTS_FILENAME", "event_points.json")

        def open_spreadsheet(gc, ref: str):
            return gc.open_by_url(ref) if str(ref).startswith("http") else gc.open_by_key(ref)

        # --- Load MAIN FORM (read-only is fine) ---
        with st.spinner("Connecting to Google Sheets..."):
            try:
                gc = gspread.authorize(creds)

                sheet_ref = st.secrets.get("GOOGLE_SHEETS_ID") or st.secrets.get("GOOGLE_SHEETS_URL")
                if not sheet_ref:
                    raise RuntimeError("Missing GOOGLE_SHEETS_ID or GOOGLE_SHEETS_URL in secrets.")
                main_sh = open_spreadsheet(gc, sheet_ref)

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

        # --- (Optional) Upload existing JSON to prefill ---
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
                            pass  # skip non-numeric
                    uploaded_weights = normalized
                    st.success(f"Loaded {len(uploaded_weights)} weights from uploaded JSON.")
                    st.code(json.dumps(uploaded_weights, indent=2), language="json")
                else:
                    st.warning("Uploaded JSON must be an object mapping event -> weight.")
            except Exception:
                st.error("Failed to parse uploaded JSON.")
                st.code(traceback.format_exc())

        # --- Editor ---
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

        # Build JSON from edited table (only numeric rows)
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
            f"After downloading **{DEFAULT_JSON_FILENAME}**, upload it to the Drive folder that contains your MAIN FORM spreadsheet.\n"
            "Tip: In Google Drive, open the MAIN FORM file, click the folder icon by its title to open the parent folder, "
            "then upload the JSON there."
        )

        st.button("⬅︎ Back to menu", on_click=lambda: go("menu"))



    elif st.session_state.page == "blank1":
        st.markdown("### Activity Report Attendance")

        import gspread
        import pandas as pd
        import traceback
        from datetime import datetime, time as dtime

        # --- UI controls ---
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
                    # authorize + open sheet
                    gc = gspread.authorize(creds)
                    sheet_ref = st.secrets.get("GOOGLE_SHEETS_ID") or st.secrets.get("GOOGLE_SHEETS_URL")
                    if not sheet_ref:
                        raise RuntimeError("Missing GOOGLE_SHEETS_ID or GOOGLE_SHEETS_URL in .streamlit/secrets.toml")

                    sh = gc.open_by_url(sheet_ref) if str(sheet_ref).startswith("http") else gc.open_by_key(sheet_ref)
                    ws_name = st.secrets.get("GOOGLE_SHEETS_WORKSHEET")
                    ws = sh.worksheet(ws_name) if ws_name else sh.get_worksheet(0)

                    # dataframe
                    records = ws.get_all_records()
                    df = pd.DataFrame(records)

                    if df.empty:
                        st.info("Sheet connected, but no rows found.")
                        st.stop()

                    # try to locate timestamp column
                    possible_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
                    if not possible_cols:
                        st.error("Could not find a timestamp column. Please check your sheet headers.")
                        st.code(f"Available columns: {list(df.columns)}")
                        st.stop()
                    ts_col = possible_cols[0]

                    # parse + filter
                    ts = pd.to_datetime(df[ts_col], errors="coerce", infer_datetime_format=True)
                    start_dt = datetime.combine(the_date, start_t)
                    end_dt   = datetime.combine(the_date, end_t)

                    mask = (ts >= start_dt) & (ts <= end_dt)
                    view = df.loc[mask].copy()

                    # show unmodified rows
                    st.success(f"Found {len(view)} records between {start_dt:%Y-%m-%d %H:%M} and {end_dt:%Y-%m-%d %H:%M}.")
                    st.dataframe(view, use_container_width=True)

                    # export options
                    csv_bytes = view.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="activity_report.csv", mime="text/csv")

                    # ---------------- NSBE registered members section ----------------
                    nsbe_cols = [c for c in df.columns if c.strip().lower() == "nsbe id"]
                    if not nsbe_cols:
                        st.info("No **NSBE ID** column found. Skipping NSBE member summary.")
                    else:
                        nsbe_col = nsbe_cols[0]
                        nsbe_numeric = pd.to_numeric(view[nsbe_col], errors="coerce")
                        nsbe_count = nsbe_numeric.notna().sum()
                        st.markdown(f"**Registered NSBE Members (numeric NSBE ID):** {nsbe_count}")
                    # ---------------- Pie Charts ----------------
                    # ---------- helpers ----------
                    # ---------- helper: pie with smart outside labels ----------
                    def pie_counts_from_column(view_df, colname, title, key, top_n=None, outside_if_pct_below=0.08):
                        import pandas as pd
                        import plotly.express as px

                        if colname not in view_df.columns:
                            st.info(f"No '{colname}' column found for breakdown.")
                            return

                        s = view_df[colname].astype(str).str.strip()
                        counts = (s[s != ""].value_counts()
                                    .rename_axis("label")
                                    .reset_index(name="count"))

                        if counts.empty:
                            st.info(f"No non-empty values in '{colname}'.")
                            return

                        # Optional: collapse long tail into "Other"
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
                            textposition=textpos,                 # inside for big slices, outside for small
                            texttemplate="%{label}: %{value} (%{percent:.1%})",
                            hovertemplate="%{label}<br>Count=%{value}<br>%{percent}",
                            sort=False,                           # keep displayed order from counts
                            pull=[0.02 if tp == "outside" else 0 for tp in textpos],  # tiny nudge for small slices
                        )
                        fig.update_layout(
                            uniformtext_minsize=10,
                            uniformtext_mode="hide",
                            margin=dict(t=60, b=20, l=20, r=20),
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True, key=key)


                    # ---------- Major ----------
                    pie_counts_from_column(view, "Major", "Attendance by Major", key="pie_major")

                    # ---------- Year ----------
                    pie_counts_from_column(view, "Year", "Attendance by Year", key="pie_year")

                    # ---------- Dues (normalized -> counts -> pie) ----------
                    dues_cols = [c for c in view.columns if "due" in c.lower()]
                    if dues_cols:
                        dues_col = dues_cols[0]

                        # normalize to Paid / Not Paid (drop unknowns)
                        def map_dues(v):
                            if pd.isna(v): return None
                            s = str(v).strip().lower()
                            if s in {"yes","y","true","1","paid"}: return "Paid"
                            if s in {"no","n","false","0","unpaid"} or "not paid" in s: return "Not Paid"
                            if "paid" in s and "not" not in s and "un" not in s: return "Paid"
                            return None

                        status = view[dues_col].apply(map_dues).dropna()
                        dues_counts = (status.value_counts()
                                            .rename_axis("label").reset_index(name="count"))

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

    elif st.session_state.page == "blank2":
        st.markdown("### Activity Report Attendance Generator.")
        st.info("Use this to generate your attendance for your specific event. Just insert in the day, and time range")
        st.button("⬅︎ Back to menu", on_click=lambda: go("menu"))
else:
    st.markdown("## ⚠️ Please use your GTSBE email!")
    st.write("Log out and sign in with the correct account.")
    

render_footer()
