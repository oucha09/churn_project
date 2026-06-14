import sys
from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

from streamlit_app import (
    configure_page,
    render_customers,
    render_dashboard,
    render_models,
    render_prediction,
)


configure_page("ChurnPulse")

pages = [
    st.Page(render_dashboard, title="Dashboard", icon=":material/dashboard:", url_path="dashboard", default=True),
    st.Page(render_prediction, title="Prediction", icon=":material/psychology:", url_path="prediction"),
    st.Page(render_customers, title="Customers", icon=":material/groups:", url_path="customers"),
    st.Page(render_models, title="Models", icon=":material/analytics:", url_path="models"),
]

with st.sidebar:
    st.markdown("## ChurnPulse")
    st.caption("AI Analytics Platform")

navigation = st.navigation(pages, position="sidebar", expanded=True)
navigation.run()
