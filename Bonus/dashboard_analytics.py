import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src"))

from streamlit_app import configure_page, render_dashboard


configure_page("ChurnPulse | Dashboard")
render_dashboard()
