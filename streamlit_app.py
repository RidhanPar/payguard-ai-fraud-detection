"""
Root Streamlit entry point for PayGuard.

Streamlit Community Cloud looks for a root-level entrypoint file.
This file imports and runs the main dashboard from app/dashboard.py.

Run locally:

    streamlit run streamlit_app.py
"""

from app.dashboard import main


if __name__ == "__main__":
    main()
