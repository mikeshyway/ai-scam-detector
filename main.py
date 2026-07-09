"""Convenience Streamlit entry point.

The canonical application implementation remains in ``app.main`` so existing
Streamlit Cloud configuration using ``app/main.py`` continues to work.
"""

from app.main import main


if __name__ == "__main__":
    main()
