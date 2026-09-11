"""Stable root entry point: python -m streamlit run main.py."""
import matplotlib
matplotlib.use('Agg')
from gui.app import run_app

if __name__ == '__main__':
    run_app()
