"""Быстрый старт: python run.py (порт 8099)"""
import subprocess, sys
subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8099"])
