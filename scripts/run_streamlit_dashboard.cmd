@echo off
cd /d "c:\Topic modelling for news articles"
start "" /min "C:\Users\rtap3\AppData\Local\Programs\Python\Python314\python.exe" -m streamlit run streamlit_app.py --server.headless true --server.port 8501 -- --artifact-dir artifacts/bundled_csv
