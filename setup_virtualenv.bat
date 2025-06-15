@echo off
REM Kotha LLM: Windows virtualenv setup
python -m venv .venv
call .venv\Scripts\activate
pip install --upgrade pip
pip install torch numpy tqdm regex
echo Optionally: pip install matplotlib rich
echo Virtual environment ready!