@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo Downloading spaCy model...
python -m spacy download en_core_web_sm

echo Setup complete!
echo.
echo To run the application:
echo   Streamlit UI: streamlit run app.py
echo   API Server:   python api.py
pause
