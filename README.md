# UPDRS Finger Tapping Test

A digital assessment tool for measuring finger tapping performance using computer vision. The app tracks your hand movements through your camera to count finger taps and analyze motor function.

## What it does

- **Mood Check**: Simple questionnaire about how you're feeling
- **Hand Testing**: Tap your thumb and index finger together for 10 seconds with each hand
- **Results**: Shows tap counts, speed, and generates a PDF report
- **Camera Tracking**: Uses your webcam to automatically detect and count finger taps

## How to run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run updrs_streamlit.py
```

3. Open your browser to `http://localhost:8501`

## Available versions

- **Streamlit App** (recommended) - Modern web interface
- **Flask Web App** - Full-featured web application  
- **Desktop Game** - Standalone Pygame application

## Requirements

- Python 3.7+
- Webcam/camera access
- Good lighting for hand detection

## Disclaimer

This tool is for research and educational purposes only. Not intended for medical diagnosis or treatment.

