# UPDRS Finger Tapping Test

A comprehensive computer vision-based application for conducting UPDRS (Unified Parkinson's Disease Rating Scale) finger tapping assessments. This project includes both a desktop Pygame version and web-based Flask/Streamlit applications.

## Features

### Core Functionality
- **Real-time hand detection and tracking** using MediaPipe
- **Finger tapping assessment** for both hands
- **Mood assessment questionnaire** for comprehensive evaluation
- **Distance measurement and analysis** between thumb and index finger
- **Real-time tap counting** with cooldown mechanisms
- **Detailed results visualization** with interactive charts
- **PDF report generation** for clinical documentation
- **Session state management** for multi-step assessments

### Available Versions
1. **Desktop Game** (`updrs_game.py`) - Pygame-based standalone application
2. **Flask Web App** (`updrs_game_web.py`) - Full-featured web application
3. **Streamlit App** (`updrs_streamlit.py`) - Modern, deployable web interface

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Choose your preferred version:

### Desktop Version
```bash
python updrs_game.py
```

### Flask Web App
```bash
python updrs_game_web.py
```
Access at `http://localhost:5001`

### Streamlit App (Recommended for Deployment)
```bash
streamlit run updrs_streamlit.py
```
Access at `http://localhost:8501`

## How to Use

### Test Process
1. **Mood Assessment**: Rate your current mood and energy level (1-5 scale)
2. **Right Hand Test**: Tap thumb and index finger together for 10 seconds
3. **Left Hand Test**: Repeat the same process with your left hand
4. **Results Analysis**: View detailed metrics, charts, and download PDF report

### Key Metrics
- **Tap Count**: Number of successful finger taps per hand
- **Tapping Rate**: Taps per second for each hand
- **Distance Analysis**: Finger separation patterns over time
- **Mood Correlation**: Relationship between mood and motor performance

## Deployment

### Streamlit Cloud (Easiest)
1. Push code to GitHub
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Deploy with one click

### Other Platforms
- Heroku
- AWS EC2
- Docker containers
- Local servers

See `streamlit_deployment_guide.md` for detailed deployment instructions.

## Technical Details

### Computer Vision
- **MediaPipe Hands**: Real-time hand landmark detection
- **Finger Tracking**: Thumb and index finger tip positions
- **Tap Detection**: Distance-based tap recognition with configurable thresholds
- **Hand Classification**: Automatic left/right hand identification

### Data Analysis
- **Real-time Processing**: Live distance and tap measurements
- **Statistical Analysis**: Mean, min, max distances and tap rates
- **Visualization**: Interactive charts using Plotly
- **Report Generation**: Professional PDF reports with charts

### User Interface
- **Responsive Design**: Works on desktop and mobile devices
- **Modern UI**: Clean, professional interface with smooth animations
- **Progress Tracking**: Real-time feedback during tests
- **Accessibility**: Clear instructions and visual feedback

## Requirements

- **Python 3.7+**
- **Camera access** for hand tracking
- **Good lighting** for optimal hand detection
- **Modern web browser** (for web versions)

## Medical Disclaimer

This application is designed for research and educational purposes. It should not be used as a substitute for professional medical diagnosis or treatment. Always consult with qualified healthcare professionals for medical assessments.

## License

This project is for educational and research purposes. Ensure compliance with medical device regulations if used in clinical settings.

