# UPDRS Streamlit App Deployment Guide

## Overview
This guide will help you deploy the UPDRS Finger Tapping Test Streamlit application to various platforms.

## Local Development

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run updrs_streamlit.py
```

The app will be available at `http://localhost:8501`

## Deployment Options

### Option 1: Streamlit Cloud (Recommended)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add Streamlit UPDRS app"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository: `chuyicui/tap_test_web`
   - Main file path: `updrs_streamlit.py`
   - Click "Deploy!"

### Option 2: Heroku

1. **Create Procfile**
   ```bash
   echo "web: streamlit run updrs_streamlit.py --server.port=$PORT --server.address=0.0.0.0" > Procfile
   ```

2. **Create setup.sh**
   ```bash
   echo "mkdir -p ~/.streamlit/
   echo \"[server]
   headless = true
   port = $PORT
   enableCORS = false
   \" > ~/.streamlit/config.toml" > setup.sh
   ```

3. **Deploy to Heroku**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 3: Docker

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   EXPOSE 8501

   CMD ["streamlit", "run", "updrs_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and Run**
   ```bash
   docker build -t updrs-streamlit .
   docker run -p 8501:8501 updrs-streamlit
   ```

### Option 4: AWS EC2

1. **Launch EC2 Instance**
   - Choose Ubuntu 20.04 LTS
   - Configure security group to allow port 8501

2. **Install Dependencies**
   ```bash
   sudo apt update
   sudo apt install python3-pip
   pip3 install -r requirements.txt
   ```

3. **Run Application**
   ```bash
   streamlit run updrs_streamlit.py --server.port=8501 --server.address=0.0.0.0
   ```

## Configuration

### Environment Variables
You can set these environment variables for configuration:

- `STREAMLIT_SERVER_PORT`: Port number (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: localhost)
- `STREAMLIT_SERVER_HEADLESS`: Run in headless mode (default: false)

### Camera Access
For camera access in deployed environments:

1. **Local Development**: Camera should work automatically
2. **Cloud Deployment**: May require additional configuration for camera access
3. **Alternative**: Use webcam simulation or file upload for testing

## Features

### Core Functionality
- ✅ Mood assessment questionnaire
- ✅ Right and left hand finger tapping tests
- ✅ Real-time hand tracking with MediaPipe
- ✅ Tap detection and counting
- ✅ Distance measurement and visualization
- ✅ Results analysis with charts
- ✅ PDF report generation
- ✅ Session state management

### UI Components
- Modern, responsive design
- Interactive charts with Plotly
- Progress tracking
- Real-time metrics
- Downloadable reports

## Troubleshooting

### Common Issues

1. **Camera not working**
   - Check camera permissions
   - Ensure camera is not being used by another application
   - For cloud deployment, camera access may be limited

2. **MediaPipe installation issues**
   ```bash
   pip install --upgrade mediapipe
   ```

3. **Streamlit not starting**
   ```bash
   streamlit --version
   pip install --upgrade streamlit
   ```

### Performance Optimization

1. **Reduce camera resolution** for better performance
2. **Adjust MediaPipe confidence thresholds**
3. **Use caching for expensive operations**

## Security Considerations

1. **HTTPS**: Always use HTTPS in production
2. **Authentication**: Add user authentication if needed
3. **Data Privacy**: Ensure patient data is handled securely
4. **Camera Access**: Implement proper camera access controls

## Monitoring

### Logs
Streamlit provides built-in logging. Check logs for:
- Application errors
- Performance metrics
- User interactions

### Analytics
Consider adding:
- User session tracking
- Performance metrics
- Error monitoring

## Support

For issues or questions:
1. Check the Streamlit documentation
2. Review the MediaPipe documentation
3. Check the application logs
4. Test locally before deploying

## License

This application is for educational and research purposes. Ensure compliance with medical device regulations if used in clinical settings.
