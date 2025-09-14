#!/usr/bin/env python3
"""
Simple test script to verify Streamlit app components work
"""

import streamlit as st
import numpy as np
import time
from datetime import datetime

def test_basic_functionality():
    """Test basic Streamlit functionality"""
    st.title("🧪 UPDRS Streamlit App Test")
    
    st.success("✅ Streamlit is working!")
    
    # Test session state
    if 'test_counter' not in st.session_state:
        st.session_state.test_counter = 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Increment Counter"):
            st.session_state.test_counter += 1
            st.rerun()
    
    with col2:
        st.metric("Counter", st.session_state.test_counter)
    
    with col3:
        st.metric("Current Time", datetime.now().strftime("%H:%M:%S"))
    
    # Test data visualization
    st.subheader("📊 Test Chart")
    
    # Generate sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)
    
    import plotly.graph_objects as go
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Test Data'))
    fig.update_layout(title="Sample Distance Over Time", height=300)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Test file operations
    st.subheader("📄 Test PDF Generation")
    
    if st.button("Generate Test PDF"):
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from io import BytesIO
        
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.drawString(100, 750, "UPDRS Test Report")
        c.drawString(100, 700, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        c.drawString(100, 650, f"Counter Value: {st.session_state.test_counter}")
        c.save()
        
        st.download_button(
            label="Download Test PDF",
            data=buffer.getvalue(),
            file_name=f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )
    
    # Test MediaPipe import
    st.subheader("🤲 MediaPipe Test")
    try:
        import mediapipe as mp
        st.success("✅ MediaPipe imported successfully!")
        
        # Test MediaPipe hands initialization
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        st.success("✅ MediaPipe Hands initialized successfully!")
        
    except ImportError as e:
        st.error(f"❌ MediaPipe import failed: {e}")
    except Exception as e:
        st.error(f"❌ MediaPipe initialization failed: {e}")
    
    # Test OpenCV import
    st.subheader("📹 OpenCV Test")
    try:
        import cv2
        st.success("✅ OpenCV imported successfully!")
        
        # Test camera availability
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            st.success("✅ Camera is available!")
            cap.release()
        else:
            st.warning("⚠️ Camera not available (this is normal in some environments)")
        
    except ImportError as e:
        st.error(f"❌ OpenCV import failed: {e}")
    except Exception as e:
        st.error(f"❌ OpenCV test failed: {e}")
    
    # Instructions
    st.subheader("🚀 Next Steps")
    st.info("""
    If all tests pass, you can run the full UPDRS app with:
    
    ```bash
    streamlit run updrs_streamlit.py
    ```
    
    The app will be available at: http://localhost:8501
    """)

if __name__ == "__main__":
    test_basic_functionality()
