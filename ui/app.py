"""
Streamlit Dashboard for MedicalAgentDiagnosis-MAD.

Multimodal Agentic Medical Diagnosis System with:
- Real AI Vision Analysis (torchxrayvision)
- Knowledge Graph Integration (Neo4j)
- Multi-Expert AI Consultation (Gemini)
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.schemas import PatientCase
from services.manager import DiagnosisManager

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="MedicalAgentDiagnosis-MAD",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .severity-normal { background-color: #e8f5e9; border-left: 5px solid #4CAF50; }
    .severity-mild { background-color: #fff3e0; border-left: 5px solid #FF9800; }
    .severity-moderate { background-color: #fff3e0; border-left: 5px solid #F57C00; }
    .severity-severe { background-color: #ffebee; border-left: 5px solid #f44336; }
    .severity-critical { background-color: #ffcdd2; border-left: 5px solid #b71c1c; }
    .diagnosis-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .expert-discussion {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
        max-height: 500px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


def run_async(coro):
    """Run an async coroutine in Streamlit's synchronous context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def get_severity_class(severity: str) -> str:
    """Get CSS class for severity level."""
    return f"severity-{severity.lower()}"


def main():
    """Main application entry point."""
    # Header
    st.markdown('<p class="main-header">MedicalAgentDiagnosis-MAD</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-Powered Multi-Expert Medical Diagnosis System</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("System Status")
        
        # Check API key
        api_key_set = bool(os.getenv("GOOGLE_API_KEY"))
        if api_key_set:
            st.success("Gemini AI: Connected")
        else:
            st.error("Gemini AI: Not Configured")
            st.caption("Add GOOGLE_API_KEY to .env")
        
        # Vision status
        st.info("Vision AI: Real Analysis Mode")
        st.caption("Using torchxrayvision DenseNet121")
        
        st.divider()
        
        st.markdown("### Expert Panel")
        st.markdown("""
        **Consultation Team:**
        - 🔬 Senior Radiologist
        - 🫁 Senior Pulmonologist  
        - 👨‍⚕️ Medical Director
        """)
        
        st.divider()
        
        st.markdown("### About")
        st.markdown("""
        **MedicalAgentDiagnosis-MAD** performs:
        - Real AI image analysis
        - Knowledge graph consultation
        - Multi-expert AI discussion
        - Comprehensive diagnosis reports
        """)

    # Main Content
    col_input, col_preview = st.columns([2, 1])

    with col_input:
        st.subheader("📋 Patient Case Input")
        
        uploaded_file = st.file_uploader(
            "Upload Medical Scan",
            type=["png", "jpg", "jpeg", "tiff", "bmp", "dcm"],
            help="Upload a chest X-ray or medical scan image",
        )
        
        patient_history = st.text_area(
            "Patient History & Symptoms",
            placeholder="Enter patient symptoms, medical history, age, relevant conditions...\n\nExample: 55-year-old male with persistent cough, shortness of breath, and chest pain for 2 weeks. History of smoking.",
            height=120,
        )
        
        col_id, col_age = st.columns(2)
        with col_id:
            case_id = st.text_input("Case ID", placeholder="e.g., CASE-2024-001")
        with col_age:
            patient_age = st.number_input("Patient Age", min_value=0, max_value=120, value=0)

    with col_preview:
        st.subheader("🖼️ Image Preview")
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Scan", use_container_width=True)
            st.caption(f"File: {uploaded_file.name}")
        else:
            st.info("Upload an image to preview")

    st.divider()

    # Analyze Button
    analyze_disabled = not uploaded_file or not patient_history or not api_key_set
    analyze_button = st.button(
        "🔍 Run AI Diagnosis",
        type="primary",
        use_container_width=True,
        disabled=analyze_disabled,
    )
    
    if analyze_disabled and not api_key_set:
        st.warning("Please configure GOOGLE_API_KEY in .env file")

    if analyze_button:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name

        try:
            # Create patient case
            patient_case = PatientCase(
                id=case_id or f"CASE-{hash(uploaded_file.name) % 10000:04d}",
                history=patient_history,
                image_path=temp_path,
                patient_age=patient_age if patient_age > 0 else None,
            )

            # Run diagnosis
            with st.spinner("🔬 Analyzing image with Vision AI..."):
                manager = DiagnosisManager(preload_vision_model=False)
            
            with st.spinner("🤖 Running multi-expert consultation..."):
                report = run_async(manager.run_diagnosis(patient_case))

            # Display Results
            st.success("✅ Analysis Complete!")
            
            # Report Header
            st.markdown("---")
            st.header(f"📊 Diagnostic Report")
            st.caption(f"Report ID: {report.report_id} | Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M')}")
            
            # Severity Banner
            severity_class = get_severity_class(report.severity)
            st.markdown(
                f'<div class="diagnosis-box {severity_class}">'
                f'<h3>Primary Diagnosis</h3>'
                f'<p style="font-size: 1.2rem;">{report.primary_diagnosis}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Key Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Severity", report.severity)
            with col2:
                st.metric("Confidence", f"{report.confidence:.0%}")
            with col3:
                st.metric("Urgency", report.urgency)
            with col4:
                st.metric("Risk Score", f"{report.vision_findings.risk_score:.0%}")
            
            # Two-column layout for details
            col_left, col_right = st.columns(2)
            
            with col_left:
                # Vision Findings
                st.subheader("🔬 Vision AI Findings")
                for finding in report.vision_findings.findings:
                    if "Error" in finding:
                        st.error(finding)
                    elif any(word in finding.lower() for word in ["detected", "high", "moderate"]):
                        st.warning(f"• {finding}")
                    else:
                        st.info(f"• {finding}")
                
                # Differential Diagnoses
                if report.differential_diagnoses:
                    st.subheader("🔄 Differential Diagnoses")
                    for dd in report.differential_diagnoses:
                        st.write(f"• {dd}")
            
            with col_right:
                # Recommendations
                st.subheader("📋 Recommended Actions")
                for i, action in enumerate(report.recommended_actions, 1):
                    st.write(f"{i}. {action}")
                
                if report.follow_up_timeline:
                    st.info(f"**Follow-up:** {report.follow_up_timeline}")
                
                # Limitations
                if report.limitations:
                    st.subheader("⚠️ Limitations")
                    for lim in report.limitations:
                        st.caption(f"• {lim}")
            
            # Expert Discussion (Expandable)
            st.markdown("---")
            with st.expander("🗣️ View Full Expert Discussion", expanded=False):
                st.markdown(
                    f'<div class="expert-discussion">{report.expert_discussion}</div>',
                    unsafe_allow_html=True
                )
            
            # Clinical Summary (Downloadable)
            st.markdown("---")
            with st.expander("📄 Clinical Summary Report", expanded=False):
                summary = report.to_clinical_summary()
                st.text(summary)
                st.download_button(
                    "📥 Download Report",
                    data=summary,
                    file_name=f"{report.report_id}.txt",
                    mime="text/plain"
                )
            
            # Consensus Status
            if report.consensus_reached:
                st.success("✅ Expert Consensus Reached")
            else:
                st.warning("⚠️ Experts did not reach full consensus - Additional review recommended")

        except Exception as e:
            st.error(f"Analysis Error: {str(e)}")
            st.exception(e)

        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    # Footer
    st.markdown("---")
    st.caption(
        "⚠️ **Medical Disclaimer**: This AI system is for research and demonstration purposes only. "
        "All findings must be reviewed by qualified healthcare professionals. "
        "Do not use for actual medical diagnosis or treatment decisions."
    )


if __name__ == "__main__":
    main()
