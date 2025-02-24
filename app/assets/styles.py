import streamlit as st

def apply_custom_styles():
    """Applies custom CSS styling to Streamlit UI."""
    st.markdown("""
        <style>
            .stTextInput>div>div>input {
                border-radius: 8px;
                padding: 10px;
            }
            .stButton>button {
                border-radius: 8px;
                background-color: #007BFF;
                color: white;
                font-weight: bold;
                transition: 0.3s;
            }
            .stButton>button:hover {
                background-color: #0056b3;
            }
        </style>
    """, unsafe_allow_html=True)
