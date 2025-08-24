try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except:
    pass


import streamlit as st
from main import crew

st.title("🏥 Healthcare Diagnostic Support System")

patient_input = st.text_area("Enter patient symptoms and history:")

if st.button("Run Diagnostic Support"):
    if patient_input.strip():
        with st.spinner("Analyzing..."):
            result = crew.kickoff(inputs={"patient_input": patient_input})
        st.subheader("✅ Results")
        st.write(result)
    else:
        st.warning("Please enter patient symptoms first.")

