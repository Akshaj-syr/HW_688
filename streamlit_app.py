import streamlit as st

st.set_page_config(page_title="HW manager")

hw1 = st.Page("hws/HW_1.py", title="HW1")
hw2 = st.Page("hws/HW_2.py", title="HW2", default=True)

pages = {"HWs": [hw1, hw2]}

st.navigation(pages).run()
