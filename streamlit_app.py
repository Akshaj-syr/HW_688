import streamlit as st

st.set_page_config(page_title="HW manager")

hw1 = st.Page("hws/HW_1.py", title="HW1")
hw2 = st.Page("hws/HW_2.py", title="HW2")
hw3 = st.Page("hws/HW_3.py", title="HW3")
hw4 = st.Page("hws/HW_4.py", title="HW4")
hw5 = st.Page("hws/HW_5.py", title="HW5")
hw7 = st.Page("hws/HW_7.py", title="HW7", default=True)

pages = {"HWs": [hw1, hw2, hw3, hw4, hw5, hw7]}

st.navigation(pages).run()
