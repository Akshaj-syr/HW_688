import streamlit as st

lab1_page = st.Page("HW_1.py", title= "Lab1")
lab2_page = st.Page("lab2.py", title= "Lab2", default = True)

pages = {"Labs": [lab1_page, lab2_page]}

pg = st.navigation(pages)  
pg.run()