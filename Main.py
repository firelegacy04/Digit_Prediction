import streamlit as st

st.title("my fifth GUI")
st.image("anime girl.jpg")
st.header("button")
if st.button("Say Hello"):
    st.write("Hello")
else:
    st.write("goodbye")
