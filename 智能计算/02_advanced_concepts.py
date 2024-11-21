import streamlit as st
import pandas as pd
import numpy as np
# Session State
# 会话状态提供了一个类似字典的界面，您可以在其中保存在脚本重新运行之间保留的信息。
# 使用带有键或属性表示法的st.session_state来存储和调用值。例如，st.session_state["my_key"]或st.session_state.my_key。
if "counter" not in st.session_state:
    st.session_state.counter = 0

st.session_state.counter += 1

st.header(f"This page has run {st.session_state.counter} times.")
st.button("Run it again")


#
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(np.random.randn(20, 2), columns=["x", "y"])

st.header("Choose a datapoint color")
color = st.color_picker("Color", "#FF0000")
st.scatter_chart(st.session_state.df, x="x", y="y", color=color)


# Cache
# Database Connection