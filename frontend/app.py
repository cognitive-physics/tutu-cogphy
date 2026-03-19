import random

import pandas as pd
import requests
import streamlit as st

INIT_REFERENCE = "这是一个中等密度的初始参考系，用于触发第一次 ETA 差分计算。"
BACKEND = "http://localhost:8000"

st.set_page_config(page_title="Cognitive Engine", layout="wide")
st.markdown(
    """
    <style>
    .main { background-color: #ffffff; }
    div.block-container { padding-top: 2rem; }
    .stMetric { border: 1px solid #eeeeee; padding: 15px; border-radius: 0px; }
    .mode-tag { padding: 5px 15px; font-family: monospace; font-weight: 900; }
    * { transition: none !important; animation: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []
if "last_ai_output" not in st.session_state:
    st.session_state.last_ai_output = INIT_REFERENCE


def simulate_ai_reply(user_message: str, compression_rate: float) -> str:
    base_len = len(user_message)
    reply_len = max(10, int(base_len * compression_rate))
    words = user_message.split()
    if not words:
        return INIT_REFERENCE
    simulated = []
    while len(" ".join(simulated)) < reply_len:
        simulated.append(random.choice(words))
    return " ".join(simulated)[:reply_len]


with st.sidebar:
    st.title("ENGINE")
    person_id = st.text_input("ID", value="admin_01")
    importance = st.slider("IMPORTANCE", 0.0, 1.0, 0.8)
    pressure = st.slider("PRESSURE", 0.0, 1.0, 0.2)
    horizon = st.slider("HORIZON", 0.0, 1.0, 0.7)
    if st.button("RESET PROFILE"):
        try:
            requests.delete(f"{BACKEND}/profile/{person_id}", timeout=5)
        except Exception:
            pass
        st.session_state.history = []
        st.session_state.last_ai_output = INIT_REFERENCE
        st.rerun()

st.subheader("COGNITIVE INPUT")
message = st.text_area(
    "LOGIC_STREAM",
    height=120,
    help="Ctrl+Enter 发送数据点",
    placeholder="输入高密度逻辑进行采样...",
)

if message:
    payload = {
        "person_id": person_id,
        "message": message,
        "importance": importance,
        "env_pressure": pressure,
        "goal_horizon": horizon,
        "last_ai_output": st.session_state.last_ai_output,
    }

    try:
        res = requests.post(f"{BACKEND}/chat", json=payload, timeout=10)
        res.raise_for_status()
        data = res.json()

        st.session_state.last_ai_output = simulate_ai_reply(
            message, data["compression_rate"]
        )
        st.session_state.history.append(
            {
                "step": len(st.session_state.history),
                "complexity": data["user_complexity"],
                "uncertainty": data["uncertainty"],
                "eta": data["eta"],
                "mu": data["mu"],
                "mode": data["output_mode"],
            }
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            mode = data["output_mode"]
            color_map = {
                "compressed": "#000000",
                "metaphor": "#777777",
                "slow_expand": "#dddddd",
            }
            text_map = {
                "compressed": "#ffffff",
                "metaphor": "#ffffff",
                "slow_expand": "#000000",
            }
            st.markdown(
                f'<div class="mode-tag" style="background:{color_map[mode]};color:{text_map[mode]};text-align:center;">{mode.upper()}</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.metric("ETA (同步率)", f"{data['eta']:.4f}")
        with c3:
            st.metric("R (认知留白)", f"{data['uncertainty']:.4f}")
        with c4:
            st.metric("STABLE", "TRUE" if data["system_stable"] else "FALSE")

        st.divider()
        df = pd.DataFrame(st.session_state.history)
        col_left, col_right = st.columns(2)
        with col_left:
            st.caption("COMPLEXITY (用户信息密度流)")
            st.line_chart(df.set_index("step")["complexity"])
            st.caption("MU (内外一致性扰动)")
            st.line_chart(df.set_index("step")["mu"])
        with col_right:
            st.caption("R (系统收敛趋势)")
            st.area_chart(df.set_index("step")["uncertainty"])
            st.caption("ETA (解码动态对齐)")
            st.line_chart(df.set_index("step")["eta"])

    except requests.exceptions.ConnectionError:
        st.error("ENGINE_OFFLINE: 无法连接到 localhost:8000，请先启动后端。")
    except Exception as exc:
        st.error(f"ENGINE_ERROR: {exc}")

with st.expander("RAW_FRAME_DATA"):
    st.write(st.session_state.history)
