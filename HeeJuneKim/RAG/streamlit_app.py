import streamlit as st
from retrieve_and_generate import hybrid_diagnosis

st.set_page_config(page_title="의료 챗봇", page_icon="🩺", layout="wide")
st.title("🧠 증상을 입력하면 EXAONE이 병명을 추론해드립니다")

# ✅ 유튜브 iframe 생성 함수
def get_youtube_iframe(disease_name):
    from urllib.parse import quote_plus
    query = quote_plus(disease_name + " 설명")
    return f"""
    <iframe width="700" height="400"
        src="https://www.youtube.com/embed?listType=search&list={query}"
        frameborder="0" allowfullscreen></iframe>
    """

# ✅ 사용자 입력
user_input = st.text_area("🍬 증상을 자유롭게 입력해 주세요", placeholder="기침이 나고 몸살 기운이 있어요")

if user_input:
    if st.button("🩺 병명 추론 실행"):
        with st.spinner("🔍 EXAONE이 병명을 추론 중입니다..."):
            predicted_disease = hybrid_diagnosis(user_input)

        st.markdown("## ✅ 추론된 병명")
        st.markdown(f"<h2 style='color: darkblue'>{predicted_disease}</h2>", unsafe_allow_html=True)

        st.markdown("## 📺 관련 YouTube 영상")
        st.markdown(get_youtube_iframe(predicted_disease), unsafe_allow_html=True)
