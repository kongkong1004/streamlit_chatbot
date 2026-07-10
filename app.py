from pathlib import Path
from datetime import datetime
import re
import urllib.parse

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama  # ✨ 로컬 LLM 연동을 위한 라이브러리 추가


# =========================
# 기본 설정 및 정적 파일 경로 지정
# =========================

st.set_page_config(
    page_title="데메위키 사내규정 챗봇",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 데메위키 챗봇")
st.caption("FAQ 검색 결과와 로컬 LLM(Ollama)을 결합하여, 사내 규정에 기반한 자연스러운 답변을 생성합니다.")


# =========================
# 파일 경로
# =========================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
STATIC_DIR = BASE_DIR / "static"

FAQ_PATH = DATA_DIR / "faq.csv"
FAILED_LOG_PATH = LOG_DIR / "failed_questions.csv"

LOG_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)


# ====================================================================
# 로컬 PDF 파일을 브라우저 링크로 변환해주는 헬퍼 함수
# ====================================================================
def get_pdf_link(policy_name):
    pdf_filename = f"{policy_name.strip()}.pdf"
    file_path = STATIC_DIR / pdf_filename
    
    if file_path.exists():
        encoded_filename = urllib.parse.quote(pdf_filename)
        return f"/app/static/{encoded_filename}"
    return None


# =========================
# 기준 점수 및 LLM 설정
# =========================

DIRECT_THRESHOLD = 0.75  # LLM이 융통성 있게 처리하므로 기준을 살짝 낮춰 유연성 확보
CAUTION_THRESHOLD = 0.60
OLLAMA_MODEL = "gemma4:e2b"


# =========================
# ✨ [신규] Ollama 답변 생성 함수 (RAG 핵심)
# =========================

def generate_llm_answer(user_question, context_faq, is_matched=True):
    """
    검색된 FAQ 데이터를 기반으로 로컬 LLM을 사용해 맞춤형 답변을 생성합니다.
    """
    if is_matched:
        # 정확하거나 유사한 FAQ 정보가 있을 때의 프롬프트
       system_prompt = """
       당신은 인사관리팀 사내규정 챗봇입니다.

       다음 규칙을 반드시 지키세요.

       1. 인사담당자처럼 답변합니다.
       2. 인사말을 하지 않습니다.
       3. "안녕하세요"라는 표현을 사용하지 않습니다.
       4. "사용자가 질문했습니다."와 같이 질문을 반복하지 않습니다.
       5. 질문 내용을 다시 그대로 말하지 않습니다.
       6. FAQ 내용만 참고하여 답변합니다.
       7. FAQ에 없는 내용은 추측하지 않습니다.
       8. 필요한 경우에만 마지막에 HR 문의를 안내합니다.
       9. FAQ에 있는 절차가 있다면 모든 절차를 순서대로 설명합니다.
       10. 답변은 친절하지만 상세하게 작성합니다. 
       """
       user_content = (
            f"[사내 규정 참고 자료]\n"
            f"- 원래 질문: {context_faq['question']}\n"
            f"- 규정 답변: {context_faq['answer']}\n"
            f"- 관련 규정/조항: {context_faq['policy']} ({context_faq['article']})\n\n"
            f"[사용자 질문]\n"
            f"\"{user_question}\"\n\n"
            f"위 참고 자료를 바탕으로 질문에 알맞은 부드러운 대화체 답변을 생성해 주세요."
        )
    else:
        # 일치하는 FAQ가 전혀 없을 때 LLM이 정중하게 거절하는 프롬프트
        system_prompt = "당신은 사내규정 챗봇입니다. 답변하기 어려운 질문에 대해 정중하게 안내하세요."
        user_content = (
            f"사용자가 사내 규정에 대해 물었으나 적절한 FAQ 항목을 찾지 못했습니다.\n"
            f"사용자 질문: \"{user_question}\"\n\n"
            f"규정을 찾지 못해 답변이 어려우니 담당 부서(인사팀)에 문의하라는 안내와 "
            f"아래 추천 질문을 참고해 달라는 정중하고 친절한 문장 3줄 이내로 생성해 주세요."
        )

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            options={"temperature": 0.2}  # 일관된 답변을 위해 온도를 낮춤
        )
        return response['message']['content']
    except Exception as e:
        return f"⚠️ Ollama 에러 발생 (모델이 켜져 있는지 확인하세요): {str(e)}\n\n기존 답변 대치:\n{context_faq['answer'] if context_faq else ''}"


# =========================
# 텍스트 처리 및 점수 함수
# =========================

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def split_keywords(text):
    text = normalize_text(text)
    if not text:
        return []
    parts = re.split(r"[,\|/·\n]+", text)
    return [p.strip() for p in parts if p.strip()]


def keyword_score(user_question, faq_keywords):
    user_question = normalize_text(user_question)
    keywords = split_keywords(faq_keywords)
    if not keywords:
        return 0.0
    hit_count = 0
    for keyword in keywords:
        if keyword and keyword in user_question:
            hit_count += 1
    return min(hit_count / max(len(keywords), 1), 1.0)


def category_score(user_question, category):
    user_question = normalize_text(user_question)
    category = normalize_text(category)
    if not category:
        return 0.0
    if category in user_question:
        return 1.0
    return 0.0


# =========================
# 데이터 로드
# =========================

@st.cache_data
def load_faq():
    if not FAQ_PATH.exists():
        return None
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(FAQ_PATH, encoding=enc)
            df.columns = [c.strip() for c in df.columns]
            required_columns = ["질문", "대답", "관련 규정", "관련 조항", "카테고리", "키워드", "업데이트 날짜"]
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                st.error(f"faq.csv에 필요한 컬럼이 없습니다: {missing}")
                return None
            for col in required_columns:
                df[col] = df[col].fillna("").astype(str)
            return df
        except Exception as e:
            last_error = e
    st.error(f"faq.csv를 읽지 못했습니다. 마지막 오류: {last_error}")
    return None


@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_data
def build_faq_embeddings(questions):
    model = load_model()
    return model.encode(questions, normalize_embeddings=True)


# =========================
# FAQ 검색
# =========================

def find_best_faq(user_question, df, faq_embeddings):
    model = load_model()
    user_embedding = model.encode([user_question], normalize_embeddings=True)
    semantic_scores = cosine_similarity(user_embedding, faq_embeddings)[0]
    results = []

    for idx, row in df.iterrows():
        semantic = float(semantic_scores[idx])
        keyword = keyword_score(user_question, row["키워드"])
        category = category_score(user_question, row["카테고리"])

        final_score = (semantic * 0.70 + keyword * 0.20 + category * 0.10)

        results.append({
            "index": idx,
            "final_score": final_score,
            "semantic_score": semantic,
            "keyword_score": keyword,
            "category_score": category,
            "question": row["질문"],
            "answer": row["대답"],
            "policy": row["관련 규정"],
            "article": row["관련 조항"],
            "category": row["카테고리"],
            "keywords": row["키워드"],
            "updated_at": row["업데이트 날짜"]
        })

    return sorted(results, key=lambda x: x["final_score"], reverse=True)


def recommend_questions(results, top_n=2):
    return sorted(results, key=lambda x: (x["keyword_score"], x["semantic_score"]), reverse=True)[:top_n]


def save_failed_question(user_question, best_match):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "날짜": now,
        "사용자 질문": user_question,
        "가장 유사한 FAQ": best_match["question"] if best_match else "",
        "최종 점수": round(best_match["final_score"], 4) if best_match else ""
    }
    log_df = pd.DataFrame([row])
    if FAILED_LOG_PATH.exists():
        old_df = pd.read_csv(FAILED_LOG_PATH, encoding="utf-8-sig")
        log_df = pd.concat([old_df, log_df], ignore_index=True)
    log_df.to_csv(FAILED_LOG_PATH, index=False, encoding="utf-8-sig")


# ====================================================================
# 세션 상태 및 화면 인터페이스 구성
# ====================================================================

if "current_question" not in st.session_state:
    st.session_state["current_question"] = ""
if "search_results" not in st.session_state:
    st.session_state["search_results"] = None
if "selected_recommendation" not in st.session_state:
    st.session_state["selected_recommendation"] = None

df = load_faq()
if df is None:
    st.warning("data/faq.csv 파일을 확인해 주세요.")
    st.stop()

faq_questions = df["질문"].fillna("").astype(str).tolist()
faq_embeddings = build_faq_embeddings(faq_questions)

user_input = st.chat_input("예: 재택근무 신청 기준이 어떻게 되나요?")

if user_input:
    st.session_state["current_question"] = user_input
    st.session_state["search_results"] = find_best_faq(user_input, df, faq_embeddings)
    st.session_state["selected_recommendation"] = None

if st.session_state["current_question"]:
    st.chat_message("user").markdown(st.session_state["current_question"])
    
    results = st.session_state["search_results"]
    best = results[0]

    with st.chat_message("assistant"):
        # 상황 A: 신뢰도가 높을 때 ➡️ LLM이 FAQ 내용을 바탕으로 예쁘게 살을 붙여 답변 생성
        if best["final_score"] >= DIRECT_THRESHOLD:
            st.success("🤖 AI가 규정 데이터를 분석하여 작성한 답변입니다.")
            
            with st.spinner("AI 답변 생성 중..."):
                llm_answer = generate_llm_answer(st.session_state["current_question"], best, is_matched=True)
            st.write(llm_answer)

            # 증거 제출 (근거 규정 표기)
            policy_name = best["policy"]
            st.markdown(f"--- \n**📄 근거 규정:** {policy_name} ({best['article']})")
            pdf_url = get_pdf_link(policy_name)
            if pdf_url:
                st.markdown(f"🔗 [**👉 [{policy_name}] 원문 PDF 열기 (새 탭)**]({pdf_url})")

        # 상황 B: 신뢰도가 애매할 때
        elif best["final_score"] >= CAUTION_THRESHOLD:
            st.warning("⚠️ 질문과 유사한 규정을 찾았으나 의도와 조금 다를 수 있습니다.")
            
            with st.spinner("AI 답변 생성 중..."):
                llm_answer = generate_llm_answer(st.session_state["current_question"], best, is_matched=True)
            st.write(llm_answer)

            policy_name = best["policy"]
            st.markdown(f"--- \n**📄 근거 규정:** {policy_name} ({best['article']})")
            pdf_url = get_pdf_link(policy_name)
            if pdf_url:
                st.markdown(f"🔗 [**👉 [{policy_name}] 원문 PDF 열기 (새 탭)**]({pdf_url})")

        # 상황 C: 아예 모르는 질문일 때 ➡️ LLM이 정중한 거절 답변 생성 + 추천 버튼 제공
        else:
            save_failed_question(st.session_state["current_question"], best)
            
            with st.spinner("안내 문구 생성 중..."):
                fallback_answer = generate_llm_answer(st.session_state["current_question"], None, is_matched=False)
            st.error(fallback_answer)
            
            st.markdown("🔍 **대신 이런 질문들은 어떠신가요?**")
            recommendations = recommend_questions(results, top_n=2)
            
            for idx, item in enumerate(recommendations, start=1):
                if st.button(f"🙋‍♂️ {item['question']}", key=f"rec_btn_{idx}_{item['index']}"):
                    st.session_state["selected_recommendation"] = item
            
            if st.session_state["selected_recommendation"] is not None:
                sel = st.session_state["selected_recommendation"]
                st.markdown("---")
                st.info(f"👉 선택하신 질문 '{sel['question']}'에 대해 생성된 AI 답변입니다.")
                
                with st.spinner("추천 답변 생성 중..."):
                    rec_llm_answer = generate_llm_answer(sel['question'], sel, is_matched=True)
                st.write(rec_llm_answer)
                
                p_name = sel["policy"]
                st.markdown(f"**📄 관련 규정:** {p_name} ({sel['article']})")
                pdf_url = get_pdf_link(p_name)
                if pdf_url:
                    st.markdown(f"🔗 [**👉 [{p_name}] 원문 PDF 열기 (새 탭)**]({pdf_url})")