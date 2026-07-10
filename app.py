from pathlib import Path
from datetime import datetime
import re
import urllib.parse

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 기본 설정 및 정적 파일 경로 지정
# =========================

st.set_page_config(
    page_title="데메위키 사내규정 챗봇",
    page_icon="💬",
    layout="wide"
)

st.title("💬 데메위키 사내규정 챗봇")
st.caption("FAQ 파일을 근거로 가장 유사한 질문을 찾아 등록된 대답과 관련 규정을 안내합니다.")


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
# 기준 점수
# =========================

DIRECT_THRESHOLD = 0.80
CAUTION_THRESHOLD = 0.65


# =========================
# 텍스트 처리 함수
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

            required_columns = [
                "질문",
                "대답",
                "관련 규정",
                "관련 조항",
                "카테고리",
                "키워드",
                "업데이트 날짜"
            ]

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

    user_embedding = model.encode(
        [user_question],
        normalize_embeddings=True
    )

    semantic_scores = cosine_similarity(user_embedding, faq_embeddings)[0]

    results = []

    for idx, row in df.iterrows():
        semantic = float(semantic_scores[idx])
        keyword = keyword_score(user_question, row["키워드"])
        category = category_score(user_question, row["카테고리"])

        final_score = (
            semantic * 0.70
            + keyword * 0.20
            + category * 0.10
        )

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

    results = sorted(
        results,
        key=lambda x: x["final_score"],
        reverse=True
    )

    return results


def recommend_questions(results, top_n=2):
    sorted_results = sorted(
        results,
        key=lambda x: (
            x["keyword_score"],
            x["semantic_score"]
        ),
        reverse=True
    )
    return sorted_results[:top_n]


# =========================
# 실패 질문 저장
# =========================

def save_failed_question(user_question, best_match):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "날짜": now,
        "사용자 질문": user_question,
        "가장 유사한 FAQ": best_match["question"] if best_match else "",
        "최종 점수": round(best_match["final_score"], 4) if best_match else "",
        "의미 유사도": round(best_match["semantic_score"], 4) if best_match else "",
        "키워드 점수": round(best_match["keyword_score"], 4) if best_match else "",
        "카테고리 점수": round(best_match["category_score"], 4) if best_match else ""
    }

    log_df = pd.DataFrame([row])

    if FAILED_LOG_PATH.exists():
        old_df = pd.read_csv(FAILED_LOG_PATH, encoding="utf-8-sig")
        log_df = pd.concat([old_df, log_df], ignore_index=True)

    log_df.to_csv(FAILED_LOG_PATH, index=False, encoding="utf-8-sig")


# ====================================================================
# 💡 [해결 책] 세션 상태 기화 및 동적 제어 변수 선언
# ====================================================================
if "current_question" not in st.session_state:
    st.session_state["current_question"] = ""
if "search_results" not in st.session_state:
    st.session_state["search_results"] = None
if "selected_recommendation" not in st.session_state:
    st.session_state["selected_recommendation"] = None

# 데이터 기본 로드
df = load_faq()
if df is None:
    st.warning("data/faq.csv 파일을 확인해 주세요.")
    st.stop()

faq_questions = df["질문"].fillna("").astype(str).tolist()
faq_embeddings = build_faq_embeddings(faq_questions)


# 1. 채팅 입력창 (사용자가 새로운 질문을 치면 세션 상태를 업데이트)
user_input = st.chat_input("예: 시프티 오류로 재택근무 출근 등록을 못했는데 어떻게 하나요?")

if user_input:
    st.session_state["current_question"] = user_input
    results = find_best_faq(user_input, df, faq_embeddings)
    st.session_state["search_results"] = results
    st.session_state["selected_recommendation"] = None  # 새 질문 시 기존 선택 초기화


# 2. 화면 렌더링 파트 (세션 상태에 기억된 질문이 있을 때 동작)
if st.session_state["current_question"]:
    # 유저 말풍선 유지
    st.chat_message("user").markdown(st.session_state["current_question"])
    
    results = st.session_state["search_results"]
    best = results[0]
    
    with st.chat_message("assistant"):
        # 상황 A: 매칭 점수가 높을 때 바로 답변 노출
        if best["final_score"] >= DIRECT_THRESHOLD:
            st.success("등록된 FAQ에서 가장 유사한 답변을 찾았습니다.")
            st.markdown("### 💬 답변")
            st.write(best["answer"])

            policy_name = best["policy"]
            st.markdown(f"### 📄 관련 규정: **{policy_name}**")
            pdf_url = get_pdf_link(policy_name)
            if pdf_url:
                st.markdown(f"🔗 [**👉 [{policy_name}] 원문 PDF 열기 (새 탭)**]({pdf_url})")
            else:
                st.caption("⚠️ `static/` 폴더에 해당 규정 명칭의 PDF 파일이 없습니다.")

            st.markdown("### 📌 관련 조항")
            st.write(best["article"])

        # 상황 B: 매칭 점수가 애매할 때 유의 안내와 함께 노출
        elif best["final_score"] >= CAUTION_THRESHOLD:
            st.warning("가장 유사한 FAQ를 찾았지만, 질문 의도와 다를 수 있습니다.")
            st.markdown("### 💬 답변")
            st.write(best["answer"])

            policy_name = best["policy"]
            st.markdown(f"### 📄 관련 규정: **{policy_name}**")
            pdf_url = get_pdf_link(policy_name)
            if pdf_url:
                st.markdown(f"🔗 [**👉 [{policy_name}] 원문 PDF 열기 (새 탭)**]({pdf_url})")
            else:
                st.caption("⚠️ `static/` 폴더에 해당 규정 명칭의 PDF 파일이 없습니다.")

            st.markdown("### 📌 관련 조항")
            st.write(best["article"])

        # 상황 C: 아예 일치하는 게 없을 때 추천 리스트 버튼 제공
        else:
            st.error("등록된 파일에서 정확히 일치하는 답변을 찾지 못했습니다.")
            
            # 로그 기록은 처음 실패 시에만 수행하도록 유도 가능 (단순 기록 유지)
            st.markdown("🔍 **혹시 아래 질문을 찾으셨나요?**")
            
            recommendations = recommend_questions(results, top_n=2)
            
            # 추천 버튼 생성
            for idx, item in enumerate(recommendations, start=1):
                # 버튼 클릭 시 해당 추천 항목을 세션에 고정시키도록 callback 또는 내장 조건문 활용
                if st.button(f"🙋‍♂️ {item['question']}", key=f"btn_{idx}_{item['index']}"):
                    st.session_state["selected_recommendation"] = item
            
            # 선택된 추천 항목이 있다면 화면 리런 후에도 해당 정보 아래에 유지 고정
            if st.session_state["selected_recommendation"] is not None:
                selected_item = st.session_state["selected_recommendation"]
                
                st.markdown("---")
                st.info(f"👉 '{selected_item['question']}'에 대한 FAQ 답변입니다.")

                st.markdown("#### 💬 답변")
                st.info(selected_item["answer"])

                p_name = selected_item["policy"]
                st.markdown(f"#### 📄 관련 규정: **{p_name}**")
                
                pdf_url = get_pdf_link(p_name)
                if pdf_url:
                    st.markdown(f"🔗 [**👉 [{p_name}] 원문 PDF 열기 (새 탭)**]({pdf_url})")
                else:
                    st.caption("⚠️ `static/` 폴더에 해당 규정 명칭의 PDF 파일이 없습니다.")

                st.markdown("#### 📌 관련 조항")
                st.code(selected_item["article"] if selected_item["article"] else "지정된 조항 없음", language="text")