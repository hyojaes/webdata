# streamlit_app.py
# -*- coding: utf-8 -*-
import os
import textwrap
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ------------------------------
# 기본 설정
# ------------------------------
st.set_page_config(
    page_title="AI × Dev: 데이터 분석 대시보드 (KR v2)",
    layout="wide"
)

# Altair 테마 (Altair 5.5+)
try:
    alt.theme.enable("quartz")  # 'default' 또는 'quartz' 사용 가능
except Exception:
    pass

# ------------------------------
# 유틸 함수
# ------------------------------
@st.cache_data(show_spinner=False, ttl=0)  # 개발 중엔 TTL=0; 배포 시 제거/조정
def load_csv(path_or_file, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path_or_file, low_memory=False, **kwargs)

EXCLUDE_TOKENS = {"", "-", "nan", "None", "null"}

def normalize_multiselect_series(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str)
    s = s.str.replace(";", ",", regex=False)
    return s

def explode_multiselect(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """다중응답을 행으로 폭발. 빈값/'-' 제거."""
    if col not in df.columns:
        return pd.DataFrame(columns=[col])
    s = normalize_multiselect_series(df[col]).str.split(",")

    def clean(lst):
        out = []
        for x in (lst or []):
            x = str(x).strip()
            if x in EXCLUDE_TOKENS:
                continue
            out.append(x)
        return out

    s = s.apply(clean)
    exploded = df.assign(**{col: s}).explode(col)
    if col in exploded.columns:
        exploded[col] = exploded[col].astype(str).str.strip()
        exploded = exploded[~exploded[col].isin(EXCLUDE_TOKENS)]
    return exploded

def value_counts_pct(series: pd.Series) -> pd.DataFrame:
    """빈도/비율 표 생성. 비어도 컬럼 스켈레톤 유지."""
    s = series.replace({np.nan: None}).dropna().astype(str)
    s = s[~s.isin(list(EXCLUDE_TOKENS))]
    if s.empty:
        return pd.DataFrame(columns=["count", "percent"])
    vc = s.value_counts()
    pct = 100 * vc / vc.sum()
    return pd.DataFrame({"count": vc.astype(int), "percent": pct.round(2)})

def prep_for_chart(vc_df: pd.DataFrame, label_name: str) -> pd.DataFrame:
    """Altair 차트용 인덱스→컬럼 전개 + dtype 강제. 비어도 컬럼 보장."""
    if vc_df is None or len(vc_df) == 0:
        return pd.DataFrame(columns=[label_name, "count", "percent"])
    df = vc_df.reset_index()
    df = df.rename(columns={df.columns[0]: label_name})
    df[label_name] = df[label_name].astype(str)
    if "count" in df.columns:
        df["count"] = pd.to_numeric(df["count"], errors="coerce")
    if "percent" in df.columns:
        df["percent"] = pd.to_numeric(df["percent"], errors="coerce")
    return df

def wrap_label(s: str, width: int = 16) -> str:
    parts = textwrap.wrap(str(s), width=width)
    return "\n".join(parts) if parts else s

def safe_sort(df: pd.DataFrame, by: str, ascending=True) -> pd.DataFrame:
    if df is None or df.empty or by not in df.columns:
        return df
    return df.sort_values(by, ascending=ascending)

def require_file(path_or_file, label: str):
    """업로드 파일은 통과, 문자열 경로면 존재 확인 후 없으면 stop."""
    if hasattr(path_or_file, "read"):
        return
    if isinstance(path_or_file, str) and not os.path.exists(path_or_file):
        st.error(f"{label} 파일을 찾을 수 없습니다: {path_or_file}\n사이드바에서 CSV 업로드 또는 경로를 확인하세요.")
        st.stop()

# ------------------------------
# 한글 매핑
# ------------------------------
TOOLS_MAP = {
    "Chatbots": "챗봇",
    "Predictive Analytics": "예측 분석",
    "Machine Learning Algorithms": "머신러닝 알고리즘",
    "Natural Language Processing": "자연어 처리",
    "Natural Language Processing (NLP)": "자연어 처리",
    "NLP": "자연어 처리",
    "ML Algorithms": "머신러닝 알고리즘",
    "GenAI tools": "생성형 AI 도구",
    "Computer Vision": "컴퓨터 비전",
    "Recommendation Systems": "추천 시스템",
    "RPA": "RPA(로봇 프로세스 자동화)",
}

BENEFITS_MAP = {
    "Increased productivity": "생산성 증가",
    "Faster prototyping": "빠른 프로토타이핑",
    "Improved documentation": "문서화 향상",
    "Better decision-making": "의사결정 품질 향상",
    "Automation of repetitive tasks": "반복 작업 자동화",
    "Enhanced code quality": "코드 품질 향상",
}

CHALLENGE_MAP = {
    "Ethical considerations": "윤리적 고려",
    "Cost implications": "비용 부담",
    "Lack of expertise": "전문성 부족",
    "Lack of expertise in AI technologies": "전문성 부족",
    "Resistance from team": "팀 내 저항",
    "Resistance from team members": "팀 내 저항",
    "Integration complexity": "시스템 통합 복잡성",
    "Integration complexities": "시스템 통합 복잡성",
    "Data privacy concerns": "데이터 프라이버시 우려",
    "Quality/accuracy concerns": "품질·정확도 우려",
    "Compliance/regulatory issues": "규제·컴플라이언스 이슈",
}

DEVTYPE_MAP = {
    "Developer, back-end": "백엔드 개발자",
    "Developer, front-end": "프론트엔드 개발자",
    "Developer, full-stack": "풀스택 개발자",
    "Developer, mobile": "모바일 개발자",
    "Developer, desktop or enterprise applications": "데스크톱/엔터프라이즈 개발자",
    "Developer, embedded applications or devices": "임베디드 개발자",
    "Developer, game or graphics": "게임/그래픽스 개발자",
    "Data scientist or machine learning specialist": "데이터 사이언티스트/ML 전문가",
    "Data or business analyst": "데이터/비즈니스 분석가",
    "Database administrator": "DB 관리자",
    "DevOps specialist": "데브옵스 전문가",
    "Security professional": "보안 전문가",
    "Engineering manager": "엔지니어링 매니저",
    "Academic researcher": "학술 연구자",
    "Scientist": "과학자",
    "Student": "학생",
    "System administrator": "시스템 관리자",
    "Cloud infrastructure engineer": "클라우드 인프라 엔지니어",
    "Site reliability engineer": "SRE",
}

# ------------------------------
# 사이드바 / 경로
# ------------------------------
st.sidebar.header("📁 데이터 소스 설정")

DEFAULT_SO = "data/survey_results_public.csv"
DEFAULT_AI = "data/Survey on Integrating Artificial Intelligence Tools within Agile Frameworks for Enhanced Software Development (Responses) - Sheet1.csv"

dataset = st.sidebar.radio(
    "분석 대상 선택",
    ["AI in Agile 설문", "Stack Overflow 2023"],
    index=0,
    key="dataset_radio"
)

st.sidebar.markdown("---")

# ============================================================
# AI in Agile 설문
# ============================================================
if dataset == "AI in Agile 설문":
    # --------------------------
    # 타이틀 & 설명 (tips 스타일)
    # --------------------------
    st.title("💡 AI in Agile 설문 대시보드")
    st.write("")
    st.write("")
    st.caption("소프트웨어 개발에서 AI 도구 사용 경험, 도입 의향, 기대 효과·우려, 코딩 속도 향상을 시각적으로 탐색하는 대시보드입니다.")

    # --------------------------
    # 데이터 업로드
    # --------------------------
    up = st.sidebar.file_uploader("AI 설문 CSV 업로드", type=["csv"], key="ai_csv")
    path = up if up is not None else DEFAULT_AI
    require_file(path, "AI 설문")
    ai = load_csv(path)

    mapper = {
        'Current Role: ': 'Role',
        'Familiarity with Agile Frameworks:': 'AgileFamiliarity',
        'Familiarity with Artificial Intelligence Tools(Like ChatGPT ):': 'AIFamiliarity',
        'Have you used artificial intelligence tools in software development projects before?': 'AIUsedBefore',
        'If yes, please specify the types of artificial intelligence tools you have used (check all that apply):': 'AIToolsUsed',
        'How do you perceive the potential benefits of integrating AI...e frameworks for software development? (Check all that apply):': 'Benefits',
        'What challenges do you foresee in integrating AI tools within agile frameworks? (Check all that apply):': 'Challenges',
        'On a scale of 1 to 5, how willing would you be to adopt AI tools within your agile development processes?': 'Willingness',
        'AI sped up coding': 'AISpeedUp',
    }
    ai = ai.rename(columns={k: v for k, v in mapper.items() if k in ai.columns})

    # 결측치 정리
    for col in [
        "Role",
        "AgileFamiliarity",
        "AIFamiliarity",
        "AIUsedBefore",
        "AIToolsUsed",
        "Benefits",
        "Challenges",
        "AISpeedUp",
    ]:
        if col in ai.columns:
            ai[col] = ai[col].fillna("")

    if "Willingness" in ai.columns:
        ai["Willingness"] = pd.to_numeric(ai["Willingness"], errors="coerce")

    # --------------------------
    # 데이터셋 소개 (tips 스타일)
    # --------------------------
    st.subheader("📋 설문 문항 소개")

    with st.expander("AI in Agile 설문 항목 설명"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("- **Role**: 현재 역할 (개발자, 학생, 매니저 등)")
            st.markdown("- **AgileFamiliarity**: 애자일 방법론에 대한 친숙도")
            st.markdown("- **AIFamiliarity**: AI 도구(예: ChatGPT)에 대한 친숙도")
            st.markdown("- **AIUsedBefore**: 개발 프로젝트에서 AI 도구 사용 여부")
        with col2:
            st.markdown("- **AIToolsUsed**: 사용해본 AI 도구 유형(다중 선택)")
            st.markdown("- **Benefits**: AI 도입 기대 효과(다중 선택)")
            st.markdown("- **Challenges**: AI 도입 시 우려·장애요인(다중 선택)")
            st.markdown("- **Willingness (1~5)**: AI 도입 의향 점수")
            st.markdown("- **AISpeedUp**: AI가 코딩 속도를 높여준다고 느꼈는지 여부")

        st.markdown("💡 해당 항목들을 기반으로 **AI 도입 준비도**와 **개발 문화 변화**를 분석할 수 있습니다.")

    # --------------------------
    # 사이드바: 데이터 필터
    # --------------------------
    st.sidebar.subheader("🔍 데이터 필터")

    roles = sorted(
        [r for r in ai.get("Role", pd.Series(dtype=str)).dropna().unique() if r]
    ) if "Role" in ai.columns else []
    fams = sorted(
        [r for r in ai.get("AIFamiliarity", pd.Series(dtype=str)).dropna().unique() if r]
    ) if "AIFamiliarity" in ai.columns else []

    sel_roles = st.sidebar.multiselect("역할(Role)", roles, default=[], key="roles_ms")
    sel_fams = st.sidebar.multiselect("AI 도구 친숙도", fams, default=[], key="fams_ms")

    min_will = None
    if "Willingness" in ai.columns:
        min_will = st.sidebar.slider(
            "최소 도입 의향 점수 (1~5)", 1, 5, 1, step=1, key="min_will_ai"
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 시각화 옵션")
    show_ai_used = st.sidebar.checkbox("AI 사용 경험 분포", True, key="show_ai_used")
    show_ai_tools = st.sidebar.checkbox("사용한 AI 도구 유형", True, key="show_ai_tools")
    show_benefits = st.sidebar.checkbox("기대 효과 분포", True, key="show_benefits")
    show_challenges = st.sidebar.checkbox("우려·장애요인 분포", True, key="show_challenges")
    show_willingness = st.sidebar.checkbox("도입 의향 분포", True, key="show_will")
    show_speedup = st.sidebar.checkbox("코딩 속도 향상 여부", True, key="show_speed")
    show_speed_vs_will = st.sidebar.checkbox("도입 의향 vs 속도 향상", True, key="show_speed_vs_will")

    # --------------------------
    # 필터 적용
    # --------------------------
    df_f = ai.copy()
    if sel_roles and "Role" in df_f.columns:
        df_f = df_f[df_f["Role"].isin(sel_roles)]
    if sel_fams and "AIFamiliarity" in df_f.columns:
        df_f = df_f[df_f["AIFamiliarity"].isin(sel_fams)]
    if min_will is not None and "Willingness" in df_f.columns:
        df_f = df_f[df_f["Willingness"].fillna(0) >= min_will]

    # --------------------------
    # 필터링된 데이터 (tips 스타일)
    # --------------------------
    st.write("")
    with st.expander(f"📊 필터링된 데이터 (총 {df_f.shape[0]}개 행)"):
        st.dataframe(df_f)

    st.write("")
    st.write("")

    # --------------------------
    # 요약 통계 (Metrics, tips 스타일)
    # --------------------------
    st.subheader("📈 요약 통계")

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        if "AIUsedBefore" in df_f.columns and len(df_f) > 0:
            used_rate = (
                df_f["AIUsedBefore"]
                .astype(str)
                .str.lower()
                .isin(["yes", "y", "true", "1"])
                .mean()
                * 100
            )
            st.metric("AI 사용 경험률(%)", f"{used_rate:.1f}")
        else:
            st.metric("AI 사용 경험률(%)", "N/A")

    with k2:
        if "Willingness" in df_f.columns and df_f["Willingness"].notna().any():
            st.metric("평균 도입 의향(1~5)", f"{df_f['Willingness'].mean():.2f}")
        else:
            st.metric("평균 도입 의향(1~5)", "N/A")

    with k3:
        if "Willingness" in df_f.columns and df_f["Willingness"].notna().any():
            st.metric("도입 의향 중앙값", f"{df_f['Willingness'].median():.2f}")
        else:
            st.metric("도입 의향 중앙값", "N/A")

    with k4:
        if "AISpeedUp" in df_f.columns and len(df_f) > 0:
            speed_rate = (
                df_f["AISpeedUp"]
                .astype(str)
                .str.lower()
                .isin(["yes", "y", "true", "1"])
                .mean()
                * 100
            )
            st.metric("코딩 속도 향상 체감률(%)", f"{speed_rate:.1f}")
        else:
            st.metric("코딩 속도 향상 체감률(%)", "N/A")

    st.write("")
    st.write("")

    # --------------------------
    # 시각화 (tips 구조 맞춤)
    # --------------------------
    st.subheader("📉 시각화")

    # 1) AI 사용 경험
    if show_ai_used:
        st.markdown("#### 1) AI 사용 경험 분포")
        if "AIUsedBefore" in df_f.columns:
            vc = value_counts_pct(df_f["AIUsedBefore"])
            d = prep_for_chart(vc, "AIUsedBefore")
            if not d.empty:
                base = alt.Chart(d).properties(height=200)
                bars = base.mark_bar().encode(
                    y=alt.Y("AIUsedBefore:N", sort="-x", title="경험 여부"),
                    x=alt.X("count:Q", title="응답 수"),
                    tooltip=["AIUsedBefore", "count", "percent"],
                )
                texts = base.mark_text(align="left", baseline="middle", dx=4).encode(
                    y=alt.Y("AIUsedBefore:N", sort="-x"),
                    x=alt.X("count:Q"),
                    text="count:Q",
                )
                st.altair_chart(bars + texts, use_container_width=True)
            st.dataframe(
                d.rename(
                    columns={"AIUsedBefore": "경험 여부", "count": "응답 수", "percent": "비율(%)"}
                )
            )

    # 2) 사용한 AI 도구 유형
    if show_ai_tools:
        st.markdown("#### 2) 사용한 AI 도구 유형")
        if "AIToolsUsed" in df_f.columns:
            tools = explode_multiselect(df_f, "AIToolsUsed")
            if len(tools) > 0:
                t = tools["AIToolsUsed"].map(TOOLS_MAP).fillna(tools["AIToolsUsed"])
                vc2 = value_counts_pct(t)
                d2 = prep_for_chart(vc2, "Tool_ko")
                d2["Tool_ko_wrapped"] = d2["Tool_ko"].apply(lambda s: wrap_label(s, 16))

                if not d2.empty:
                    base2 = alt.Chart(d2).properties(height=max(220, 28 * len(d2)))
                    bars2 = base2.mark_bar().encode(
                        y=alt.Y("Tool_ko_wrapped:N", sort="-x", title="도구"),
                        x=alt.X("count:Q", title="응답 수"),
                        tooltip=["Tool_ko", "count", "percent"],
                    )
                    texts2 = base2.mark_text(align="left", baseline="middle", dx=4).encode(
                        y=alt.Y("Tool_ko_wrapped:N", sort="-x"),
                        x=alt.X("count:Q"),
                        text="count:Q",
                    )
                    st.altair_chart(bars2 + texts2, use_container_width=True)

                tbl2 = d2[["Tool_ko", "count", "percent"]].rename(
                    columns={"Tool_ko": "도구", "count": "응답 수", "percent": "비율(%)"}
                )
                st.dataframe(safe_sort(tbl2, "응답 수", ascending=False))

    # 3) 기대 효과
    if show_benefits:
        st.markdown("#### 3) 기대 효과(Benefits)")
        if "Benefits" in df_f.columns:
            ben = explode_multiselect(df_f, "Benefits")
            if len(ben) > 0:
                ben["Benefits"] = ben["Benefits"].astype(str).str.strip()
                ben = ben[~ben["Benefits"].isin(EXCLUDE_TOKENS)]
                ben_ko = ben["Benefits"].map(BENEFITS_MAP).fillna(ben["Benefits"])

                vc3 = value_counts_pct(ben_ko)
                d3 = prep_for_chart(vc3, "Benefit_ko")
                d3["Benefit_ko_wrapped"] = d3["Benefit_ko"].apply(
                    lambda s: wrap_label(s, 16)
                )

                if not d3.empty:
                    base3 = alt.Chart(d3).properties(height=max(220, 28 * len(d3)))
                    bars3 = base3.mark_bar().encode(
                        y=alt.Y("Benefit_ko_wrapped:N", sort="-x", title="기대 효과"),
                        x=alt.X("count:Q", title="응답 수"),
                        tooltip=["Benefit_ko", "count", "percent"],
                    )
                    texts3 = base3.mark_text(align="left", baseline="middle", dx=4).encode(
                        y=alt.Y("Benefit_ko_wrapped:N", sort="-x"),
                        x=alt.X("count:Q"),
                        text="count:Q",
                    )
                    st.altair_chart(bars3 + texts3, use_container_width=True)

                st.dataframe(
                    d3[["Benefit_ko", "count", "percent"]].rename(
                        columns={
                            "Benefit_ko": "기대 효과(한글)",
                            "count": "응답 수",
                            "percent": "비율(%)",
                        }
                    )
                )

    # 4) 우려/장애요인
    if show_challenges:
        st.markdown("#### 4) 우려/장애요인(Challenges)")
        if "Challenges" in df_f.columns:
            ch = explode_multiselect(df_f, "Challenges")
            if len(ch) > 0:
                ch["Challenges"] = ch["Challenges"].astype(str).str.strip()
                ch = ch[~ch["Challenges"].isin(EXCLUDE_TOKENS)]
                ch_ko = ch["Challenges"].map(CHALLENGE_MAP).fillna(ch["Challenges"])

                vc4 = value_counts_pct(ch_ko)
                d4 = prep_for_chart(vc4, "Challenge_ko")
                d4["Challenge_ko_wrapped"] = d4["Challenge_ko"].apply(
                    lambda s: wrap_label(s, 16)
                )

                if not d4.empty:
                    base4 = alt.Chart(d4).properties(height=max(220, 28 * len(d4)))
                    bars4 = base4.mark_bar().encode(
                        y=alt.Y("Challenge_ko_wrapped:N", sort="-x", title="장애요인"),
                        x=alt.X("count:Q", title="응답 수"),
                        tooltip=["Challenge_ko", "count", "percent"],
                    )
                    texts4 = base4.mark_text(align="left", baseline="middle", dx=4).encode(
                        y=alt.Y("Challenge_ko_wrapped:N", sort="-x"),
                        x=alt.X("count:Q"),
                        text="count:Q",
                    )
                    st.altair_chart(bars4 + texts4, use_container_width=True)

                tbl4 = d4[["Challenge_ko", "count", "percent"]].rename(
                    columns={"Challenge_ko": "장애요인(한글)", "count": "응답 수", "percent": "비율(%)"}
                )
                st.dataframe(safe_sort(tbl4, "응답 수", ascending=False))

    # 5) 도입 의향 분포
    if show_willingness:
        st.markdown("#### 5) 도입 의향 분포 (1=낮음, 5=매우 높음)")
        if "Willingness" in df_f.columns and df_f["Willingness"].notna().any():
            vc5 = value_counts_pct(df_f["Willingness"].dropna())
            d5 = prep_for_chart(vc5, "Score")

            if not d5.empty and "Score" in d5.columns:
                chart5 = alt.Chart(d5).mark_bar().encode(
                    x=alt.X("Score:O", title="도입 의향 점수"),
                    y=alt.Y("count:Q", title="응답 수"),
                    tooltip=["Score", "count", "percent"],
                )
                st.altair_chart(chart5, use_container_width=True)

                tbl5 = d5.rename(
                    columns={"Score": "점수", "count": "응답 수", "percent": "비율(%)"}
                )
                st.dataframe(safe_sort(tbl5, "점수"))
            else:
                st.info("유효한 도입 의향 데이터가 없습니다.")
        else:
            st.info("도입 의향(Willingness) 컬럼이 없거나 값이 없습니다.")

    # 6) AI 사용 후 코딩 속도 향상 여부
    if show_speedup:
        st.markdown("#### 6) AI 사용 후 코딩 속도 향상 여부")
        if "AISpeedUp" in df_f.columns:
            vc6 = value_counts_pct(df_f["AISpeedUp"])
            d6 = prep_for_chart(vc6, "AISpeedUp")

            if not d6.empty:
                base6 = alt.Chart(d6).properties(height=200)
                bars6 = base6.mark_bar().encode(
                    y=alt.Y("AISpeedUp:N", sort="-x", title="코딩 속도 향상 여부"),
                    x=alt.X("count:Q", title="응답 수"),
                    tooltip=["AISpeedUp", "count", "percent"],
                )
                texts6 = base6.mark_text(align="left", baseline="middle", dx=4).encode(
                    y=alt.Y("AISpeedUp:N", sort="-x"),
                    x=alt.X("count:Q"),
                    text="count:Q",
                )
                st.altair_chart(bars6 + texts6, use_container_width=True)

            st.dataframe(
                d6.rename(
                    columns={"AISpeedUp": "코딩 속도 향상 여부", "count": "응답 수", "percent": "비율(%)"}
                )
            )
        else:
            st.info("'AI sped up coding' 컬럼(AISpeedUp)이 데이터에 없습니다.")

    # 7) 도입 의향 vs 코딩 속도 향상
    if show_speed_vs_will:
        st.markdown("#### 7) 도입 의향 vs 코딩 속도 향상 관계")
        if "Willingness" in df_f.columns and "AISpeedUp" in df_f.columns:
            tmp = df_f.copy()
            tmp["AISpeedUpLabel"] = (
                tmp["AISpeedUp"]
                .astype(str)
                .str.lower()
                .map(
                    {
                        "yes": "향상 느꼈음",
                        "y": "향상 느꼈음",
                        "true": "향상 느꼈음",
                        "1": "향상 느꼈음",
                    }
                )
            )
            tmp["AISpeedUpLabel"] = tmp["AISpeedUpLabel"].fillna("향상 못 느낌/미응답")
            tmp = tmp[tmp["Willingness"].notna()]
            if not tmp.empty:
                grp = (
                    tmp.groupby("AISpeedUpLabel")["Willingness"]
                    .agg(["mean", "count"])
                    .reset_index()
                )
                grp["mean"] = grp["mean"].round(2)
                chart7 = alt.Chart(grp).mark_bar().encode(
                    x=alt.X("AISpeedUpLabel:N", title="코딩 속도 향상 인식"),
                    y=alt.Y("mean:Q", title="평균 도입 의향 점수"),
                    tooltip=["AISpeedUpLabel", "mean", "count"],
                )
                st.altair_chart(chart7, use_container_width=True)
                st.dataframe(
                    grp.rename(
                        columns={
                            "AISpeedUpLabel": "코딩 속도 향상 인식",
                            "mean": "평균 도입 의향",
                            "count": "응답 수",
                        }
                    )
                )
            else:
                st.info("도입 의향과 속도 향상 응답이 모두 있는 데이터가 없습니다.")
        else:
            st.info("Willingness 또는 AISpeedUp 컬럼이 없습니다.")

    # --------------------------
    # CSV 다운로드
    # --------------------------
    st.download_button(
        "🔽 현재 필터 결과 CSV 다운로드",
        data=df_f.to_csv(index=False).encode("utf-8-sig"),
        file_name="ai_agile_filtered.csv",
        mime="text/csv",
    )

# ============================================================
# Stack Overflow 2023
# ============================================================
else:
    # --------------------------
    # 타이틀 & 설명
    # --------------------------
    st.title("💡 Stack Overflow 2023 개발자 설문 대시보드")
    st.write("")
    st.write("")
    st.caption("개발자 직무, 사용 언어, 경력, 국가, 조직 규모 분포를 탐색하는 대시보드입니다.")

    # 데이터 업로드
    up = st.sidebar.file_uploader("SO 2023 CSV 업로드", type=["csv"], key="so_csv")
    path = up if up is not None else DEFAULT_SO
    require_file(path, "SO 2023 데이터")
    so = load_csv(path)

    # 데이터셋 소개
    st.subheader("📋 설문 문항 소개")
    with st.expander("Stack Overflow 2023 주요 컬럼 설명"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("- **DevType**: 개발자 직무 유형(백엔드, 프론트엔드, 데이터 등)")
            st.markdown("- **LanguageHaveWorkedWith**: 사용해본 프로그래밍 언어 목록")
            st.markdown("- **YearsCodePro**: 프로 개발 경력(년)")
        with c2:
            st.markdown("- **Country**: 거주 국가")
            st.markdown("- **OrgSize**: 현재 조직 규모")
            st.markdown("- 기타 컬럼: 근무 형태, 학력, 사용 도구 등")

        st.markdown("💡 직무와 언어, 경력, 조직 규모를 함께 보면 **개발 생태계의 구조**를 파악할 수 있습니다.")

    # Sidebar 필터
    st.sidebar.subheader("🔍 데이터 필터")

    countries = sorted(
        [c for c in so.get("Country", pd.Series(dtype=str)).dropna().unique() if c]
    ) if "Country" in so.columns else []
    sel_countries = st.sidebar.multiselect(
        "국가", countries, default=[], key="country_ms"
    )

    orgs = sorted(
        [o for o in so.get("OrgSize", pd.Series(dtype=str)).dropna().unique() if o]
    ) if "OrgSize" in so.columns else []
    sel_orgs = st.sidebar.multiselect(
        "조직 규모", orgs, default=[], key="org_ms"
    )

    min_years = None
    if "YearsCodePro" in so.columns:
        min_years = st.sidebar.slider(
            "최소 경력(년)", 0, 50, 0, step=1, key="min_years_so"
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 시각화 옵션")
    show_devtype = st.sidebar.checkbox("직무(DevType) 분포", True, key="show_devtype")
    show_lang = st.sidebar.checkbox("사용 언어 분포", True, key="show_lang")
    show_years = st.sidebar.checkbox("경력(YearsCodePro) 분포", True, key="show_years")
    show_country = st.sidebar.checkbox("국가 분포(Country)", True, key="show_country")
    show_orgsize = st.sidebar.checkbox("조직 규모 분포(OrgSize)", True, key="show_orgsize")

    # 필터 적용
    so_f = so.copy()
    if sel_countries and "Country" in so_f.columns:
        so_f = so_f[so_f["Country"].isin(sel_countries)]
    if sel_orgs and "OrgSize" in so_f.columns:
        so_f = so_f[so_f["OrgSize"].isin(sel_orgs)]
    if min_years is not None and "YearsCodePro" in so_f.columns:
        y = so_f["YearsCodePro"].replace(
            {"Less than 1 year": "0", "More than 50 years": "51"}
        )
        y_num = pd.to_numeric(y, errors="coerce")
        so_f = so_f[y_num >= min_years]

    # 필터링된 데이터
    st.write("")
    with st.expander(f"📊 필터링된 데이터 (총 {so_f.shape[0]}개 행)"):
        st.dataframe(so_f.head(200))

    st.write("")
    st.write("")

    # 요약 통계 (간단 KPI)
    st.subheader("📈 요약 통계")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("응답자 수", f"{so_f.shape[0]}")
    with c2:
        if "Country" in so_f.columns:
            st.metric("국가 수", f"{so_f['Country'].nunique()}")
        else:
            st.metric("국가 수", "N/A")
    with c3:
        if "LanguageHaveWorkedWith" in so_f.columns:
            lang_tmp = so_f["LanguageHaveWorkedWith"].fillna("").str.split(";").explode().str.strip()
            lang_tmp = lang_tmp[~lang_tmp.isin(EXCLUDE_TOKENS)]
            st.metric("사용 언어 수", f"{lang_tmp.nunique()}")
        else:
            st.metric("사용 언어 수", "N/A")
    with c4:
        if "YearsCodePro" in so_f.columns:
            y = so_f["YearsCodePro"].replace(
                {"Less than 1 year": "0", "More than 50 years": "51"}
            )
            y_num = pd.to_numeric(y, errors="coerce")
            if y_num.notna().any():
                st.metric("평균 경력(년)", f"{y_num.mean():.1f}")
            else:
                st.metric("평균 경력(년)", "N/A")
        else:
            st.metric("평균 경력(년)", "N/A")

    st.write("")
    st.write("")

    # 시각화 영역
    st.subheader("📉 시각화")

    # 1) DevType
    if show_devtype:
        st.markdown("#### 1) 직무(DevType) 분포")
        if "DevType" in so_f.columns:
            dev = so_f["DevType"].fillna("").str.split(";").explode().str.strip()
            dev = dev[~dev.isin(EXCLUDE_TOKENS)]
            if len(dev) > 0:
                dev_ko = dev.map(DEVTYPE_MAP).fillna(dev)
                vc = value_counts_pct(dev_ko)
                d = prep_for_chart(vc, "DevType_ko")
                d["DevType_ko_wrapped"] = d["DevType_ko"].apply(lambda s: wrap_label(s, 16))

                if not d.empty:
                    base = alt.Chart(d).properties(height=max(220, 28 * len(d)))
                    bars = base.mark_bar().encode(
                        y=alt.Y("DevType_ko_wrapped:N", sort="-x", title="직무"),
                        x=alt.X("count:Q", title="응답 수"),
                        tooltip=["DevType_ko", "count", "percent"],
                    )
                    texts = base.mark_text(align="left", baseline="middle", dx=4).encode(
                        y=alt.Y("DevType_ko_wrapped:N", sort="-x"),
                        x=alt.X("count:Q"),
                        text="count:Q",
                    )
                    st.altair_chart(bars + texts, use_container_width=True)

                tbl_dev = d.rename(
                    columns={"DevType_ko": "직무", "count": "응답 수", "percent": "비율(%)"}
                )
                st.dataframe(safe_sort(tbl_dev, "응답 수", ascending=False))

    # 2) LanguageHaveWorkedWith
    if show_lang:
        st.markdown("#### 2) 사용 언어(LanguageHaveWorkedWith)")
        if "LanguageHaveWorkedWith" in so_f.columns:
            lang = so_f["LanguageHaveWorkedWith"].fillna("").str.split(";").explode().str.strip()
            lang = lang[~lang.isin(EXCLUDE_TOKENS)]
            if len(lang) > 0:
                vc2 = value_counts_pct(lang)
                d2 = prep_for_chart(vc2, "Language")
                d2["Language_wrapped"] = d2["Language"].apply(lambda s: wrap_label(s, 16))

                if not d2.empty:
                    base2 = alt.Chart(d2).properties(height=max(220, 28 * len(d2)))
                    bars2 = base2.mark_bar().encode(
                        y=alt.Y("Language_wrapped:N", sort="-x", title="언어"),
                        x=alt.X("count:Q", title="응답 수"),
                        tooltip=["Language", "count", "percent"],
                    )
                    texts2 = base2.mark_text(align="left", baseline="middle", dx=4).encode(
                        y=alt.Y("Language_wrapped:N", sort="-x"),
                        x=alt.X("count:Q"),
                        text="count:Q",
                    )
                    st.altair_chart(bars2 + texts2, use_container_width=True)

                st.dataframe(
                    d2.rename(
                        columns={"Language": "언어", "count": "응답 수", "percent": "비율(%)"}
                    ).pipe(lambda x: safe_sort(x, "응답 수", ascending=False))
                )

    # 3) YearsCodePro
    if show_years:
        st.markdown("#### 3) 경력(YearsCodePro) 분포")
        if "YearsCodePro" in so_f.columns:
            y = so_f["YearsCodePro"].replace(
                {"Less than 1 year": "0", "More than 50 years": "51"}
            )
            y_num = pd.to_numeric(y, errors="coerce").dropna()
            vc3 = value_counts_pct(y_num)
            d3 = prep_for_chart(vc3, "Years")
            if not d3.empty:
                chart3 = alt.Chart(d3).mark_bar().encode(
                    x=alt.X("Years:Q", title="경력(년)"),
                    y=alt.Y("count:Q", title="응답 수"),
                    tooltip=["Years", "count", "percent"],
                )
                st.altair_chart(chart3, use_container_width=True)
            st.dataframe(
                d3.rename(
                    columns={"Years": "경력(년)", "count": "응답 수", "percent": "비율(%)"}
                ).pipe(lambda x: safe_sort(x, "Years"))
            )

    # 4) 국가 분포
    if show_country:
        st.markdown("#### 4) 국가 분포(Country)")
        if "Country" in so_f.columns:
            vc_c = value_counts_pct(so_f["Country"])
            dc = prep_for_chart(vc_c, "Country")
            dc["Country_wrapped"] = dc["Country"].apply(lambda s: wrap_label(s, 18))
            if not dc.empty:
                base_c = alt.Chart(dc).properties(height=max(220, 26 * len(dc)))
                bars_c = base_c.mark_bar().encode(
                    y=alt.Y("Country_wrapped:N", sort="-x", title="국가"),
                    x=alt.X("count:Q", title="응답 수"),
                    tooltip=["Country", "count", "percent"],
                )
                texts_c = base_c.mark_text(align="left", baseline="middle", dx=4).encode(
                    y=alt.Y("Country_wrapped:N", sort="-x"),
                    x=alt.X("count:Q"),
                    text="count:Q",
                )
                st.altair_chart(bars_c + texts_c, use_container_width=True)
            st.dataframe(
                dc.rename(
                    columns={"Country": "국가", "count": "응답 수", "percent": "비율(%)"}
                ).pipe(lambda x: safe_sort(x, "응답 수", ascending=False))
            )

    # 5) 조직 규모 분포
    if show_orgsize:
        st.markdown("#### 5) 조직 규모 분포(OrgSize)")
        if "OrgSize" in so_f.columns:
            vc_o = value_counts_pct(so_f["OrgSize"])
            do = prep_for_chart(vc_o, "OrgSize")
            do["OrgSize_wrapped"] = do["OrgSize"].apply(lambda s: wrap_label(s, 24))
            if not do.empty:
                base_o = alt.Chart(do).properties(height=max(220, 26 * len(do)))
                bars_o = base_o.mark_bar().encode(
                    y=alt.Y("OrgSize_wrapped:N", sort="-x", title="조직 규모"),
                    x=alt.X("count:Q", title="응답 수"),
                    tooltip=["OrgSize", "count", "percent"],
                )
                texts_o = base_o.mark_text(align="left", baseline="middle", dx=4).encode(
                    y=alt.Y("OrgSize_wrapped:N", sort="-x"),
                    x=alt.X("count:Q"),
                    text="count:Q",
                )
                st.altair_chart(bars_o + texts_o, use_container_width=True)
            st.dataframe(
                do.rename(
                    columns={"OrgSize": "조직 규모", "count": "응답 수", "percent": "비율(%)"}
                ).pipe(lambda x: safe_sort(x, "응답 수", ascending=False))
            )

    # CSV 다운로드
    st.download_button(
        "🔽 현재 필터 결과 CSV 다운로드",
        data=so_f.to_csv(index=False).encode("utf-8-sig"),
        file_name="so2023_filtered.csv",
        mime="text/csv",
    )
