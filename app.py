import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, auc, roc_curve, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.datasets import load_iris, load_wine, fetch_california_housing
import warnings
warnings.filterwarnings("ignore")

# ----------------------
# 1. 페이지 기본 설정
# ----------------------
st.set_page_config(
    page_title="하이브리드모형 동적 프레임워크（의사결정나무+회귀분석）",
    page_icon="📊",
    layout="wide"
)

# 全局状态管理（默认步骤设为「数据上传」步骤，索引1）
if "step" not in st.session_state:
    st.session_state.step = 1  # 0:초기설정 1:데이터업로드（默认） 2:데이터시각화 3:데이터전처리 4:모델학습 5:예측 6:평가
if "data" not in st.session_state:
    st.session_state.data = {"merged": None, "is_sample": False, "discretized_cols": None}
if "preprocess" not in st.session_state:
    st.session_state.preprocess = {"imputer": None, "scaler": None, "encoders": None, "feature_cols": None, "target_col": None}
if "models" not in st.session_state:
    st.session_state.models = {"regression": None, "decision_tree": None, "mixed_weights": {"regression": 0.3, "decision_tree": 0.7}}
if "task" not in st.session_state:
    st.session_state.task = "logit"

# ----------------------
# 2. 사이드바：단계导航（保留所有步骤，默认选中数据上传）
# ----------------------
st.sidebar.title("📌 하이브리드모형 작업 흐름")
st.sidebar.divider()

# 完整步骤列表（保留数据上传步骤）
steps = ["초기 설정", "데이터 업로드", "데이터 시각화", "데이터 전처리", "모델 학습", "모델 예측", "성능 평가"]
for i, step_name in enumerate(steps):
    # 默认选中数据上传步骤（按钮高亮）
    is_default = (i == 1) and (st.session_state.step == 1)
    btn_kwargs = {"key": f"btn_{i}"}
    if is_default:
        btn_kwargs["type"] = "primary"  # 默认步骤按钮高亮
    
    if st.sidebar.button(step_name, **btn_kwargs):
        st.session_state.step = i

#  핵심 설정（작업 유형 + 혼합 가중치）
st.sidebar.divider()
st.sidebar.subheader("핵심 설정")
st.session_state.task = st.sidebar.radio("작업 유형", options=["logit", "의사결정나무"], index=0)

if st.session_state.step >= 4:  # 모델 학습后 가중치 조정
    st.sidebar.subheader("하이브리드모형 가중치")
    reg_weight = st.sidebar.slider(
        "회귀 분석 가중치（해석력 강함）",
        min_value=0.0, max_value=1.0, value=st.session_state.models["mixed_weights"]["regression"], step=0.1
    )
    st.session_state.models["mixed_weights"]["regression"] = reg_weight
    st.session_state.models["mixed_weights"]["decision_tree"] = 1 - reg_weight
    st.sidebar.text(f"의사결정나무 가중치（정확도 높음）：{1 - reg_weight:.1f}")

# ----------------------
# 3. 메인 페이지：단계별内容（修复rerun错误，移除timeout参数）
# ----------------------
st.title("📊 하이브리드모형 동적 배포 프레임워크")
st.markdown("**데이터 선택后 바로 시각화부터 진행，예제 데이터或自有 데이터로 전과정을 완성**")
st.markdown("### 🧩 핵심 모델：회귀 분석（Regression）+ 의사결정나무（Decision Tree）")
st.divider()

# ----------------------
#  단계 0：초기 설정（안내 페이지）
# ----------------------
if st.session_state.step == 0:
    st.subheader("🎉 하이브리드모형 동적 프레임워크에 오신 것을 환영합니다")
    st.markdown("""
    본 프레임워크는 **데이터 업로드 단계에서 데이터를 선택**（예제 데이터 또는 자신의 데이터 업로드）하여 사용할 수 있으며，사전 전처리나 모델 학습이 필요 없습니다. 핵심 흐름은 다음과 같습니다：
    
    1. **데이터 선택**：데이터 업로드 단계에서 예제 데이터 사용 또는 자신의 데이터 업로드
    2. **데이터 시각화**：범주형 변수 또는离散化된 수치형 변수를 선택하여 다양한 그래프로 데이터 탐색
    3. **데이터 전처리**：결측값 채우기、범주형 특징 인코딩
    4. **모델 학습**：「회귀 분석+의사결정나무」하이브리드모형 학습
    5. **모델 예측**：단일 데이터 입력 또는 일괄 업로드 예측을 지원
    6. **성능 평가**：하이브리드모형과 단일 모형의 성능을 비교
    
    ### 적용 가능场景
    - logit 작업（분류）：사용자가 서비스를 수락할지 여부、위반 여부等 이진 예측（모델：로지스틱 회귀+분류 의사결정나무）
    - 의사결정나무 작업（회귀）：판매량、금액、평점等 연속값 예측（모델：선형 회귀+회귀 의사결정나무）
    
    ### 왼쪽「데이터 업로드」를 클릭하거나 아래 버튼으로 바로 이동하세요！
    """)
    
    # 快速跳转按钮
    if st.button("🚀 바로 데이터 선택으로 이동", type="primary"):
        st.session_state.step = 1
        st.rerun()

# ----------------------
#  단계 1：데이터 업로드（核心修复：移除timeout参数，仅保留手动跳转）
# ----------------------
elif st.session_state.step == 1:
    st.subheader("📤 데이터 선택（자신의 데이터 업로드 또는 예제 데이터 사용）")
    
    #  탭 분할：자신의 데이터 / 예제 데이터（保留原有功能）
    tab1, tab2 = st.tabs(["📁 자신의 데이터 업로드", "📊 예제 데이터 사용"])
    
    # Tab 1：기존 자료 업로드功能 + 手动跳转
    with tab1:
        st.markdown("지원 형식：CSV、Parquet、Excel（.xlsx/.xls）")
        st.markdown("⚠️  파일에 타겟 열（예측할 변수）과 특징 열（예측에 사용할 변수）이 모두 포함되어야 합니다")
        
        uploaded_file = st.file_uploader("데이터 파일 선택", type=["csv", "parquet", "xlsx", "xls"], key="single_file")
        
        if uploaded_file is not None:
            try:
                #  다양한 형식 파일 읽기
                if uploaded_file.name.endswith(".csv"):
                    df_merged = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(".parquet"):
                    df_merged = pd.read_parquet(uploaded_file)
                elif uploaded_file.name.endswith((".xlsx", ".xls")):
                    df_merged = pd.read_excel(uploaded_file)
                else:
                    st.error("지원하지 않는 파일 형식입니다！CSV/Parquet/Excel 파일을 업로드하세요")
                    st.stop()
                
                #  데이터 저장（标记为非示例数据，初始化离散化列）
                st.session_state.data["merged"] = df_merged
                st.session_state.data["is_sample"] = False
                st.session_state.data["discretized_cols"] = None
                
                #  데이터 정보 표시
                st.success(f"✅ 데이터 업로드 성공！")
                st.metric("데이터 양", f"{len(df_merged):,} 행 × {len(df_merged.columns)} 열")
                st.markdown("### 데이터 미리보기")
                st.dataframe(df_merged.head(5), use_container_width=True)
                
                #  手动跳转按钮（移除自动跳转，加强提示）
                st.divider()
                col1, col2, col3 = st.columns(3)
                with col2:
                    st.markdown("📊 데이터 시각화 단계로 이동하세요")
                    if st.button("🚀 데이터 시각화로 이동", type="primary"):
                        st.session_state.step = 2
                        st.rerun()
            
            except Exception as e:
                st.error(f"데이터 읽기 실패：{str(e)}")
    
    # Tab 2：예제 데이터 사용 + 手动跳转（修复核心错误）
    with tab2:
        st.markdown("### 📋 예제 데이터 선택")
        st.markdown("아래 예제 데이터를 선택하여 프레임워크 기능을 바로体验하세요！")
        
        #  작업 유형에 맞는 예제 데이터 제공
        sample_data_option = st.radio(
            "예제 데이터 종류",
            options=[
                "분류 예제：와인 품질 분류（logit 작업용）",
                "회귀 예제：캘리포니아 주택 가격（의사결정나무 작업용）"
            ],
            index=0
        )
        
        #  예제 데이터 설명
        if sample_data_option == "분류 예제：와인 품질 분류（logit 작업용）":
            st.markdown("""
            **데이터 설명**：
            - 데이터 소스：sklearn 내장 와인 데이터셋（Wine Dataset）
            - 데이터 크기：178 행 × 14 열（13개 특징 + 1개 타겟）
            - 특징 변수：알코올 함량、산도、당분 등 와인 속성
            - 타겟 변수：와인 품질（1=좋은 와인，0=일반 와인）- 이진 분류
            """)
            #  자동으로 작업 유형을 logit으로 설정
            if st.session_state.task != "logit":
                st.session_state.task = "logit"
                st.info("✅ 작업 유형이 자동으로「logit（분류）」로 설정되었습니다")
        
        else:
            st.markdown("""
            **데이터 설명**：
            - 데이터 소스：sklearn 내장 캘리포니아 주택 가격 데이터셋（California Housing）
            - 데이터 크기：20,640 행 × 9 열（8개 특징 + 1개 타겟）
            - 특징 변수：거주자 평균 소득、가구 수、방 개수等 지역 속성
            - 타겟 변수：주택 가격 중앙값（단위：10만 달러）- 연속값 회귀
            """)
            #  자동으로 작업 유형을 의사결정나무（회귀）로 설정
            if st.session_state.task != "의사결정나무":
                st.session_state.task = "의사결정나무"
                st.info("✅ 작업 유형이 자동으로「의사결정나무（회귀）」로 설정되었습니다")
        
        #  예제 데이터 로드 + 手动跳转（移除timeout参数）
        col1, col2, col3 = st.columns(3)
        with col2:
            load_btn = st.button("📥 예제 데이터 로드", type="primary")
        
        if load_btn:
            try:
                if sample_data_option == "분류 예제：와인 품질 분류（logit 작업용）":
                    #  와인 데이터 로드 + 전처리（이진 분류로 변환）
                    wine = load_wine()
                    df_merged = pd.DataFrame(data=wine.data, columns=wine.feature_names)
                    df_merged["wine_quality"] = wine.target
                    df_merged = df_merged[df_merged["wine_quality"] < 2]
                    df_merged["wine_quality"] = df_merged["wine_quality"].map({0: 0, 1: 1})
                    #  컬럼명 한글화
                    df_merged.columns = [
                        "알코올 함량", "말산", "회분", "회분 알칼리도", "마그네슘", "총 폴리페놀",
                        "플라보노이드 폴리페놀", "비플라보노이드 폴리페놀", "프로안토시아닌", "색상 강도",
                        "색상", "희석율", "프롤린", "와인 품질（타겟）"
                    ]
                
                else:
                    #  캘리포니아 주택 가격 데이터 로드
                    california = fetch_california_housing()
                    df_merged = pd.DataFrame(data=california.data, columns=california.feature_names)
                    df_merged["house_price"] = california.target
                    #  컬럼명 한글화
                    df_merged.columns = [
                        "거주자 평균 소득", "주택 연령 중앙값", "총 방 개수", "총 침실 개수",
                        "인구 수", "가구 수", "위도", "경도", "주택 가격 중앙값（타겟）"
                    ]
                    #  데이터 샘플링（1000행）
                    df_merged = df_merged.sample(n=1000, random_state=42).reset_index(drop=True)
                
                #  데이터 저장
                st.session_state.data["merged"] = df_merged
                st.session_state.data["is_sample"] = True
                st.session_state.data["discretized_cols"] = None
                
                st.success("🎉 예제 데이터 로드 성공！")
                st.metric("데이터 양", f"{len(df_merged):,} 행 × {len(df_merged.columns)} 열")
                st.markdown("### 데이터 미리보기")
                st.dataframe(df_merged.head(5), use_container_width=True)
                
                #  手动跳转按钮（核心修复：移除timeout=3参数）
                st.divider()
                col1, col2, col3 = st.columns(3)
                with col2:
                    st.markdown("📊 데이터 시각화 단계로 이동하세요")
                    if st.button("🚀 데이터 시각화로 이동", type="primary"):
                        st.session_state.step = 2
                        st.rerun()
                
            except Exception as e:
                st.error(f"예제 데이터 로드 실패：{str(e)}")

# ----------------------
#  단계 2：데이터 시각화（保留原有优化：数值型变量离散化）
# ----------------------
elif st.session_state.step == 2:
    st.subheader("📊 데이터 시각화")
    
    #  检查数据是否已选择
    if st.session_state.data["merged"] is None:
        st.warning("⚠️ 먼저「데이터 업로드」단계에서 데이터를 선택（업로드或예제加载）하세요")
        #  快速跳转按钮
        if st.button("🚀 데이터 선택으로 이동", type="primary"):
            st.session_state.step = 1
            st.rerun()
    else:
        df = st.session_state.data["merged"].copy()
        discretized_cols = st.session_state.data["discretized_cols"]
        
        #  1. 变量类型自动识别
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        
        #  2. 数值型变量离散化功能（展开/收起面板）
        with st.expander("🔧 수치형 변수离散化（범주형 변수가 없을 때 사용）", expanded=False):
            st.markdown("수치형 변수를 지정된 구간으로 나누어 범주형 변수로 변환합니다（막대그래프/박스플롯 등에 사용 가능）")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                discretize_var = st.selectbox("离散化할 수치형 변수 선택", options=num_cols, index=0 if num_cols else None, disabled=not num_cols)
            with col2:
                discretize_method = st.selectbox("离散化 방식", options=["분位数 분할", "고정 구간 분할"], index=0)
            with col3:
                n_bins = st.number_input("구간 개수", min_value=2, max_value=10, value=4, step=1)
            
            #  离散化 실행按钮
            if st.button("离散化 실행", type="secondary"):
                if not discretize_var:
                    st.error("离散化할 변수를 선택해야 합니다！")
                else:
                    try:
                        #  离散化逻辑
                        if discretize_method == "분位数 분할":
                            df[f"{discretize_var}_범주"], bins = pd.qcut(
                                df[discretize_var].dropna(), 
                                q=n_bins, 
                                labels=[f"{discretize_var}_{i+1}등급" for i in range(n_bins)],
                                duplicates="drop"
                            )
                        else:
                            df[f"{discretize_var}_범주"], bins = pd.cut(
                                df[discretize_var].dropna(), 
                                bins=n_bins, 
                                labels=[f"{discretize_var}_{i+1}등급" for i in range(n_bins)],
                                include_lowest=True
                            )
                        
                        #  离散化된 변수명 저장
                        discretized_col = f"{discretize_var}_범주"
                        st.session_state.data["discretized_cols"] = discretized_col
                        st.session_state.data["merged"] = df
                        
                        st.success(f"✅ {discretize_var}를 {n_bins}개의 범주로离散化 완료！")
                        st.write(f"离散化된 변수명：{discretized_col}")
                        st.write(f"구간 경계값：{np.round(bins, 2)}")
                        
                        #  离散化 결과 미리보기
                        st.markdown("### 离散化 결과 미리보기")
                        preview_df = df[[discretize_var, discretized_col]].head(10)
                        st.dataframe(preview_df, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"离散化 실패：{str(e)}")
            
            #  离散化变量 삭제按钮
            if discretized_cols:
                if st.button("离散化 변수 삭제", type="destructive"):
                    df = df.drop(columns=[discretized_cols])
                    st.session_state.data["merged"] = df
                    st.session_state.data["discretized_cols"] = None
                    st.success("离散化 변수를 삭제했습니다！")
                    st.rerun()
        
        #  3. 更新变量列表（原有범주형变量 + 离散化变量）
        updated_cat_cols = cat_cols.copy()
        if discretized_cols:
            updated_cat_cols.append(discretized_cols)
        
        #  4. 变量选择面板
        st.markdown("### 📋 변수 선택")
        col1, col2, col3 = st.columns(3)
        with col1:
            x_var_options = ["선택 안 함"] + updated_cat_cols
            x_var = st.selectbox("X축：범주형 변수（막대/박스/바이올린/산점도/선 그래프에 필수）", 
                               options=x_var_options, index=0)
            x_var = None if x_var == "선택 안 함" else x_var
        with col2:
            y_var = st.selectbox("Y축：수치형 변수（필수）", options=num_cols, 
                               index=0 if num_cols else None, disabled=not num_cols)
        with col3:
            graph_types = [
                "막대 그래프（평균값）", 
                "박스 플롯（분포）", 
                "바이올린 플롯（분포+밀도）",
                "산점도（개별 데이터）",
                "선 그래프（추세）",
                "히스토그램（분포）"
            ]
            graph_type = st.selectbox("📊 그래프 유형", options=graph_types, index=0)
        
        #  5. 图表绘制
        st.divider()
        if y_var:
            if graph_type == "히스토그램（분포）":
                st.markdown(f"### {y_var} 분포（히스토그램）")
                plot_df = df[[y_var] + ([x_var] if x_var else [])].dropna()
                
                try:
                    bins = st.slider("히스토그램 구간 개수", min_value=10, max_value=100, value=30, step=5)
                    
                    if x_var:
                        fig = px.histogram(
                            plot_df, 
                            x=y_var,
                            color=x_var,
                            barmode="overlay",
                            opacity=0.7,
                            nbins=bins,
                            title=f"{x_var}별 {y_var} 분포",
                            labels={y_var: y_var, x_var: x_var},
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                    else:
                        fig = px.histogram(
                            plot_df,
                            x=y_var,
                            nbins=bins,
                            title=f"{y_var} 전체 분포",
                            labels={y_var: y_var, "count": "빈도수"},
                            color_discrete_sequence=["#636EFA"],
                            marginal="box"
                        )
                    
                    fig.update_layout(width=1200, height=600, xaxis_title_font=dict(size=14), yaxis_title_font=dict(size=14), title_font=dict(size=16, weight="bold"))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    #  统计信息
                    st.markdown("### 📋 분포 통계 정보")
                    stats = plot_df[y_var].describe().round(3)
                    stats_df = pd.DataFrame({
                        "통계량": ["개수", "평균값", "표준편차", "최소값", "제1사분위수", "중앙값", "제3사분위수", "최대값"],
                        "값": [stats["count"], stats["mean"], stats["std"], stats["min"], stats["25%"], stats["50%"], stats["75%"], stats["max"]]
                    })
                    st.dataframe(stats_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"히스토그램 생성 실패：{str(e)}")
            
            else:
                if not x_var:
                    st.warning("""
                    ⚠️ 막대 그래프/박스 플롯/바이올린 플롯/산점도/선 그래프는 X축（범주형 변수）를 선택해야 합니다！
                    - 1. 데이터에 범주형 변수가 있는 경우：위 X축 선택박스에서 직접 선택
                    - 2. 데이터에 범주형 변수가 없는 경우：위「수치형 변수离散化」패널에서 수치형 변수를 범주형으로 변환 후 선택
                    """)
                    st.stop()
                
                st.markdown(f"### {x_var} vs {y_var} ({graph_type.split('（')[0]})")
                plot_df = df[[x_var, y_var]].dropna()
                
                try:
                    if graph_type == "막대 그래프（평균값）":
                        bar_data = plot_df.groupby(x_var)[y_var].mean().reset_index()
                        fig = px.bar(
                            bar_data, x=x_var, y=y_var, 
                            title=f"{x_var}별 {y_var} 평균값",
                            labels={y_var: f"{y_var} 평균값", x_var: x_var},
                            color=x_var, color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                    
                    elif graph_type == "박스 플롯（분포）":
                        fig = px.box(
                            plot_df, x=x_var, y=y_var,
                            title=f"{x_var}별 {y_var} 분포",
                            labels={y_var: y_var, x_var: x_var},
                            color=x_var, color_discrete_sequence=px.colors.qualitative.Set2
                        )
                    
                    elif graph_type == "바이올린 플롯（분포+밀도）":
                        fig = px.violin(
                            plot_df, x=x_var, y=y_var,
                            title=f"{x_var}별 {y_var} 분포 및 밀도",
                            labels={y_var: y_var, x_var: x_var},
                            color=x_var, box=True,
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                    
                    elif graph_type == "산점도（개별 데이터）":
                        fig = px.scatter(
                            plot_df, x=x_var, y=y_var,
                            title=f"{x_var} vs {y_var} 개별 데이터 분포",
                            labels={y_var: y_var, x_var: x_var},
                            color=x_var, opacity=0.6,
                            color_discrete_sequence=px.colors.qualitative.Vivid
                        )
                    
                    elif graph_type == "선 그래프（추세）":
                        line_data = plot_df.groupby(x_var)[y_var].mean().reset_index()
                        fig = px.line(
                            line_data, x=x_var, y=y_var,
                            title=f"{x_var}별 {y_var} 추세",
                            labels={y_var: y_var, x_var: x_var},
                            color_discrete_sequence=["#1f77b4"],
                            markers=True
                        )
                    
                    fig.update_layout(width=1200, height=600, xaxis_title_font=dict(size=14), yaxis_title_font=dict(size=14), title_font=dict(size=16, weight="bold"))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    #  统计信息
                    st.markdown("### 📋 통계 정보")
                    stats_df = plot_df.groupby(x_var)[y_var].agg([
                        "count", "mean", "std", "min", "25%", "50%", "75%", "max"
                    ]).round(3)
                    stats_df.columns = ["데이터 개수", "평균값", "표준편차", "최소값", "제1사분위수", "중앙값", "제3사분위수", "최대값"]
                    st.dataframe(stats_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"그래프 생성 실패：{str(e)}")
        else:
            st.warning("Y축（수치형 변수）를 선택해야 합니다")
        
        #  下一步 안내
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col3:
            if st.button("🔧 데이터 전처리로 이동", type="primary"):
                st.session_state.step = 3
                st.rerun()

# ----------------------
#  단계 3：데이터 전처리（保留原有功能）
# ----------------------
elif st.session_state.step == 3:
    st.subheader("🧹 데이터 전처리")
    
    if st.session_state.data["merged"] is None:
        st.warning("⚠️ 먼저「데이터 업로드」단계에서 데이터를 선택하세요")
        if st.button("🚀 데이터 선택으로 이동", type="primary"):
            st.session_state.step = 1
            st.rerun()
    else:
        df_merged = st.session_state.data["merged"].copy()
        discretized_cols = st.session_state.data["discretized_cols"]
        
        #  1. 데이터 개요
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 데이터 기본 정보")
            st.write(f"총 데이터 양：{len(df_merged):,} 행 × {len(df_merged.columns)} 열")
            if discretized_cols:
                st.success(f"离散化 변수가 존재합니다：{discretized_cols}")
            st.write("데이터 유형 분포：")
            st.dataframe(df_merged.dtypes.value_counts().reset_index(), use_container_width=True)
        
        with col2:
            st.markdown("### 결측값 분포")
            missing_info = df_merged.isnull().sum()[df_merged.isnull().sum() > 0].reset_index()
            missing_info.columns = ["필드명", "결측값 개수"]
            if len(missing_info) > 0:
                st.dataframe(missing_info, use_container_width=True)
                fig_missing = px.imshow(df_merged.isnull(), color_continuous_scale="Reds", title="결측값 히트맵")
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("결측값이 없습니다！")
        
        #  2. 전처리 설정
        st.divider()
        st.markdown("### 전처리 매개변수 설정")
        
        #  타겟 열 선택
        st.markdown("#### 타겟 열 선택（예측할 변수）")
        target_options = df_merged.columns.tolist()
        default_target_idx = 0
        if st.session_state.data["is_sample"]:
            if st.session_state.task == "logit":
                default_target_idx = target_options.index("와인 품질（타겟）") if "와인 품질（타겟）" in target_options else 0
            else:
                default_target_idx = target_options.index("주택 가격 중앙값（타겟）") if "주택 가격 중앙값（타겟）" in target_options else 0
        
        target_col = st.selectbox(
            "타겟 열 선택", 
            options=target_options, 
            index=default_target_idx
        )
        st.session_state.preprocess["target_col"] = target_col
        
        #  특징 열 선택（默认排除离散化变量）
        exclude_cols = [target_col]
        if discretized_cols:
            exclude_cols.append(discretized_cols)
        
        exclude_cols = st.multiselect(
            "제외할 열 선택（예：ID、무관한 필드、离散化变量）", 
            options=[col for col in df_merged.columns if col != target_col],
            default=[discretized_cols] if discretized_cols else []
        )
        feature_cols = [col for col in df_merged.columns if col not in exclude_cols + [target_col]]
        
        if not feature_cols:
            st.warning("특징 열이 선택되지 않았습니다！제외할 열을 조정하세요")
        st.session_state.preprocess["feature_cols"] = feature_cols
        
        #  결측값 처리
        st.markdown("#### 결측값 처리")
        impute_strategy = st.selectbox("수치형 결측값 채우기 방식", options=["중앙값", "평균값", "최빈값"], index=0)
        impute_strategy_map = {"중앙값": "median", "평균값": "mean", "최빈값": "most_frequent"}
        
        #  범주형 특징 인코딩
        st.markdown("#### 범주형 특징 인코딩")
        cat_encoding = st.selectbox("범주형 특징 인코딩 방식", options=["레이블 인코딩（LabelEncoder）", "원-핫 인코딩（OneHotEncoder）"], index=0)
        
        #  3. 전처리 실행
        if st.button("전처리 시작", type="primary"):
            if not feature_cols:
                st.error("전처리 실패：특징 열이 없습니다！")
                st.stop()
            
            try:
                X = df_merged[feature_cols].copy()
                y = df_merged[target_col].copy()
                
                #  수치형과 범주형 특징 분리
                num_cols = X.select_dtypes(include=["int64", "float64"]).columns
                cat_cols = X.select_dtypes(include=["object", "category"]).columns
                
                #  수치형 전처리
                imputer = SimpleImputer(strategy=impute_strategy_map[impute_strategy])
                X[num_cols] = imputer.fit_transform(X[num_cols])
                
                scaler = StandardScaler()
                X[num_cols] = scaler.fit_transform(X[num_cols])
                
                #  범주형 전처리
                encoders = {}
                for col in cat_cols:
                    X[col] = X[col].fillna("알 수 없음").astype(str)
                    
                    if cat_encoding == "레이블 인코딩（LabelEncoder）":
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col])
                        encoders[col] = le
                    else:
                        ohe = OneHotEncoder(sparse_output=False, drop="first")
                        ohe_result = ohe.fit_transform(X[[col]])
                        ohe_cols = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]
                        X = pd.concat([X.drop(col, axis=1), pd.DataFrame(ohe_result, columns=ohe_cols)], axis=1)
                        encoders[col] = (ohe, ohe_cols)
                
                #  전처리组件 저장
                st.session_state.preprocess["imputer"] = imputer
                st.session_state.preprocess["scaler"] = scaler
                st.session_state.preprocess["encoders"] = encoders
                st.session_state.preprocess["feature_cols"] = list(X.columns)
                
                #  전처리된 데이터 저장
                st.session_state.data["X_processed"] = X
                st.session_state.data["y_processed"] = y
                
                st.success("데이터 전처리 완료！")
                st.markdown(f"전처리 후 특징 수：{len(X.columns)}")
                st.dataframe(X.head(3), use_container_width=True)
                
                #  下一步跳转按钮
                col1, col2, col3 = st.columns(3)
                with col3:
                    if st.button("🚀 모델 학습으로 이동", type="primary"):
                        st.session_state.step = 4
                        st.rerun()
            except Exception as e:
                st.error(f"전처리 실패：{str(e)}")

# ----------------------
#  단계 4：모델 학습（保留原有功能）
# ----------------------
elif st.session_state.step == 4:
    st.subheader("🚀 하이브리드모형 학습（회귀 분석 + 의사결정나무）")
    
    if "X_processed" not in st.session_state.data or "y_processed" not in st.session_state.data:
        st.warning("⚠️ 먼저「데이터 전처리」단계를 완료하세요")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔧 데이터 전처리로 이동", type="primary"):
                st.session_state.step = 3
                st.rerun()
    else:
        X = st.session_state.data["X_processed"]
        y = st.session_state.data["y_processed"]
        
        #  데이터 분할
        st.markdown("### 학습 설정")
        test_size = st.slider("테스트集 비율", min_value=0.1, max_value=0.3, value=0.2, step=0.05)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, 
            stratify=y if st.session_state.task == "logit" else None
        )
        
        #  모델 선택
        if st.session_state.task == "logit":
            reg_model = LogisticRegression(max_iter=1000)
            dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
        else:
            reg_model = LinearRegression()
            dt_model = DecisionTreeRegressor(random_state=42, max_depth=10)
        
        #  모델 학습
        if st.button("모델 학습 시작", type="primary"):
            with st.spinner("모델 학습 중..."):
                reg_model.fit(X_train, y_train)
                dt_model.fit(X_train, y_train)
                
                #  모델 저장
                st.session_state.models["regression"] = reg_model
                st.session_state.models["decision_tree"] = dt_model
                
                #  데이터 저장
                st.session_state.data["X_train"] = X_train
                st.session_state.data["X_test"] = X_test
                st.session_state.data["y_train"] = y_train
                st.session_state.data["y_test"] = y_test
                
                st.success("모델 학습 완료！")
                st.markdown("✅ 학습된 모델：")
                st.markdown("- 회귀 분석（로지스틱/선형，해석력 강함）")
                st.markdown("- 의사결정나무（분류/회귀，정확도 높음）")
                st.markdown("- 하이브리드모형（전两者 가중融合）")
                
                #  下一步跳转按钮
                col1, col2, col3 = st.columns(3)
                with col3:
                    if st.button("🎯 모델 예측으로 이동", type="primary"):
                        st.session_state.step = 5
                        st.rerun()

# ----------------------
#  단계 5：모델 예측（保留原有功能）
# ----------------------
elif st.session_state.step == 5:
    st.subheader("🎯 모델 예측")
    
    if st.session_state.models["regression"] is None or st.session_state.models["decision_tree"] is None:
        st.warning("⚠️ 먼저「모델 학습」단계를 완료하세요")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🚀 모델 학습으로 이동", type="primary"):
                st.session_state.step = 4
                st.rerun()
    else:
        #  예측 함수
        def predict(input_data):
            X = input_data.copy()
            preprocess = st.session_state.preprocess
            num_cols = X.select_dtypes(include=["int64", "float64"]).columns
            cat_cols = X.select_dtypes(include=["object", "category"]).columns
            
            #  수치형 전처리
            X[num_cols] = preprocess["imputer"].transform(X[num_cols])
            X[num_cols] = preprocess["scaler"].transform(X[num_cols])
            
            #  범주형 전처리
            for col in cat_cols:
                X[col] = X[col].fillna("알 수 없음").astype(str)
                encoder = preprocess["encoders"][col]
                
                if isinstance(encoder, LabelEncoder):
                    X[col] = X[col].replace([x for x in X[col].unique() if x not in encoder.classes_], "알 수 없음")
                    if "알 수 없음" not in encoder.classes_:
                        encoder.classes_ = np.append(encoder.classes_, "알 수 없음")
                    X[col] = encoder.transform(X[col])
                else:
                    ohe, ohe_cols = encoder
                    ohe_result = ohe.transform(X[[col]])
                    X = pd.concat([X.drop(col, axis=1), pd.DataFrame(ohe_result, columns=ohe_cols)], axis=1)
            
            #  특징 열 순서 일치
            X = X[preprocess["feature_cols"]]
            
            #  하이브리드 예측
            reg_weight = st.session_state.models["mixed_weights"]["regression"]
            dt_weight = st.session_state.models["mixed_weights"]["decision_tree"]
            reg_model = st.session_state.models["regression"]
            dt_model = st.session_state.models["decision_tree"]
            
            if st.session_state.task == "logit":
                reg_proba = reg_model.predict_proba(X)[:, 1]
                dt_proba = dt_model.predict_proba(X)[:, 1]
                mixed_proba = reg_weight * reg_proba + dt_weight * dt_proba
                pred = (mixed_proba >= 0.5).astype(int)
                return pred, mixed_proba
            else:
                reg_pred = reg_model.predict(X)
                dt_pred = dt_model.predict(X)
                mixed_pred = reg_weight * reg_pred + dt_weight * dt_pred
                return mixed_pred, None
        
        #  예측 방식 선택
        predict_mode = st.radio("예측 방식", options=["단일 데이터 입력", "일괄 업로드 CSV"])
        
        #  단일 입력 예측
        if predict_mode == "단일 데이터 입력":
            st.markdown("#### 단일 데이터 입력（특징값을 입력하세요）")
            feature_cols = st.session_state.preprocess["feature_cols"]
            input_data = {}
            
            with st.form("single_pred_form"):
                cols = st.columns(3)
                for i, col in enumerate(feature_cols[:9]):
                    with cols[i % 3]:
                        #  예제 데이터인 경우 기본값 제공
                        default_value = 0.0
                        if st.session_state.data["is_sample"]:
                            if "알코올 함량" in col or "거주자 평균 소득" in col:
                                default_value = st.session_state.data["X_processed"][col].mean()
                            elif "주택 연령 중앙값" in col:
                                default_value = st.session_state.data["X_processed"][col].mean()
                            else:
                                default_value = st.session_state.data["X_processed"][col].mean()
                        
                        input_data[col] = st.number_input(col, value=float(default_value))
                
                submit_btn = st.form_submit_button("예측 시작")
            
            if submit_btn:
                input_df = pd.DataFrame([input_data])
                pred, proba = predict(input_df)
                
                st.divider()
                st.markdown("### 예측 결과")
                if st.session_state.task == "logit":
                    st.metric("예측 결과", "좋은 와인（양성）" if pred[0] == 1 else "일반 와인（음성）")
                    st.metric("양성 확률", f"{proba[0]:.3f}" if proba is not None else "-")
                else:
                    st.metric("주택 가격 예측 결과", f"{pred[0]:.2f} × 10만 달러")
        
        #  일괄 업로드 예측
        else:
            st.markdown("#### 일괄 업로드 CSV 예측")
            uploaded_file = st.file_uploader("특징 열을 포함한 CSV 파일 업로드", type=["csv"])
            
            if uploaded_file is not None:
                batch_df = pd.read_csv(uploaded_file)
                st.metric("업로드 데이터 양", f"{len(batch_df):,} 행")
                st.dataframe(batch_df.head(3), use_container_width=True)
                
                required_features = st.session_state.preprocess["feature_cols"]
                missing_features = [col for col in required_features if col not in batch_df.columns]
                if missing_features:
                    st.warning(f"업로드된 파일에 필요한 특징 열이 없습니다：{', '.join(missing_features)}")
                else:
                    if st.button("일괄 예측 시작"):
                        with st.spinner("예측 중..."):
                            pred, proba = predict(batch_df)
                            batch_df["하이브리드모형 예측 결과"] = pred
                            if proba is not None:
                                batch_df["양성 확률"] = proba.round(3)
                            
                            st.divider()
                            st.markdown("### 일괄 예측 결과")
                            st.dataframe(
                                batch_df[["하이브리드모형 예측 결과"] + (["양성 확률"] if proba is not None else []) + feature_cols[:3]],
                                use_container_width=True
                            )
                            
                            #  결과 다운로드
                            csv = batch_df.to_csv(index=False, encoding="utf-8-sig")
                            st.download_button(
                                label="예측 결과 다운로드",
                                data=csv,
                                file_name="하이브리드모형_일괄예측결과.csv",
                                mime="text/csv"
                            )
        
        #  下一步跳转按钮
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col3:
            if st.button("📈 성능 평가로 이동", type="primary"):
                st.session_state.step = 6
                st.rerun()

# ----------------------
#  단계 6：성능 평가（保留原有功能）
# ----------------------
elif st.session_state.step == 6:
    st.subheader("📈 모델 성능 평가")
    
    if st.session_state.models["regression"] is None or st.session_state.models["decision_tree"] is None:
        st.warning("⚠️ 먼저「모델 학습」단계를 완료하세요")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🚀 모델 학습으로 이동", type="primary"):
                st.session_state.step = 4
                st.rerun()
    else:
        X_test = st.session_state.data["X_test"]
        y_test = st.session_state.data["y_test"]
        reg_model = st.session_state.models["regression"]
        dt_model = st.session_state.models["decision_tree"]
        reg_weight = st.session_state.models["mixed_weights"]["regression"]
        dt_weight = st.session_state.models["mixed_weights"]["decision_tree"]
        
        #  각 모델 예측
        if st.session_state.task == "logit":
            reg_pred = reg_model.predict(X_test)
            dt_pred = dt_model.predict(X_test)
            reg_proba = reg_model.predict_proba(X_test)[:, 1]
            dt_proba = dt_model.predict_proba(X_test)[:, 1]
            mixed_proba = reg_weight * reg_proba + dt_weight * dt_proba
            mixed_pred = (mixed_proba >= 0.5).astype(int)
            
            #  분류 지표 계산
            def calc_class_metrics(y_true, y_pred, y_proba):
                acc = accuracy_score(y_true, y_pred)
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                auc_score = auc(fpr, tpr)
                return {"정확도": acc, "AUC": auc_score}
            
            reg_metrics = calc_class_metrics(y_test, reg_pred, reg_proba)
            dt_metrics = calc_class_metrics(y_test, dt_pred, dt_proba)
            mixed_metrics = calc_class_metrics(y_test, mixed_pred, mixed_proba)
            
            metrics_df = pd.DataFrame({
                "모델": ["회귀 분석（로지스틱）", "의사결정나무（분류）", "하이브리드모형"],
                "정확도": [reg_metrics["정확도"], dt_metrics["정확도"], mixed_metrics["정확도"]],
                "AUC": [reg_metrics["AUC"], dt_metrics["AUC"], mixed_metrics["AUC"]]
            }).round(3)
        
        else:
            reg_pred = reg_model.predict(X_test)
            dt_pred = dt_model.predict(X_test)
            mixed_pred = reg_weight * reg_pred + dt_weight * dt_pred
            
            #  회귀 지표 계산
            def calc_reg_metrics(y_true, y_pred):
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                return {"MAE": mae, "RMSE": rmse, "R²": r2}
            
            reg_metrics = calc_reg_metrics(y_test, reg_pred)
            dt_metrics = calc_reg_metrics(y_test, dt_pred)
            mixed_metrics = calc_reg_metrics(y_test, mixed_pred)
            
            metrics_df = pd.DataFrame({
                "모델": ["회귀 분석（선형）", "의사결정나무（회귀）", "하이브리드모형"],
                "MAE": [reg_metrics["MAE"], dt_metrics["MAE"], mixed_metrics["MAE"]],
                "RMSE": [reg_metrics["RMSE"], dt_metrics["RMSE"], mixed_metrics["RMSE"]],
                "R²": [reg_metrics["R²"], dt_metrics["R²"], mixed_metrics["R²"]]
            }).round(3)
        
        #  지표 비교
        st.markdown("### 모델 성능 비교")
        st.dataframe(metrics_df, use_container_width=True)
        
        #  시각화 비교
        col1, col2 = st.columns(2)
        
        if st.session_state.task == "logit":
            with col1:
                st.markdown("### ROC-AUC 곡선")
                fpr_reg, tpr_reg, _ = roc_curve(y_test, reg_proba)
                fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_proba)
                fpr_mixed, tpr_mixed, _ = roc_curve(y_test, mixed_proba)
                
                fig_auc = go.Figure()
                fig_auc.add_trace(go.Scatter(x=fpr_reg, y=tpr_reg, name=f"회귀 분석 (AUC={reg_metrics['AUC']:.3f})"))
                fig_auc.add_trace(go.Scatter(x=fpr_dt, y=tpr_dt, name=f"의사결정나무 (AUC={dt_metrics['AUC']:.3f})"))
                fig_auc.add_trace(go.Scatter(x=fpr_mixed, y=tpr_mixed, name=f"하이브리드모형 (AUC={mixed_metrics['AUC']:.3f})", line_dash="dash", line_width=3))
                fig_auc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="랜덤 추측", line_color="gray", line_dash="dot"))
                st.plotly_chart(fig_auc, use_container_width=True)
            
            with col2:
                st.markdown("### 혼동 행렬（하이브리드모형）")
                cm = confusion_matrix(y_test, mixed_pred)
                cm_df = pd.DataFrame(cm, index=["실제 일반 와인", "실제 좋은 와인"], columns=["예측 일반 와인", "예측 좋은 와인"])
                fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues")
                st.plotly_chart(fig_cm, use_container_width=True)
        
        else:
            with col1:
                st.markdown("### 예측값 vs 실제값（하이브리드모형）")
                fig_pred = px.scatter(x=y_test, y=mixed_pred, title="실제 주택 가격 vs 예측 가격", labels={"x": "실제 가격（10만 달러）", "y": "예측 가격（10만 달러）"})
                fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], line_color="red", name="이상적인 피팅 라인"))
                st.plotly_chart(fig_pred, use_container_width=True)
            
            with col2:
                st.markdown("### 잔차 그래프（하이브리드모형）")
                residuals = y_test - mixed_pred
                fig_res = px.scatter(x=mixed_pred, y=residuals, title="예측 가격 vs 잔차", labels={"x": "예측 가격（10만 달러）", "y": "잔차"})
                fig_res.add_trace(go.Scatter(x=[mixed_pred.min(), mixed_pred.max()], y=[0, 0], line_color="red", name="잔차=0 라인"))
                st.plotly_chart(fig_res, use_container_width=True)
        
        #  특징 중요도
        st.divider()
        st.markdown("### 모델 해석：핵심 특징 중요도")
        feature_importance = pd.DataFrame({
            "특징명": st.session_state.preprocess["feature_cols"],
            "중요도": dt_model.feature_importances_
        }).sort_values("중요도", ascending=False).head(10)
        
        fig_importance = px.bar(feature_importance, x="중요도", y="특징명", orientation="h", color="중요도", color_continuous_scale="viridis")
        st.plotly_chart(fig_importance, use_container_width=True)
        
        #  다시 시작按钮
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("🔄 전과정 다시 시작", type="primary"):
                #  상태 초기화
                st.session_state.data = {"merged": None, "is_sample": False, "discretized_cols": None}
                st.session_state.preprocess = {"imputer": None, "scaler": None, "encoders": None, "feature_cols": None, "target_col": None}
                st.session_state.models = {"regression": None, "decision_tree": None, "mixed_weights": {"regression": 0.3, "decision_tree": 0.7}}
                st.session_state.step = 1  # 回到数据上传步骤
                st.rerun()
