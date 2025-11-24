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
    accuracy_score, auc, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    mean_absolute_error, mean_squared_error, r2_score, classification_report
)
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

# 전역 상태 관리
if "step" not in st.session_state:
    st.session_state.step = 0  # 0:초기화면 1:데이터업로드 2:데이터시각화 3:데이터전처리 4:모델학습 5:예측 6:평가
if "data" not in st.session_state:
    st.session_state.data = {"merged": None}
if "preprocess" not in st.session_state:
    st.session_state.preprocess = {"imputer": None, "scaler": None, "encoders": None, "feature_cols": None, "target_col": None}
if "models" not in st.session_state:
    st.session_state.models = {"regression": None, "decision_tree": None, "mixed_weights": {"regression": 0.3, "decision_tree": 0.7}}
if "task" not in st.session_state:
    st.session_state.task = "logit"  # logit(분류) / 의사결정나무(회귀)

# ----------------------
# 2. 사이드바：단계导航 + 핵심 설정
# ----------------------
st.sidebar.title("📌 하이브리드모형 작업 흐름")
st.sidebar.divider()

# 단계导航 버튼
steps = ["초기 설정", "데이터 업로드", "데이터 시각화", "데이터 전처리", "모델 학습", "모델 예측", "성능 평가"]
for i, step_name in enumerate(steps):
    if st.sidebar.button(step_name, key=f"btn_{i}"):
        st.session_state.step = i

# 핵심 설정
st.sidebar.divider()
st.sidebar.subheader("핵심 설정")
st.session_state.task = st.sidebar.radio(
    "작업 유형", 
    options=["logit", "의사결정나무"], 
    index=0,
    format_func=lambda x: "분류 작업（예：수락여부）" if x == "logit" else "회귀 작업（예：금액예측）"
)

if st.session_state.step >= 4:  # 모델 학습 후 가중치 조정
    st.sidebar.subheader("하이브리드모형 가중치")
    reg_weight = st.sidebar.slider(
        "회귀 분석 가중치（해석력 강함）",
        min_value=0.0, max_value=1.0, value=st.session_state.models["mixed_weights"]["regression"], step=0.1
    )
    st.session_state.models["mixed_weights"]["regression"] = reg_weight
    st.session_state.models["mixed_weights"]["decision_tree"] = 1 - reg_weight
    st.sidebar.text(f"의사결정나무 가중치（정확도 높음）：{1 - reg_weight:.1f}")

# ----------------------
# 3. 메인 페이지：단계별 내용 표시
# ----------------------
st.title("📊 하이브리드모형 동적 배포 프레임워크")
st.markdown("**단일 원본 데이터 파일 업로드 후，시각화→전처리→학습→예측 전과정을 한 번에 완성**")
st.markdown("### 🧩 핵심 모델：회귀 분석（Regression）+ 의사결정나무（Decision Tree）")
st.divider()

# ----------------------
#  단계 0：초기 설정
# ----------------------
if st.session_state.step == 0:
    st.subheader("🎉 하이브리드모형 동적 프레임워크에 오신 것을 환영합니다")
    st.markdown("""
    본 프레임워크는 **데이터 수령 후 직접 업로드하여 사용**할 수 있으며，사전 전처리나 모델 학습이 필요 없습니다. 핵심 흐름은 다음과 같습니다：
    
    1. **데이터 업로드**：단일 원본 파일（CSV/Parquet/Excel）을 업로드
    2. **데이터 시각화**：범주형 변수와 수치형 변수를 선택하여 다양한 그래프로 데이터 탐색
    3. **데이터 전처리**：결측값 채우기、범주형 특징 인코딩
    4. **모델 학습**：「회귀 분석+의사결정나무」하이브리드모형 학습
    5. **모델 예측**：단일 데이터 입력 또는 일괄 업로드 예측을 지원
    6. **성능 평가**：하이브리드모형과 단일 모형의 성능을 비교
    
    ### 적용 가능 환경
    - logit 작업（분류）：사용자가 서비스를 수락할지 여부、위반 여부等 이진 예측（모델：로지스틱 회귀+분류 의사결정나무）
    - 의사결정나무 작업（회귀）：판매량、금액、평점等 연속값 예측（모델：선형 회귀+회귀 의사결정나무）
    
    ### 왼쪽「데이터 업로드」를 클릭하여 사용을 시작하세요！
    """)

# ----------------------
#  단계 1：데이터 업로드
# ----------------------
elif st.session_state.step == 1:
    st.subheader("📤 데이터 업로드（단일 파일）")
    st.markdown("지원 형식：CSV、Parquet、Excel（.xlsx/.xls）")
    st.markdown("⚠️  파일에 타겟 열（예측할 변수）과 특징 열（예측에 사용할 변수）이 모두 포함되어야 합니다")
    
    uploaded_file = st.file_uploader("데이터 파일 선택", type=["csv", "parquet", "xlsx", "xls"], key="single_file")
    
    if uploaded_file is not None:
        try:
            # 파일 읽기
            if uploaded_file.name.endswith(".csv"):
                df_merged = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".parquet"):
                df_merged = pd.read_parquet(uploaded_file)
            elif uploaded_file.name.endswith((".xlsx", ".xls")):
                df_merged = pd.read_excel(uploaded_file)
            else:
                st.error("지원하지 않는 파일 형식입니다！CSV/Parquet/Excel 파일을 업로드하세요")
                st.stop()
            
            # 데이터 저장 및 검증
            if df_merged.empty:
                st.error("업로드된 파일이 비어 있습니다！")
                st.stop()
            
            st.session_state.data["merged"] = df_merged
            
            # 데이터 정보 표시
            st.success(f"데이터 업로드 성공！")
            st.metric("데이터 양", f"{len(df_merged):,} 행 × {len(df_merged.columns)} 열")
            st.markdown("### 데이터 미리보기")
            st.dataframe(df_merged.head(5), use_container_width=True)
            
            # 데이터 기본 정보
            st.markdown("### 데이터 기본 정보")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**열 이름**")
                st.write(", ".join(df_merged.columns.tolist()[:10]) + ("..." if len(df_merged.columns) > 10 else ""))
            with col2:
                st.write("**결측값 총 개수**")
                st.write(f"{df_merged.isnull().sum().sum()} 개")
            with col3:
                st.write("**데이터 유형**")
                st.write(df_merged.dtypes.value_counts().to_string())
            
            st.divider()
            st.info("📊 데이터 탐색을 위해 왼쪽 사이드바에서「데이터 시각화」단계로 이동하세요")
        
        except Exception as e:
            st.error(f"데이터 읽기 실패：{str(e)}")

# ----------------------
#  단계 2：데이터 시각화
# ----------------------
elif st.session_state.step == 2:
    st.subheader("📊 데이터 시각화")
    
    if st.session_state.data["merged"] is None:
        st.warning("먼저「데이터 업로드」단계를 완료하세요")
    else:
        df = st.session_state.data["merged"]
        
        # 변수 유형识别
        st.markdown("### 변수 선택")
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        
        # 变量选择
        col1, col2, col3 = st.columns(3)
        with col1:
            x_var = st.selectbox("📋 X축：범주형 변수（선택 사항）", options=["선택 안 함"] + cat_cols, index=0)
            x_var = None if x_var == "선택 안 함" else x_var
        with col2:
            y_var = st.selectbox("📈 Y축：수치형 변수（필수）", options=num_cols, index=0 if num_cols else None, disabled=not num_cols)
        with col3:
            graph_types = [
                "막대 그래프（평균값）", "박스 플롯（분포）", "바이올린 플롯（분포+밀도）",
                "산점도（개별 데이터）", "선 그래프（추세）", "히스토그램（분포）"
            ]
            graph_type = st.selectbox("📊 그래프 유형", options=graph_types, index=0)
        
        # 绘图逻辑
        st.divider()
        if y_var:
            if graph_type == "히스토그램（분포）":
                st.markdown(f"### {y_var} 분포（히스토그램）")
                plot_df = df[[y_var] + ([x_var] if x_var else [])].dropna()
                
                try:
                    bins = st.slider("히스토그램 구간 개수", min_value=10, max_value=100, value=30, step=5)
                    
                    if x_var:  # 按类别分组
                        fig = px.histogram(
                            plot_df, x=y_var, color=x_var, barmode="overlay", opacity=0.7,
                            nbins=bins, title=f"{x_var}별 {y_var} 분포",
                            labels={y_var: y_var, x_var: x_var},
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                    else:  # 单变量
                        fig = px.histogram(
                            plot_df, x=y_var, nbins=bins, title=f"{y_var} 전체 분포",
                            labels={y_var: y_var, "count": "빈도수"},
                            color_discrete_sequence=["#636EFA"],
                            marginal="box"
                        )
                    
                    fig.update_layout(width=1200, height=600,
                                     xaxis_title_font=dict(size=14),
                                     yaxis_title_font=dict(size=14),
                                     title_font=dict(size=16, weight="bold"))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 统计信息
                    st.markdown("### 📋 분포 통계 정보")
                    stats = plot_df[y_var].describe().round(3)
                    stats_df = pd.DataFrame({
                        "통계량": ["개수", "평균값", "표준편차", "최소값", "제1사분위수", "중앙값", "제3사분위수", "최대값"],
                        "값": [stats["count"], stats["mean"], stats["std"],
                               stats["min"], stats["25%"], stats["50%"],
                               stats["75%"], stats["max"]]
                    })
                    st.dataframe(stats_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"히스토그램 생성 실패：{str(e)}")
            
            else:
                if not x_var:
                    st.warning("막대 그래프/박스 플롯/바이올린 플롯/산점도/선 그래프는 X축（범주형 변수）를 선택해야 합니다")
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
                    
                    fig.update_layout(width=1200, height=600,
                                     xaxis_title_font=dict(size=14),
                                     yaxis_title_font=dict(size=14),
                                     title_font=dict(size=16, weight="bold"))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 统计信息
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
        
        st.divider()
        st.info("🔧 데이터 전처리를 위해 왼쪽 사이드바에서「데이터 전처리」단계로 이동하세요")

# ----------------------
#  단계 3：데이터 전처리
# ----------------------
elif st.session_state.step == 3:
    st.subheader("🧹 데이터 전처리")
    
    if st.session_state.data["merged"] is None:
        st.warning("먼저「데이터 업로드」단계를 완료하세요")
    else:
        df_merged = st.session_state.data["merged"]
        
        # 数据概览
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 데이터 기본 정보")
            st.write(f"총 데이터 양：{len(df_merged):,} 행 × {len(df_merged.columns)} 열")
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
        
        # 预处理设置
        st.divider()
        st.markdown("### 전처리 매개변수 설정")
        
        # 目标列选择
        if len(df_merged.columns) > 0:
            target_col = st.selectbox(
                "타겟 열 선택（예측할 변수）", 
                options=df_merged.columns, 
                index=0
            )
            st.session_state.preprocess["target_col"] = target_col
        else:
            st.error("데이터에 열이 존재하지 않습니다！올바른 데이터 파일을 업로드하세요")
            st.stop()
        
        # 特征列选择
        exclude_cols = st.multiselect(
            "제외할 열 선택（예：ID、무관한 필드）", 
            options=[col for col in df_merged.columns if col != target_col]
        )
        feature_cols = [col for col in df_merged.columns if col not in exclude_cols + [target_col]]
        
        if not feature_cols:
            st.warning("특징 열이 선택되지 않았습니다！제외할 열을 조정하세요")
        st.session_state.preprocess["feature_cols"] = feature_cols
        
        # 缺失值处理
        st.markdown("#### 결측값 처리")
        impute_strategy = st.selectbox("수치형 결측값 채우기 방식", options=["중앙값", "평균값", "최빈값"], index=0)
        impute_strategy_map = {"중앙값": "median", "평균값": "mean", "최빈값": "most_frequent"}
        
        # 类别特征编码
        st.markdown("#### 범주형 특징 인코딩")
        cat_encoding = st.selectbox("범주형 특징 인코딩 방식", options=["레이블 인코딩（LabelEncoder）", "원-핫 인코딩（OneHotEncoder）"], index=0)
        
        # 执行预处理
        if st.button("전처리 시작"):
            if not feature_cols:
                st.error("전처리 실패：특징 열이 없습니다！")
                st.stop()
            
            try:
                X = df_merged[feature_cols].copy()
                y = df_merged[target_col].copy()
                
                # 分离数值和类别特征
                num_cols = X.select_dtypes(include=["int64", "float64"]).columns
                cat_cols = X.select_dtypes(include=["object", "category"]).columns
                
                # 数值型预处理
                imputer = SimpleImputer(strategy=impute_strategy_map[impute_strategy])
                X[num_cols] = imputer.fit_transform(X[num_cols])
                
                scaler = StandardScaler()
                X[num_cols] = scaler.fit_transform(X[num_cols])
                
                # 类别型预处理
                encoders = {}
                for col in cat_cols:
                    X[col] = X[col].fillna("알 수 없음").astype(str)
                    
                    if cat_encoding == "레이블 인코딩（LabelEncoder）":
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col])
                        encoders[col] = le
                    else:  # 独热编码
                        ohe = OneHotEncoder(sparse_output=False, drop="first")
                        ohe_result = ohe.fit_transform(X[[col]])
                        ohe_cols = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]
                        X = pd.concat([X.drop(col, axis=1), pd.DataFrame(ohe_result, columns=ohe_cols)], axis=1)
                        encoders[col] = (ohe, ohe_cols)
                
                # 保存预处理组件
                st.session_state.preprocess["imputer"] = imputer
                st.session_state.preprocess["scaler"] = scaler
                st.session_state.preprocess["encoders"] = encoders
                st.session_state.preprocess["feature_cols"] = list(X.columns)
                
                # 保存预处理后的数据
                st.session_state.data["X_processed"] = X
                st.session_state.data["y_processed"] = y
                
                st.success("데이터 전처리 완료！")
                st.markdown(f"전처리 후 특징 수：{len(X.columns)}")
                st.dataframe(X.head(3), use_container_width=True)
            except Exception as e:
                st.error(f"전처리 실패：{str(e)}")

# ----------------------
#  단계 4：모델 학습
# ----------------------
elif st.session_state.step == 4:
    st.subheader("🚀 하이브리드모형 학습（회귀 분석 + 의사결정나무）")
    
    if "X_processed" not in st.session_state.data or "y_processed" not in st.session_state.data:
        st.warning("먼저「데이터 전처리」단계를 완료하세요")
    else:
        X = st.session_state.data["X_processed"]
        y = st.session_state.data["y_processed"]
        
        # 训练设置
        st.markdown("### 학습 설정")
        test_size = st.slider("테스트集 비율", min_value=0.1, max_value=0.3, value=0.2, step=0.05)
        
        # 分层抽样参数设置
        stratify_param = None
        if st.session_state.task == "logit":  # 分类任务
            y_unique_count = y.nunique()
            if y_unique_count >= 2:
                y_value_counts = y.value_counts()
                if (y_value_counts >= 1).all():
                    stratify_param = y
                    st.info(f"✅分层抽样 적용：目标变量에 {y_unique_count} 个类别 존재（样本数：{y_value_counts.to_dict()}）")
                else:
                    st.warning(f"⚠️  일부类别样本数为0，分层抽样禁用")
            else:
                st.warning(f"⚠️ 目标变量只有 {y_unique_count} 个类别，分层抽样禁用")
        else:
            st.info("ℹ️ 回归任务不支持分层抽样，使用普通随机抽样")
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, 
            stratify=stratify_param
        )
        
        # 模型选择
        if st.session_state.task == "logit":  # 分类
            reg_model = LogisticRegression(max_iter=1000)
            dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
        else:  # 回归
            reg_model = LinearRegression()
            dt_model = DecisionTreeRegressor(random_state=42, max_depth=10)
        
        # 模型训练
        if st.button("모델 학습 시작"):
            with st.spinner("모델 학습 중..."):
                try:
                    reg_model.fit(X_train, y_train)
                    dt_model.fit(X_train, y_train)
                    
                    # 保存模型和数据集
                    st.session_state.models["regression"] = reg_model
                    st.session_state.models["decision_tree"] = dt_model
                    st.session_state.data["X_train"] = X_train
                    st.session_state.data["X_test"] = X_test
                    st.session_state.data["y_train"] = y_train
                    st.session_state.data["y_test"] = y_test
                    
                    st.success("모델 학습 완료！")
                    st.markdown("✅ 학습된 모델：")
                    st.markdown("- 회귀 분석（로지스틱/선형，해석력 강함）")
                    st.markdown("- 의사결정나무（분류/회귀，정확도 높음）")
                    st.markdown("- 하이브리드모형（전两者 가중融合）")
                    
                    # 数据集信息
                    st.markdown(f"📊 학습集：{len(X_train):,} 행 | 테스트集：{len(X_test):,} 행")
                    if st.session_state.task == "logit":
                        st.markdown(f"🎯 训练集类别分布：{y_train.value_counts().to_dict()}")
                        st.markdown(f"🎯 测试集类别分布：{y_test.value_counts().to_dict()}")
                except Exception as e:
                    st.error(f"모델 학습 실패：{str(e)}")

# ----------------------
#  단계 5：모델 예측
# ----------------------
elif st.session_state.step == 5:
    st.subheader("🎯 모델 예측")
    
    if st.session_state.models["regression"] is None or st.session_state.models["decision_tree"] is None:
        st.warning("먼저「모델 학습」단계를 완료하세요")
    else:
        # 预测函数
        def predict(input_data):
            X = input_data.copy()
            preprocess = st.session_state.preprocess
            num_cols = X.select_dtypes(include=["int64", "float64"]).columns
            cat_cols = X.select_dtypes(include=["object", "category"]).columns
            
            # 数值型预处理
            X[num_cols] = preprocess["imputer"].transform(X[num_cols])
            X[num_cols] = preprocess["scaler"].transform(X[num_cols])
            
            # 类别型预处理
            for col in cat_cols:
                X[col] = X[col].fillna("알 수 없음").astype(str)
                encoder = preprocess["encoders"][col]
                
                if isinstance(encoder, LabelEncoder):
                    # 处理未见过的类别
                    unknown_vals = [x for x in X[col].unique() if x not in encoder.classes_]
                    if unknown_vals:
                        X[col] = X[col].replace(unknown_vals, "알 수 없음")
                    if "알 수 없음" not in encoder.classes_:
                        encoder.classes_ = np.append(encoder.classes_, "알 수 없음")
                    X[col] = encoder.transform(X[col])
                else:  # OneHotEncoder
                    ohe, ohe_cols = encoder
                    ohe_result = ohe.transform(X[[col]])
                    X = pd.concat([X.drop(col, axis=1), pd.DataFrame(ohe_result, columns=ohe_cols)], axis=1)
            
            # 确保特征列顺序一致
            X = X.reindex(columns=preprocess["feature_cols"], fill_value=0)
            
            # 混合模型预测
            reg_weight = st.session_state.models["mixed_weights"]["regression"]
            dt_weight = st.session_state.models["mixed_weights"]["decision_tree"]
            reg_model = st.session_state.models["regression"]
            dt_model = st.session_state.models["decision_tree"]
            
            if st.session_state.task == "logit":  # 分类
                reg_proba = reg_model.predict_proba(X)[:, 1]
                dt_proba = dt_model.predict_proba(X)[:, 1]
                mixed_proba = reg_weight * reg_proba + dt_weight * dt_proba
                pred = (mixed_proba >= 0.5).astype(int)
                return pred, mixed_proba
            else:  # 回归
                reg_pred = reg_model.predict(X)
                dt_pred = dt_model.predict(X)
                mixed_pred = reg_weight * reg_pred + dt_weight * dt_pred
                return mixed_pred, None
        
        # 预测方式选择
        predict_mode = st.radio("예측 방식", options=["단일 데이터 입력", "일괄 업로드 CSV"])
        
        # 单数据输入
        if predict_mode == "단일 데이터 입력":
            st.markdown("#### 단일 데이터 입력（특징값을 입력하세요）")
            feature_cols = st.session_state.preprocess["feature_cols"]
            input_data = {}
            
            with st.form("single_pred_form"):
                # 支持更多特征（分页显示）
                total_features = len(feature_cols)
                page = st.selectbox("페이지", options=range((total_features // 9) + 1), index=0)
                start_idx = page * 9
                end_idx = start_idx + 9
                display_cols = feature_cols[start_idx:end_idx]
                
                cols = st.columns(3)
                for i, col in enumerate(display_cols):
                    with cols[i % 3]:
                        if col in st.session_state.data["X_processed"].select_dtypes(include=["int64", "float64"]).columns:
                            input_data[col] = st.number_input(col, value=0.0)
                        else:
                            unique_vals = st.session_state.data["X_processed"][col].unique()[:10]
                            input_data[col] = st.selectbox(col, options=unique_vals)
                
                submit_btn = st.form_submit_button("예측 시작")
            
            if submit_btn:
                input_df = pd.DataFrame([input_data])
                pred, proba = predict(input_df)
                
                st.divider()
                st.markdown("### 예측 결과")
                if st.session_state.task == "logit":
                    st.metric("예측 결과", "양성" if pred[0] == 1 else "음성")
                    st.metric("양성 확률", f"{proba[0]:.3f}" if proba is not None else "-")
                else:
                    st.metric("예측 결과", f"{pred[0]:.2f}")
        
        # 批量上传预测
        else:
            st.markdown("#### 일괄 업로드 CSV 예측")
            uploaded_file = st.file_uploader("특징 열을 포함한 CSV 파일 업로드", type=["csv"])
            
            if uploaded_file is not None:
                batch_df = pd.read_csv(uploaded_file)
                st.metric("업로드 데이터 양", f"{len(batch_df):,} 행")
                st.dataframe(batch_df.head(3), use_container_width=True)
                
                # 特征列检查
                required_features = st.session_state.preprocess["feature_cols"]
                missing_features = [col for col in required_features if col not in batch_df.columns]
                if missing_features:
                    st.warning(f"필요한 특징 열이 없습니다：{', '.join(missing_features)}")
                else:
                    if st.button("일괄 예측 시작"):
                        with st.spinner("예측 중..."):
                            pred, proba = predict(batch_df)
                            batch_df["하이브리드모형 예측 결과"] = pred
                            if proba is not None:
                                batch_df["양성 확률"] = proba.round(3)
                            
                            st.divider()
                            st.markdown("### 일괄 예측 결과")
                            display_cols = ["하이브리드모형 예측 결과"] + (["양성 확률"] if proba is not None else []) + feature_cols[:3]
                            st.dataframe(batch_df[display_cols], use_container_width=True)
                            
                            # 下载功能
                            csv = batch_df.to_csv(index=False, encoding="utf-8-sig")
                            st.download_button(
                                label="예측 결과 다운로드",
                                data=csv,
                                file_name="하이브리드모형_일괄예측결과.csv",
                                mime="text/csv"
                            )

# ----------------------
#  단계 6：성능 평가（完善版）
# ----------------------
elif st.session_state.step == 6:
    st.subheader("📈 모델 성능 평가")
    
    if st.session_state.models["regression"] is None or st.session_state.models["decision_tree"] is None:
        st.warning("먼저「모델 학습」단계를 완료하세요")
    else:
        X_test = st.session_state.data["X_test"]
        y_test = st.session_state.data["y_test"]
        reg_model = st.session_state.models["regression"]
        dt_model = st.session_state.models["decision_tree"]
        reg_weight = st.session_state.models["mixed_weights"]["regression"]
        dt_weight = st.session_state.models["mixed_weights"]["decision_tree"]
        
        # 计算各模型预测结果
        if st.session_state.task == "logit":  # 分类任务
            # 预测结果
            reg_pred = reg_model.predict(X_test)
            dt_pred = dt_model.predict(X_test)
            reg_proba = reg_model.predict_proba(X_test)[:, 1]
            dt_proba = dt_model.predict_proba(X_test)[:, 1]
            mixed_proba = reg_weight * reg_proba + dt_weight * dt_proba
            mixed_pred = (mixed_proba >= 0.5).astype(int)
            
            # 计算评估指标
            reg_acc = accuracy_score(y_test, reg_pred)
            dt_acc = accuracy_score(y_test, dt_pred)
            mixed_acc = accuracy_score(y_test, mixed_pred)
            
            # ROC-AUC
            reg_fpr, reg_tpr, _ = roc_curve(y_test, reg_proba)
            dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_proba)
            mixed_fpr, mixed_tpr, _ = roc_curve(y_test, mixed_proba)
            
            reg_auc = auc(reg_fpr, reg_tpr)
            dt_auc = auc(dt_fpr, dt_tpr)
            mixed_auc = auc(mixed_fpr, mixed_tpr)
            
            # 混淆矩阵
            reg_cm = confusion_matrix(y_test, reg_pred)
            dt_cm = confusion_matrix(y_test, dt_pred)
            mixed_cm = confusion_matrix(y_test, mixed_pred)
            
            # 显示指标对比
            st.markdown("### 📌 分类性能指标对比")
            metrics_df = pd.DataFrame({
                "模型": ["회귀 분석", "의사결정나무", "하이브리드모형"],
                "정확도（Accuracy）": [f"{reg_acc:.4f}", f"{dt_acc:.4f}", f"{mixed_acc:.4f}"],
                "AUC": [f"{reg_auc:.4f}", f"{dt_auc:.4f}", f"{mixed_auc:.4f}"]
            })
            st.dataframe(metrics_df, use_container_width=True)
            
            # 详细分类报告
            st.markdown("### 📋 详细分类报告（하이브리드모형）")
            report = classification_report(y_test, mixed_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
            
            # ROC曲线
            st.markdown("### 📉 ROC曲线对比")
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=reg_fpr, y=reg_tpr, mode='lines', name=f'회귀 분석 (AUC={reg_auc:.3f})'))
            fig_roc.add_trace(go.Scatter(x=dt_fpr, y=dt_tpr, mode='lines', name=f'의사결정나무 (AUC={dt_auc:.3f})'))
            fig_roc.add_trace(go.Scatter(x=mixed_fpr, y=mixed_tpr, mode='lines', name=f'하이브리드모형 (AUC={mixed_auc:.3f})'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='랜덤 예측'))
            fig_roc.update_layout(
                title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                width=800, height=600
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            
            # 混淆矩阵
            st.markdown("### 🔍 混淆矩阵对比")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("회귀 분석")
                fig_reg_cm = px.imshow(reg_cm, text_auto=True, title="Confusion Matrix", 
                                      labels=dict(x="예측값", y="실제값"),
                                      x=["음성", "양성"], y=["음성", "양성"])
                st.plotly_chart(fig_reg_cm, use_container_width=True)
            with col2:
                st.subheader("의사결정나무")
                fig_dt_cm = px.imshow(dt_cm, text_auto=True, title="Confusion Matrix", 
                                     labels=dict(x="예측값", y="실제값"),
                                     x=["음성", "양성"], y=["음성", "양성"])
                st.plotly_chart(fig_dt_cm, use_container_width=True)
            with col3:
                st.subheader("하이브리드모형")
                fig_mixed_cm = px.imshow(mixed_cm, text_auto=True, title="Confusion Matrix", 
                                        labels=dict(x="예측값", y="실제값"),
                                        x=["음성", "양성"], y=["음성", "양성"])
                st.plotly_chart(fig_mixed_cm, use_container_width=True)
        
        else:  # 回归任务
            # 预测结果
            reg_pred = reg_model.predict(X_test)
            dt_pred = dt_model.predict(X_test)
            mixed_pred = reg_weight * reg_pred + dt_weight * dt_pred
            
            # 计算评估指标
            def regression_metrics(y_true, y_pred):
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true, y_pred)
                return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2}
            
            reg_metrics = regression_metrics(y_test, reg_pred)
            dt_metrics = regression_metrics(y_test, dt_pred)
            mixed_metrics = regression_metrics(y_test, mixed_pred)
            
            # 显示指标对比
            st.markdown("### 📌 回归性能指标对比")
            metrics_df = pd.DataFrame({
                "模型": ["회귀 분석", "의사결정나무", "하이브리드모형"],
                "MAE（平均绝对误差）": [f"{reg_metrics['MAE']:.4f}", f"{dt_metrics['MAE']:.4f}", f"{mixed_metrics['MAE']:.4f}"],
                "RMSE（均方根误差）": [f"{reg_metrics['RMSE']:.4f}", f"{dt_metrics['RMSE']:.4f}", f"{mixed_metrics['RMSE']:.4f}"],
                "R²（决定系数）": [f"{reg_metrics['R²']:.4f}", f"{dt_metrics['R²']:.4f}", f"{mixed_metrics['R²']:.4f}"]
            })
            st.dataframe(metrics_df, use_container_width=True)
            
            # 预测值 vs 实际值散点图
            st.markdown("### 📊 预测值 vs 实际值")
            fig_pred = px.scatter(x=y_test, y=mixed_pred, 
                                 labels={"x": "实际值", "y": "预测值"},
                                 title="하이브리드모형：实际值 vs 预测值")
            # 添加参考线（y=x）
            min_val = min(y_test.min(), mixed_pred.min())
            max_val = max(y_test.max(), mixed_pred.max())
            fig_pred.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                                         mode='lines', line=dict(dash='dash', color='red'),
                                         showlegend=False))
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # 误差分布直方图
            st.markdown("### 📉 预测误差分布")
            errors = y_test - mixed_pred
            fig_error = px.histogram(errors, title="预测误差分布（实际值-预测值）",
                                    labels={"value": "误差值", "count": "频数"})
            fig_error.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_error, use_container_width=True)
        
        st.divider()
        st.success("모델 성능 평가 완료！")
