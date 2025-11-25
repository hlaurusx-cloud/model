import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    mean_squared_error, r2_score
)
import re
import warnings

# 경고 무시
warnings.filterwarnings("ignore")

# ----------------------
# 1. 페이지 기본 설정
# ----------------------
st.set_page_config(
    page_title="하이브리드모형 개발 (Smart Cleaning)",
    page_icon="📊",
    layout="wide"
)

# 전역 상태 관리
if "step" not in st.session_state:
    st.session_state.step = 0 
if "data" not in st.session_state:
    st.session_state.data = {"merged": None}
if "preprocess" not in st.session_state:
    st.session_state.preprocess = {"imputer": None, "scaler": None, "encoders": None}
if "models" not in st.session_state:
    # models 저장소: 회귀(base), 트리(residual or class), 가중치
    st.session_state.models = {"regression": None, "decision_tree": None, "mixed_weights": {"regression": 0.5}}
if "task" not in st.session_state:
    st.session_state.task = "logit" 

# ----------------------
# 2. 기능 함수 정의 (핵심 수정 사항)
# ----------------------

def smart_clean_data(df):
    """
    업로드된 데이터의 특수문자(%, years 등)를 제거하고 수치형으로 변환합니다.
    """
    df_clean = df.copy()
    
    # 1. 퍼센트(%) 제거 및 실수 변환 (int_rate, revol_util 등)
    # 데이터프레임의 모든 object 컬럼을 순회하며 %가 포함된 경우 변환 시도
    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            # 샘플 데이터를 확인하여 '%'가 포함된 경우
            if df_clean[col].astype(str).str.contains('%').any():
                try:
                    # % 제거 및 공백 제거 후 float 변환
                    df_clean[col] = df_clean[col].str.replace('%', '').str.strip().astype(float)
                except:
                    pass # 변환 실패 시 원본 유지

    # 2. 근속 연수(emp_length) 숫자 추출 ('10+ years' -> 10, '< 1 year' -> 0)
    if 'emp_length' in df_clean.columns:
        def clean_emp(val):
            if pd.isna(val): return np.nan
            val = str(val)
            if '<' in val: return 0
            # 숫자만 추출
            nums = re.findall(r'\d+', val)
            return int(nums[0]) if nums else 0
        
        df_clean['emp_length'] = df_clean['emp_length'].apply(clean_emp)
        
    return df_clean

# ----------------------
# 3. 사이드바 네비게이션
# ----------------------
st.sidebar.title("📌 분석 프로세스")
steps = ["1. 데이터 업로드", "2. 데이터 시각화", "3. 데이터 전처리", "4. 모델 학습", "5. 결과 평가"]

# 단계 이동 버튼
for i, step_name in enumerate(steps):
    if st.sidebar.button(step_name, key=f"step_btn_{i}"):
        st.session_state.step = i + 1

st.sidebar.divider()
st.sidebar.subheader("⚙️ 모델 설정")
st.session_state.task = st.sidebar.radio(
    "작업 유형 선택", 
    ["logit (분류: 승인/거절)", "regression (회귀: 이자율 예측)"]
)

if st.session_state.task == "logit" and st.session_state.step >= 4:
    st.sidebar.markdown("---")
    weight = st.sidebar.slider("회귀 모델 가중치 (0~1)", 0.0, 1.0, 0.5)
    st.session_state.models["mixed_weights"]["regression"] = weight

# ----------------------
# 4. 메인 페이지 로직
# ----------------------
st.title("📊 하이브리드모형 개발 프레임워크")

# --- 단계 1: 데이터 업로드 ---
if st.session_state.step == 1:
    st.subheader("📤 1. 데이터 파일 업로드")
    st.info("csv 파일을 업로드하면 자동으로 '%' 기호 등을 처리하여 숫자로 변환합니다.")
    
    uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])
    
    if uploaded_file:
        try:
            raw_df = pd.read_csv(uploaded_file)
            
            # [수정] 스마트 클리닝 함수 적용
            df_cleaned = smart_clean_data(raw_df)
            
            st.session_state.data["merged"] = df_cleaned
            st.success(f"데이터 로드 및 정제 완료! ({len(df_cleaned)} 행)")
            
            st.markdown("#### ▼ 데이터 미리보기 (전처리 전 원본 확인)")
            st.dataframe(df_cleaned.head())
            
        except Exception as e:
            st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

# --- 단계 2: 데이터 시각화 ---
elif st.session_state.step == 2:
    st.subheader("📈 2. 데이터 시각화")
    df = st.session_state.data.get("merged")
    
    if df is None:
        st.warning("데이터를 먼저 업로드하세요.")
    else:
        # 수치형 컬럼만 추출
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            x_val = st.selectbox("X축 변수", df.columns)
        with col2:
            y_val = st.selectbox("Y축 변수 (수치형)", num_cols)
            
        if st.button("차트 생성"):
            if x_val in num_cols and y_val in num_cols:
                fig = px.scatter(df, x=x_val, y=y_val, title=f"{x_val} vs {y_val} 산점도")
            else:
                fig = px.box(df, x=x_val, y=y_val, title=f"{x_val}별 {y_val} 분포")
            st.plotly_chart(fig, use_container_width=True)

# --- 단계 3: 데이터 전처리 (결측치 및 인코딩) ---
elif st.session_state.step == 3:
    st.subheader("🛠 3. 변수 선택 및 인코딩")
    df = st.session_state.data.get("merged")
    
    if df is None:
        st.warning("데이터가 없습니다.")
    else:
        # 타겟 변수 선택
        target_col = st.selectbox("타겟 변수(예측할 값)를 선택하세요", df.columns, index=len(df.columns)-1)
        st.session_state.preprocess["target_col"] = target_col
        
        # 특징 변수 선택 (타겟 제외)
        feature_cols = st.multiselect(
            "학습에 사용할 특징(X)을 선택하세요", 
            [c for c in df.columns if c != target_col],
            default=[c for c in df.columns if c != target_col][:5] # 기본적으로 앞의 5개 선택
        )
        
        if st.button("전처리 실행 (결측치 처리 & 인코딩)"):
            if not feature_cols:
                st.error("특징 변수를 하나 이상 선택해주세요.")
            else:
                try:
                    X = df[feature_cols].copy()
                    y = df[target_col].copy()
                    
                    # 결측치 처리 (수치형: 평균, 범주형: 최빈값)
                    num_features = X.select_dtypes(include=np.number).columns
                    cat_features = X.select_dtypes(exclude=np.number).columns
                    
                    imputer_num = SimpleImputer(strategy='mean')
                    imputer_cat = SimpleImputer(strategy='most_frequent')
                    
                    if len(num_features) > 0:
                        X[num_features] = imputer_num.fit_transform(X[num_features])
                    if len(cat_features) > 0:
                        X[cat_features] = imputer_cat.fit_transform(X[cat_features])
                    
                    # 라벨 인코딩 (범주형 -> 숫자)
                    encoders = {}
                    for col in cat_features:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                        encoders[col] = le
                        
                    # 전처리된 데이터 저장
                    st.session_state.data["X_processed"] = X
                    st.session_state.data["y_processed"] = y
                    st.session_state.preprocess["feature_cols"] = feature_cols
                    
                    st.success("전처리가 완료되었습니다!")
                    st.dataframe(X.head())
                    
                except Exception as e:
                    st.error(f"전처리 중 오류: {e}")

# --- 단계 4: 모델 학습 (핵심 로직 수정) ---
elif st.session_state.step == 4:
    st.subheader("🤖 4. 하이브리드 모델 학습")
    
    if "X_processed" not in st.session_state.data:
        st.warning("3단계 전처리를 먼저 완료해주세요.")
    else:
        X = st.session_state.data["X_processed"]
        y = st.session_state.data["y_processed"]
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if st.button("모델 학습 시작"):
            with st.spinner("모델을 학습 중입니다..."):
                try:
                    # [CASE 1] 회귀 (Regression): 잔차 학습 (Residual Learning)
                    if st.session_state.task == "regression":
                        # 1. Base Model: 선형 회귀
                        lr = LinearRegression()
                        lr.fit(X_train, y_train)
                        
                        # 2. 잔차 계산 (실제값 - 선형회귀 예측값)
                        train_pred = lr.predict(X_train)
                        train_residuals = y_train - train_pred
                        
                        # 3. Residual Model: 의사결정나무 (잔차 예측)
                        dt = DecisionTreeRegressor(max_depth=5, random_state=42)
                        dt.fit(X_train, train_residuals)
                        
                        st.session_state.models["regression"] = lr
                        st.session_state.models["decision_tree"] = dt
                        
                        st.success("✅ 회귀 하이브리드 모델 학습 완료! (선형회귀 + 잔차 트리)")
                        
                    # [CASE 2] 분류 (Logit): 앙상블 (Ensemble)
                    else:
                        # 타겟이 숫자가 아닐 경우를 대비해 인코딩
                        if y_train.dtype == object:
                            le_target = LabelEncoder()
                            y_train = le_target.fit_transform(y_train)
                            y_test = le_target.transform(y_test)
                            st.session_state.preprocess["le_target"] = le_target
                            
                        lr = LogisticRegression(max_iter=1000)
                        dt = DecisionTreeClassifier(max_depth=5, random_state=42)
                        
                        lr.fit(X_train, y_train)
                        dt.fit(X_train, y_train)
                        
                        st.session_state.models["regression"] = lr
                        st.session_state.models["decision_tree"] = dt
                        
                        st.success("✅ 분류 하이브리드 모델 학습 완료! (로지스틱 + 의사결정나무)")
                    
                    # 테스트 셋 저장
                    st.session_state.data["X_test"] = X_test
                    st.session_state.data["y_test"] = y_test
                    
                except Exception as e:
                    st.error(f"학습 중 오류 발생: {e}")

# --- 단계 5: 결과 평가 ---
elif st.session_state.step == 5:
    st.subheader("🏆 5. 모델 성능 평가")
    
    if st.session_state.models["regression"] is None:
        st.warning("모델 학습을 먼저 진행해주세요.")
    else:
        X_test = st.session_state.data["X_test"]
        y_test = st.session_state.data["y_test"]
        
        lr_model = st.session_state.models["regression"]
        dt_model = st.session_state.models["decision_tree"]
        
        # [CASE 1] 회귀 평가 (Residual Method)
        if st.session_state.task == "regression":
            # 예측
            pred_base = lr_model.predict(X_test)       # 선형 회귀 예측
            pred_resid = dt_model.predict(X_test)      # 잔차 예측
            final_pred = pred_base + pred_resid        # 최종 합산
            
            # 성능 지표
            mse_base = mean_squared_error(y_test, pred_base)
            mse_hybrid = mean_squared_error(y_test, final_pred)
            r2 = r2_score(y_test, final_pred)
            
            # 메트릭 표시
            c1, c2, c3 = st.columns(3)
            c1.metric("선형회귀 MSE", f"{mse_base:.4f}")
            c2.metric("하이브리드 MSE", f"{mse_hybrid:.4f}", delta=f"{mse_base - mse_hybrid:.4f} 개선")
            c3.metric("R² (설명력)", f"{r2:.4f}")
            
            # 시각화: 실제값 vs 예측값
            viz_df = pd.DataFrame({'Actual': y_test, 'Predicted': final_pred})
            fig = px.scatter(viz_df, x='Actual', y='Predicted', title="실제값 vs 하이브리드 예측값")
            # 기준선 추가
            fig.add_trace(go.Scatter(
                x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                mode='lines', name='정답 라인', line=dict(color='red', dash='dash')
            ))
            st.plotly_chart(fig, use_container_width=True)
            
        # [CASE 2] 분류 평가 (Ensemble Method)
        else:
            # 확률 예측
            prob_lr = lr_model.predict_proba(X_test)[:, 1]
            prob_dt = dt_model.predict_proba(X_test)[:, 1]
            
            w = st.session_state.models["mixed_weights"]["regression"]
            final_prob = (w * prob_lr) + ((1 - w) * prob_dt)
            final_pred = (final_prob >= 0.5).astype(int)
            
            acc = accuracy_score(y_test, final_pred)
            cm = confusion_matrix(y_test, final_pred)
            
            st.metric("정확도 (Accuracy)", f"{acc:.4f}")
            
            fig = px.imshow(cm, text_auto=True, title="Confusion Matrix",
                            labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig, use_container_width=True)
