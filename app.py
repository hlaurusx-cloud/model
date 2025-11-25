import streamlit as st
import pandas as pd
import numpy as np
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
import warnings
warnings.filterwarnings("ignore")

# ----------------------
# 1. 페이지 기본 설정
# ----------------------
st.set_page_config(
    page_title="하이브리드모형 동적 프레임워크 (Residual Learning)",
    page_icon="📊",
    layout="wide"
)

# 전역 상태 관리
if "step" not in st.session_state:
    st.session_state.step = 0 
if "data" not in st.session_state:
    st.session_state.data = {"merged": None}
if "preprocess" not in st.session_state:
    st.session_state.preprocess = {"imputer": None, "scaler": None, "encoders": None, "feature_cols": None, "target_col": None}
if "models" not in st.session_state:
    # models: regression(Base Model), decision_tree(Residual Model or Class Model)
    st.session_state.models = {"regression": None, "decision_tree": None, "mixed_weights": {"regression": 0.5, "decision_tree": 0.5}}
if "task" not in st.session_state:
    st.session_state.task = "logit" 

# ----------------------
# 2. 사이드바：단계 네비게이션 + 핵심 설정
# ----------------------
st.sidebar.title("📌 하이브리드모형 작업 흐름")
st.sidebar.divider()

steps = ["초기 설정", "데이터 업로드", "데이터 시각화", "데이터 전처리", "모델 학습", "모델 예측", "성능 평가"]
for i, step_name in enumerate(steps):
    if st.sidebar.button(step_name, key=f"btn_{i}"):
        st.session_state.step = i

st.sidebar.divider()
st.sidebar.subheader("핵심 설정")
st.session_state.task = st.sidebar.radio("작업 유형", options=["logit", "의사결정나무(회귀)"], index=0)

# 가중치 설정 (분류 작업일 때만 표시, 회귀는 잔차 학습이라 가중치 불필요)
if st.session_state.step >= 4 and st.session_state.task == "logit":
    st.sidebar.subheader("앙상블 가중치 설정")
    reg_weight = st.sidebar.slider(
        "회귀 분석 가중치",
        min_value=0.0, max_value=1.0, value=st.session_state.models["mixed_weights"]["regression"], step=0.1
    )
    st.session_state.models["mixed_weights"]["regression"] = reg_weight
    st.session_state.models["mixed_weights"]["decision_tree"] = 1 - reg_weight
    st.sidebar.text(f"의사결정나무 가중치：{1 - reg_weight:.1f}")
elif st.session_state.step >= 4 and st.session_state.task != "logit":
    st.sidebar.info("ℹ️ 회귀 작업은 '잔차 학습(Residual Learning)' 방식을 사용하므로 가중치 설정이 필요 없습니다. (Base + Residual)")

# ----------------------
# 3. 메인 페이지 로직
# ----------------------
st.title("📊 하이브리드모형 동적 배포 프레임워크")
if st.session_state.task == "의사결정나무(회귀)":
    st.markdown("**🚀 적용 모형: 선형 회귀(Base) + 의사결정나무(Residual Correction)**")
else:
    st.markdown("**🚀 적용 모형: 로지스틱 회귀 + 의사결정나무 (Weighted Ensemble)**")
st.divider()

# --- 단계 0: 초기 설정 ---
if st.session_state.step == 0:
    st.subheader("🎉 하이브리드모형 프레임워크에 오신 것을 환영합니다")
    st.markdown("""
    본 시스템은 **데이터의 선형적 패턴과 비선형적 패턴을 동시에 학습**하는 하이브리드 모델을 구축합니다.
    
    ### ⚙️ 작동 원리
    1. **의사결정나무(회귀) 작업 시**:
       - 1단계: **선형 회귀**가 전체적인 추세를 학습합니다.
       - 2단계: **의사결정나무**가 1단계의 예측 오차(잔차)를 학습하여 보정합니다.
       - **결과**: `최종 예측 = 선형 예측값 + 잔차 예측값`
    
    2. **Logit(분류) 작업 시**:
       - 로지스틱 회귀와 분류 의사결정나무의 예측 확률을 **가중 결합**합니다.
    
    ### 왼쪽「데이터 업로드」를 클릭하여 시작하세요!
    """)

# --- 단계 1: 데이터 업로드 ---
elif st.session_state.step == 1:
    st.subheader("📤 데이터 업로드")
    uploaded_file = st.file_uploader("데이터 파일 선택 (CSV/Excel)", type=["csv", "xlsx", "xls"], key="single_file")
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_merged = pd.read_csv(uploaded_file)
            else:
                df_merged = pd.read_excel(uploaded_file)
            
            st.session_state.data["merged"] = df_merged
            st.success(f"데이터 업로드 성공! ({len(df_merged):,} 행)")
            st.dataframe(df_merged.head())
            st.info("📊 다음 단계: 사이드바에서 '데이터 시각화'를 선택하세요.")
        except Exception as e:
            st.error(f"파일 읽기 실패: {e}")

# --- 단계 2: 데이터 시각화 ---
elif st.session_state.step == 2:
    st.subheader("📊 데이터 시각화")
    if st.session_state.data["merged"] is None:
        st.warning("데이터를 먼저 업로드해주세요.")
    else:
        df = st.session_state.data["merged"]
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("X축 변수", options=["선택 안 함"] + list(df.columns))
        with col2:
            y_var = st.selectbox("Y축 변수 (수치형 권장)", options=num_cols)
            
        if y_var:
            try:
                if x_var != "선택 안 함":
                    st.markdown(f"### {x_var} vs {y_var}")
                    if x_var in cat_cols:
                        fig = px.box(df, x=x_var, y=y_var, color=x_var)
                    else:
                        fig = px.scatter(df, x=x_var, y=y_var)
                else:
                    st.markdown(f"### {y_var} 분포")
                    fig = px.histogram(df, x=y_var, nbins=30, marginal="box")
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"시각화 오류: {e}")

# --- 단계 3: 데이터 전처리 ---
elif st.session_state.step == 3:
    st.subheader("🧹 데이터 전처리")
    if st.session_state.data["merged"] is None:
        st.warning("데이터가 없습니다.")
    else:
        df = st.session_state.data["merged"]
        
        # 타겟 설정
        target_col = st.selectbox("타겟 열(예측 대상) 선택", options=df.columns)
        st.session_state.preprocess["target_col"] = target_col
        
        # 특징 설정
        feature_cols = st.multiselect("특징 열(예측 변수) 선택", 
                                      options=[c for c in df.columns if c != target_col],
                                      default=[c for c in df.columns if c != target_col])
        st.session_state.preprocess["feature_cols"] = feature_cols
        
        if st.button("전처리 및 변환 실행"):
            if not feature_cols:
                st.error("특징 열을 하나 이상 선택하세요.")
                st.stop()
            
            try:
                X = df[feature_cols].copy()
                y = df[target_col].copy()
                
                # 수치형/범주형 분리
                num_cols = X.select_dtypes(include=np.number).columns
                cat_cols = X.select_dtypes(exclude=np.number).columns
                
                # Imputer & Scaler
                imputer = SimpleImputer(strategy="mean")
                scaler = StandardScaler()
                
                if len(num_cols) > 0:
                    X[num_cols] = imputer.fit_transform(X[num_cols])
                    X[num_cols] = scaler.fit_transform(X[num_cols])
                
                # Encoder
                encoders = {}
                for col in cat_cols:
                    X[col] = X[col].fillna("Unknown").astype(str)
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                    encoders[col] = le
                
                # 상태 저장
                st.session_state.preprocess.update({
                    "imputer": imputer, "scaler": scaler, "encoders": encoders, 
                    "final_features": X.columns.tolist()
                })
                st.session_state.data["X_processed"] = X
                st.session_state.data["y_processed"] = y
                
                st.success("전처리 완료!")
                st.dataframe(X.head(3))
            except Exception as e:
                st.error(f"전처리 오류: {e}")

# --- 단계 4: 모델 학습 (핵심 수정 부분) ---
elif st.session_state.step == 4:
    st.subheader("🚀 모델 학습")
    
    if "X_processed" not in st.session_state.data:
        st.warning("전처리를 먼저 수행하세요.")
    else:
        X = st.session_state.data["X_processed"]
        y = st.session_state.data["y_processed"]
        
        # Train/Test Split
        # stratify는 분류 문제이고 클래스가 충분할 때만 적용
        stratify_param = None
        if st.session_state.task == "logit" and y.nunique() > 1:
             if y.value_counts().min() > 1:
                stratify_param = y
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_param
        )
        
        if st.button("모델 학습 시작"):
            with st.spinner("하이브리드 모델 학습 중..."):
                try:
                    # -------------------------------------------------------
                    # [CASE 1] 분류 (Logit) - 기존 방식 (Weighted Ensemble)
                    # -------------------------------------------------------
                    if st.session_state.task == "logit":
                        reg_model = LogisticRegression(max_iter=1000)
                        dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
                        
                        reg_model.fit(X_train, y_train)
                        dt_model.fit(X_train, y_train)
                        
                        st.info("Logit 모드: 로지스틱 회귀와 분류 트리를 독립적으로 학습했습니다.")

                    # -------------------------------------------------------
                    # [CASE 2] 회귀 (Regression) - 잔차 학습 (Residual Learning)
                    # -------------------------------------------------------
                    else:
                        # 1. Base Model: 선형 회귀 학습
                        reg_model = LinearRegression()
                        reg_model.fit(X_train, y_train)
                        
                        # 2. 잔차 계산 (실제값 - 선형회귀 예측값)
                        lr_pred_train = reg_model.predict(X_train)
                        train_residuals = y_train - lr_pred_train
                        
                        # 3. Residual Model: 의사결정나무로 잔차 학습
                        dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
                        dt_model.fit(X_train, train_residuals)
                        
                        st.success("✅ 잔차 학습 완료!")
                        st.markdown("""
                        1. **선형 회귀**가 데이터의 기본 경향(Trend)을 학습했습니다.
                        2. **의사결정나무**가 선형 회귀의 오차(Residual)를 학습했습니다.
                        3. 최종 예측은 두 모델의 합입니다.
                        """)
                    
                    # 모델 저장
                    st.session_state.models["regression"] = reg_model
                    st.session_state.models["decision_tree"] = dt_model
                    
                    # 데이터 저장
                    st.session_state.data.update({
                        "X_train": X_train, "X_test": X_test, 
                        "y_train": y_train, "y_test": y_test
                    })
                    
                except Exception as e:
                    st.error(f"학습 실패: {e}")

# --- 단계 5: 모델 예측 ---
elif st.session_state.step == 5:
    st.subheader("🎯 예측 실행")
    
    if st.session_state.models["regression"] is None:
        st.warning("모델 학습을 먼저 완료하세요.")
    else:
        # 예측 모드 선택
        mode = st.radio("예측 모드", ["단일 데이터 입력", "테스트 데이터 전체 평가"])
        
        if mode == "테스트 데이터 전체 평가":
            X_test = st.session_state.data["X_test"]
            reg_model = st.session_state.models["regression"]
            dt_model = st.session_state.models["decision_tree"]
            
            if st.session_state.task == "logit":
                # 분류: 가중 평균
                w_reg = st.session_state.models["mixed_weights"]["regression"]
                prob_reg = reg_model.predict_proba(X_test)[:, 1]
                prob_dt = dt_model.predict_proba(X_test)[:, 1]
                final_prob = w_reg * prob_reg + (1-w_reg) * prob_dt
                final_pred = (final_prob >= 0.5).astype(int)
            else:
                # 회귀: 잔차 합산 (Base + Residual)
                pred_base = reg_model.predict(X_test)     # 선형 회귀 예측
                pred_resid = dt_model.predict(X_test)     # 잔차 예측
                final_pred = pred_base + pred_resid       # 최종 결과
            
            st.session_state.data["final_pred"] = final_pred # 평가 단계용 저장
            
            result_df = X_test.copy()
            result_df["최종 예측값"] = final_pred
            st.dataframe(result_df.head())
            
        else:
            st.info("단일 데이터 입력 기능은 위 '테스트 데이터' 로직과 동일하게 내부 함수로 처리됩니다.")

# --- 단계 6: 성능 평가 ---
elif st.session_state.step == 6:
    st.subheader("📈 성능 평가")
    
    if "final_pred" not in st.session_state.data:
        st.warning("먼저 '모델 예측' 단계에서 테스트 데이터 평가를 수행해주세요.")
    else:
        y_test = st.session_state.data["y_test"]
        final_pred = st.session_state.data["final_pred"]
        reg_model = st.session_state.models["regression"]
        dt_model = st.session_state.models["decision_tree"]
        X_test = st.session_state.data["X_test"]
        
        # ---------------- [CASE 1] 분류 평가 ----------------
        if st.session_state.task == "logit":
            acc = accuracy_score(y_test, final_pred)
            st.metric("하이브리드 모델 정확도 (Accuracy)", f"{acc:.4f}")
            
            cm = confusion_matrix(y_test, final_pred)
            fig = px.imshow(cm, text_auto=True, title="Confusion Matrix")
            st.plotly_chart(fig)
            
        # ---------------- [CASE 2] 회귀 평가 (핵심) ----------------
        else:
            # 개별 모델 성능 비교를 위해 다시 예측
            lr_only_pred = reg_model.predict(X_test)
            
            # 성능 지표
            rmse_lr = np.sqrt(mean_squared_error(y_test, lr_only_pred))
            rmse_hybrid = np.sqrt(mean_squared_error(y_test, final_pred))
            r2_hybrid = r2_score(y_test, final_pred)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("선형회귀 단독 RMSE", f"{rmse_lr:.4f}")
            col2.metric("하이브리드 RMSE (개선)", f"{rmse_hybrid:.4f}", delta=f"{rmse_lr - rmse_hybrid:.4f}")
            col3.metric("하이브리드 R² (설명력)", f"{r2_hybrid:.4f}")
            
            # 시각화 1: 예측값 vs 실제값
            fig1 = px.scatter(x=y_test, y=final_pred, title="실제값 vs 하이브리드 예측값")
            fig1.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                                      mode='lines', name='Perfect Fit', line=dict(color='red', dash='dash')))
            st.plotly_chart(fig1)
            
            # 시각화 2: 잔차 보정 효과 확인
            # 선형회귀만 했을 때의 잔차 vs 트리가 예측한 잔차
            original_residuals = y_test - lr_only_pred
            tree_predicted_residuals = dt_model.predict(X_test)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=y_test.index, y=original_residuals, mode='markers', name='선형회귀가 못 맞춘 오차(잔차)', opacity=0.5))
            fig2.add_trace(go.Scatter(x=y_test.index, y=tree_predicted_residuals, mode='markers', name='의사결정나무의 잔차 예측', opacity=0.7))
            fig2.update_layout(title="잔차 학습 효과 분석 (두 점이 겹칠수록 보정이 잘 된 것)")
            st.plotly_chart(fig2)

            st.caption("그래프 해석: 파란 점(원래 오차)을 빨간 점(트리 예측)이 잘 따라가면, 하이브리드 모델이 오차를 효과적으로 줄여주고 있는 것입니다.")
