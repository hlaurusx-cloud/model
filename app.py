import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import re

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="이자율 예측 하이브리드 모델", layout="wide")
st.title("💰 대출 이자율(Interest Rate) 예측 하이브리드 모델")
st.markdown("""
이 시스템은 **선형 회귀(Linear Regression)**와 **의사결정나무(Decision Tree)**를 결합하여 
대출 이자율을 정밀하게 예측합니다. **아래 버튼을 눌러 데이터 파일을 업로드해주세요.**
""")

# --- 2. 데이터 로드 및 전처리 함수 ---
@st.cache_data
def load_and_preprocess_data(file):
    # 업로드된 파일을 바로 pandas로 읽습니다
    df = pd.read_csv(file)
    
    # 1. 이자율(Target) 전처리: ' 10.37%' -> 10.37 (float)
    if df['int_rate'].dtype == object:
        df['int_rate'] = df['int_rate'].str.strip().str.replace('%', '').astype(float)
        
    # 2. 리볼빙 사용률 전처리: '86.6%' -> 86.6
    if 'revol_util' in df.columns and df['revol_util'].dtype == object:
        df['revol_util'] = df['revol_util'].str.strip().str.replace('%', '').astype(float)

    # 3. 근속 연수 전처리: 숫자만 추출 (예: '10+ years' -> 10)
    def clean_emp_length(val):
        if pd.isna(val): return 0
        val = str(val)
        if '<' in val: return 0
        numbers = re.findall(r'\d+', val)
        if numbers: return int(numbers[0])
        return 0
    
    if 'emp_length' in df.columns:
        df['emp_length_clean'] = df['emp_length'].apply(clean_emp_length)
    
    # 4. 사용할 주요 변수 선택 (수치형 위주)
    features = ['loan_amnt', 'annual_inc', 'dti', 'emp_length_clean', 'revol_util', 'total_acc']
    target = 'int_rate'
    
    # 결측치 처리 (평균값 대치)
    df_model = df[features + [target]].copy()
    df_model = df_model.fillna(df_model.mean())
    
    return df_model

# --- 3. 메인 로직 실행 ---

# 파일 업로더 위젯 추가
uploaded_file = st.file_uploader("📂 'loanstats_15000_cleaned.csv' 파일을 여기에 드래그하거나 선택하세요", type=['csv'])

if uploaded_file is not None:
    try:
        data = load_and_preprocess_data(uploaded_file)
        
        st.subheader("1. 데이터 미리보기 (전처리 완료)")
        st.dataframe(data.head())
        
        # 학습/테스트 데이터 분리
        X = data.drop('int_rate', axis=1)
        y = data['int_rate']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        st.markdown("---")
        st.subheader("2. 모델 학습 진행")
        
        col1, col2 = st.columns(2)
        
        # --- Step 1: 선형 회귀 (Base Model) ---
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred_train = lr.predict(X_train)
        lr_pred_test = lr.predict(X_test)
        
        # 잔차 계산 (실제값 - 선형회귀 예측값)
        train_residuals = y_train - lr_pred_train
        
        with col1:
            st.info("🔹 1단계: 선형 회귀 완료")
            st.write("전체적인 선형 트렌드를 학습했습니다.")
            
        # --- Step 2: 의사결정나무 (Residual Model) ---
        dt = DecisionTreeRegressor(max_depth=5, random_state=42)
        dt.fit(X_train, train_residuals)
        dt_pred_test_residuals = dt.predict(X_test)
        
        with col2:
            st.success("🔸 2단계: 잔차 학습(Tree) 완료")
            st.write("선형 회귀가 놓친 비선형 패턴(오차)을 학습했습니다.")
            
        # --- Step 3: 최종 결합 ---
        final_pred = lr_pred_test + dt_pred_test_residuals
        
        st.markdown("---")
        st.subheader("3. 성능 평가 및 시각화")
        
        # 성능 지표 계산
        mse_lr = mean_squared_error(y_test, lr_pred_test)
        mse_hybrid = mean_squared_error(y_test, final_pred)
        r2_hybrid = r2_score(y_test, final_pred)
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("선형회귀 단독 MSE", f"{mse_lr:.4f}")
        m_col2.metric("하이브리드 모델 MSE", f"{mse_hybrid:.4f}", delta=f"{mse_lr - mse_hybrid:.4f} (개선)")
        m_col3.metric("하이브리드 모델 R²", f"{r2_hybrid:.4f}")
        
        # 시각화 (Matplotlib)
        st.markdown("#### 📊 실제 이자율 vs 예측 이자율 비교")
        viz_df = pd.DataFrame({'Actual': y_test, 'Predicted': final_pred}).reset_index(drop=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(viz_df['Actual'], viz_df['Predicted'], alpha=0.5, color='blue', label='Data Points')
        ax.plot([viz_df['Actual'].min(), viz_df['Actual'].max()], 
                [viz_df['Actual'].min(), viz_df['Actual'].max()], 
                'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel("Actual Interest Rate (%)")
        ax.set_ylabel("Predicted Interest Rate (%)")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")

else:
    st.warning("⚠️ 분석을 시작하려면 CSV 파일을 업로드해주세요.")
