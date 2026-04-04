"""
信贷风险评估系统 / Credit Risk Assessment System - Streamlit Web 应用
支持中英文切换 / Supports Chinese and English

使用方法 / Usage:
    streamlit run app.py

作者 / Author: COMP5572 Project Team
日期 / Date: 2026-04-04
"""

# ========== 1. 导入依赖 / Import Dependencies ==========
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import json
from datetime import datetime
from pathlib import Path

from sklearn.preprocessing import OrdinalEncoder
import plotly.graph_objects as go
import plotly.express as px

# ========== 2. 语言字典 / Language Dictionary ==========
TRANSLATIONS = {
    'zh': {
        # 页面配置
        'page_title': '信贷风险评估系统',
        'page_icon': '🏦',

        # 侧边栏
        'sidebar_title': '🏦 信贷风险评估',
        'select_function': '选择功能',

        # 页面名称
        'page_single': '📝 单客户预测',
        'page_batch': '📤 批量预测',
        'page_report': '📊 模型报告',
        'page_history': '📜 预测历史',
        'page_about': 'ℹ️ 关于',

        # 单客户预测
        'single_header': '📝 单客户风险评估',
        'enter_info': '请输入客户信息',
        'basic_info': '📋 基本信息',
        'financial_info': '💰 财务信息',
        'employment_info': '💼 就业信息',
        'asset_info': '🏠 资产信息',
        'external_scores': '📊 外部征信评分',
        'start_assessment': '🔍 开始评估',
        'load_sample': '📋 加载示例数据',

        # 基本信息
        'gender': '性别',
        'age': '年龄',
        'age_hint': '将自动转换为出生天数',
        'marital_status': '婚姻状况',
        'children': '子女数量',
        'family_members': '家庭成员数',

        # 财务信息
        'annual_income': '年收入（元）',
        'loan_amount': '贷款金额（元）',
        'monthly_payment': '月供（元）',
        'goods_price': '商品价格（元）',

        # 就业信息
        'income_type': '收入类型',
        'occupation': '职业类型',
        'employment_years': '工作年限（年）',
        'employment_hint': '将自动转换为工作天数',

        # 资产信息
        'own_car': '是否有车 (FLAG_OWN_CAR)',
        'own_realty': '是否有房 (FLAG_OWN_REALTY)',
        'car_age': '车龄 (OWN_CAR_AGE)',
        'housing_type': '住房类型 (NAME_HOUSING_TYPE)',

        # 外部评分
        'ext_score_1': '外部评分1 (EXT_SOURCE_1)',
        'ext_score_2': '外部评分2 (EXT_SOURCE_2)',
        'ext_score_3': '外部评分3 (EXT_SOURCE_3)',

        # 性别选项
        'gender_male': '男',
        'gender_female': '女',

        # 婚姻状况
        'status_married': '已婚',
        'status_single': '单身',
        'status_civil': '未婚同居',
        'status_widow': '丧偶',
        'status_separated': '分居',

        # 收入类型
        'income_working': '在职',
        'income_state': '公务员',
        'income_pensioner': '退休',
        'income_commercial': '个体工商户',

        # 职业
        'occupation_laborers': '劳工',
        'occupation_core': '核心员工',
        'occupation_sales': '销售',
        'occupation_managers': '经理',
        'occupation_drivers': '司机',
        'occupation_other': '其他',

        # 住房类型
        'housing_house': 'House / apartment',
        'housing_rented': 'Rented apartment',
        'housing_parents': 'With parents',
        'housing_other': 'Other',

        # 结果
        'results_header': '📊 评估结果',
        'default_prob': '违约概率',
        'decision_suggestion': '决策建议:',
        'risk_assessment': '风险评估',
        'description': '💡 {desc}',

        # 风险等级
        'risk_low': '低风险',
        'risk_medium': '中等风险',
        'risk_high': '高风险',
        'decision_approve': '建议批准',
        'decision_review': '需要考量',
        'decision_reject': '建议拒绝',
        'desc_low': '客户违约风险较低，可以放心批准贷款',
        'desc_medium': '客户有一定违约风险，建议要求额外担保或降低贷款金额',
        'desc_high': '客户违约风险较高，不建议批准贷款',

        # SHAP 分析
        'shap_header': '🔍 决策依据分析',
        'positive_factors': '✅ 有利因素（降低风险）',
        'negative_factors': '⚠️ 风险因素（增加风险）',

        # 批量预测
        'batch_header': '📤 批量预测（CSV上传）',
        'step1': '步骤1: 下载CSV模板',
        'download_template': '📥 下载CSV模板',
        'step2': '步骤2: 上传数据文件',
        'select_csv': '选择CSV文件',
        'loaded_rows': '✅ 已读取 {n} 行数据',
        'data_preview': '数据预览',
        'step3': '步骤3: 开始预测',
        'batch_predict': '🚀 批量预测',
        'predicting': '预测中...',
        'prediction_complete': '✅ 预测完成！',
        'prediction_results': '预测结果',
        'download_results': '💾 下载结果CSV',

        # 模型报告
        'report_header': '📊 模型透明度报告',
        'model_performance': '📈 模型性能',
        'auc': 'AUC',
        'f1_score': 'F1 Score',
        'best_threshold': '最佳阈值',
        'feature_importance': '📊 特征重要性 (Top 15)',
        'top15_importance': 'Top 15 Feature Importances',
        'fairness_analysis': '⚖️ 公平性分析',
        'gender_di': '性别 DI (差异影响比率)',
        'age_di': '年龄 DI (差异影响比率)',
        'acceptable': '✅ 可接受 (≥0.8)',
        'needs_improvement': '⚠️ 需改进',
        'di_hint': '💡 DI < 0.8 表示存在明显歧视，需要改进模型',

        # 历史
        'history_header': '📜 预测历史',
        'no_records': '暂无预测记录',
        'total_records': '总记录数',
        'single_predictions': '单客户预测',
        'batch_predictions': '批量预测',
        'history_records': '历史记录',
        'clear_history': '🗑️ 清空历史记录',
        'history_cleared': '历史记录已清空',

        # 关于
        'about_header': 'ℹ️ 关于本项目',
        'project_title': '🏦 信贷风险评估系统',
        'project_info': '### 项目信息',
        'project_name': '- **项目名称**: COMP5572 消费者信贷风险预测',
        'course': '- **课程**: Interdisciplinary Project Assignment',
        'data_source': '- **数据来源**: Home Credit Default Risk (Kaggle)',
        'model_info': '### 模型信息',
        'algorithm': '- **算法**: LightGBM Gradient Boosting',
        'samples': '- **训练样本**: 108,657',
        'features': '- **特征数量**: 33 (含4个衍生特征)',
        'validation': '- **验证方式**: 5-Fold Stratified Cross-Validation',
        'core_features': '### 核心功能',
        'feature1': '1. ✅ 单客户实时风险评估',
        'feature2': '2. ✅ 批量CSV预测',
        'feature3': '3. ✅ SHAP决策解释',
        'feature4': '4. ✅ 模型透明度报告',
        'feature5': '5. ✅ 公平性分析',
        'feature6': '6. ✅ 预测历史记录',
        'disclaimer': '⚠️ **免责声明**',
        'disclaimer_text': '''
本系统仅用于教育演示目的，不可用于实际信贷决策。
实际应用需要额外的验证、合规审查和监管批准。
''',

        # 设置
        'settings_header': '⚙️ 设置 / Settings',
        'language': '语言 / Language',
        'advanced_settings': '⚙️ 高级设置',
        'threshold_caption': '调整分类阈值（影响风险评估严格程度）',
        'threshold_hint': '值越大 → 越宽松（更易批准）',
        'threshold_label': '分类阈值',
        'threshold_help': '违约概率超过此值将被标记为高风险',
        'current_threshold': '当前阈值:',
        'model_info_header': 'ℹ️ 当前模型',

        # 错误
        'model_load_failed': '模型加载失败',
        'prediction_failed': '预测失败',
        'processing_failed': '处理失败',

        # 输入摘要格式
        'input_summary_format': '{gender}, {age}岁, 收入{income}k',
    },

    'en': {
        # Page config
        'page_title': 'Credit Risk Assessment',
        'page_icon': '🏦',

        # Sidebar
        'sidebar_title': '🏦 Credit Risk Assessment',
        'select_function': 'Select Function',

        # Page names
        'page_single': '📝 Single Prediction',
        'page_batch': '📤 Batch Prediction',
        'page_report': '📊 Model Report',
        'page_history': '📜 Prediction History',
        'page_about': 'ℹ️ About',

        # Single prediction
        'single_header': '📝 Single Customer Assessment',
        'enter_info': 'Enter Customer Information',
        'basic_info': '📋 Basic Information',
        'financial_info': '💰 Financial Information',
        'employment_info': '💼 Employment Information',
        'asset_info': '🏠 Asset Information',
        'external_scores': '📊 External Credit Scores',
        'start_assessment': '🔍 Start Assessment',
        'load_sample': '📋 Load Sample Data',

        # Basic info
        'gender': 'Gender',
        'age': 'Age',
        'age_hint': 'Auto-converted to birth days',
        'marital_status': 'Marital Status',
        'children': 'Number of Children',
        'family_members': 'Family Members',

        # Financial info
        'annual_income': 'Annual Income',
        'loan_amount': 'Loan Amount',
        'monthly_payment': 'Monthly Payment',
        'goods_price': 'Goods Price',

        # Employment info
        'income_type': 'Income Type',
        'occupation': 'Occupation Type',
        'employment_years': 'Employment Duration (Years)',
        'employment_hint': 'Auto-converted to employment days',

        # Asset info
        'own_car': 'Owns Car (FLAG_OWN_CAR)',
        'own_realty': 'Owns Real Estate (FLAG_OWN_REALTY)',
        'car_age': 'Car Age (OWN_CAR_AGE)',
        'housing_type': 'Housing Type (NAME_HOUSING_TYPE)',

        # External scores
        'ext_score_1': 'External Score 1 (EXT_SOURCE_1)',
        'ext_score_2': 'External Score 2 (EXT_SOURCE_2)',
        'ext_score_3': 'External Score 3 (EXT_SOURCE_3)',

        # Gender options
        'gender_male': 'Male',
        'gender_female': 'Female',

        # Marital status
        'status_married': 'Married',
        'status_single': 'Single',
        'status_civil': 'Civil Marriage',
        'status_widow': 'Widow',
        'status_separated': 'Separated',

        # Income type
        'income_working': 'Working',
        'income_state': 'State servant',
        'income_pensioner': 'Pensioner',
        'income_commercial': 'Commercial associate',

        # Occupation
        'occupation_laborers': 'Laborers',
        'occupation_core': 'Core staff',
        'occupation_sales': 'Sales staff',
        'occupation_managers': 'Managers',
        'occupation_drivers': 'Drivers',
        'occupation_other': 'Other',

        # Housing type
        'housing_house': 'House / apartment',
        'housing_rented': 'Rented apartment',
        'housing_parents': 'With parents',
        'housing_other': 'Other',

        # Results
        'results_header': '📊 Assessment Results',
        'default_prob': 'Default Probability',
        'decision_suggestion': 'Recommendation:',
        'risk_assessment': 'Risk Assessment',
        'description': '💡 {desc}',

        # Risk levels
        'risk_low': 'Low Risk',
        'risk_medium': 'Medium Risk',
        'risk_high': 'High Risk',
        'decision_approve': 'Approve',
        'decision_review': 'Review Required',
        'decision_reject': 'Reject',
        'desc_low': 'Low default risk, safe to approve loan',
        'desc_medium': 'Moderate default risk, consider additional collateral or reduce loan amount',
        'desc_high': 'High default risk, loan not recommended',

        # SHAP analysis
        'shap_header': '🔍 Decision Analysis',
        'positive_factors': '✅ Positive Factors (Reduce Risk)',
        'negative_factors': '⚠️ Risk Factors (Increase Risk)',

        # Batch prediction
        'batch_header': '📤 Batch Prediction (CSV Upload)',
        'step1': 'Step 1: Download CSV Template',
        'download_template': '📥 Download CSV Template',
        'step2': 'Step 2: Upload Data File',
        'select_csv': 'Select CSV file',
        'loaded_rows': '✅ Loaded {n} rows',
        'data_preview': 'Data Preview',
        'step3': 'Step 3: Start Prediction',
        'batch_predict': '🚀 Batch Predict',
        'predicting': 'Predicting...',
        'prediction_complete': '✅ Prediction complete!',
        'prediction_results': 'Prediction Results',
        'download_results': '💾 Download Results CSV',

        # Model report
        'report_header': '📊 Model Transparency Report',
        'model_performance': '📈 Model Performance',
        'auc': 'AUC',
        'f1_score': 'F1 Score',
        'best_threshold': 'Best Threshold',
        'feature_importance': '📊 Feature Importance (Top 15)',
        'top15_importance': 'Top 15 Feature Importances',
        'fairness_analysis': '⚖️ Fairness Analysis',
        'gender_di': 'Gender DI (Disparate Impact)',
        'age_di': 'Age DI (Disparate Impact)',
        'acceptable': '✅ Acceptable (≥0.8)',
        'needs_improvement': '⚠️ Needs Improvement',
        'di_hint': '💡 DI < 0.8 indicates significant bias, model needs improvement',

        # History
        'history_header': '📜 Prediction History',
        'no_records': 'No prediction records',
        'total_records': 'Total Records',
        'single_predictions': 'Single Predictions',
        'batch_predictions': 'Batch Predictions',
        'history_records': 'History Records',
        'clear_history': '🗑️ Clear History',
        'history_cleared': 'History cleared',

        # About
        'about_header': 'ℹ️ About This Project',
        'project_title': '🏦 Credit Risk Assessment System',
        'project_info': '### Project Information',
        'project_name': '- **Project Name**: COMP5572 Consumer Credit Risk Prediction',
        'course': '- **Course**: Interdisciplinary Project Assignment',
        'data_source': '- **Data Source**: Home Credit Default Risk (Kaggle)',
        'model_info': '### Model Information',
        'algorithm': '- **Algorithm**: LightGBM Gradient Boosting',
        'samples': '- **Training Samples**: 108,657',
        'features': '- **Features**: 33 (including 4 engineered features)',
        'validation': '- **Validation**: 5-Fold Stratified Cross-Validation',
        'core_features': '### Key Features',
        'feature1': '1. ✅ Single Customer Real-time Risk Assessment',
        'feature2': '2. ✅ Batch CSV Prediction',
        'feature3': '3. ✅ SHAP Decision Explanation',
        'feature4': '4. ✅ Model Transparency Report',
        'feature5': '5. ✅ Fairness Analysis',
        'feature6': '6. ✅ Prediction History',
        'disclaimer': '⚠️ **Disclaimer**',
        'disclaimer_text': '''
This system is for educational demonstration purposes only and should not be used
for actual credit decisions. Real-world applications require additional validation,
compliance review, and regulatory approval.
''',

        # Settings
        'settings_header': '⚙️ Settings / 设置',
        'language': 'Language / 语言',
        'advanced_settings': '⚙️ Advanced Settings',
        'threshold_caption': 'Adjust classification threshold (affects risk assessment strictness)',
        'threshold_hint': 'Higher value → More lenient (easier to approve)',
        'threshold_label': 'Classification Threshold',
        'threshold_help': 'Default probability above this value will be marked as high risk',
        'current_threshold': 'Current threshold:',
        'model_info_header': 'ℹ️ Current Model',

        # Errors
        'model_load_failed': 'Model loading failed',
        'prediction_failed': 'Prediction failed',
        'processing_failed': 'Processing failed',

        # Input summary format
        'input_summary_format': '{gender}, {age}yo, Income {income}k',
    }
}

# 特征名称翻译（保持不变）
FEATURE_NAME_MAP_ZH = {
    'EXT_SOURCE_3': '外部征信评分3', 'EXT_SOURCE_2': '外部征信评分2',
    'EXT_SOURCE_1': '外部征信评分1', 'DAYS_BIRTH': '年龄',
    'DAYS_EMPLOYED': '工作年限', 'AMT_ANNUITY': '月供金额',
    'AMT_CREDIT': '贷款金额', 'ANNUITY_INCOME_RATIO': '月供收入比',
    'CREDIT_INCOME_RATIO': '信贷收入比', 'ANNUITY_CREDIT_RATIO': '月供信贷比',
    'AGE_YEARS': '年龄', 'AMT_INCOME_TOTAL': '年收入',
    'CNT_FAM_MEMBERS': '家庭成员数', 'CNT_CHILDREN': '子女数量',
    'CODE_GENDER': '性别', 'NAME_EDUCATION_TYPE': '教育程度',
    'NAME_FAMILY_STATUS': '婚姻状况', 'NAME_INCOME_TYPE': '收入类型',
    'OCCUPATION_TYPE': '职业类型', 'FLAG_OWN_REALTY': '是否有房产',
    'FLAG_OWN_CAR': '是否有车', 'OWN_CAR_AGE': '车龄',
    'NAME_HOUSING_TYPE': '住房类型', 'REGION_POPULATION_RELATIVE': '地区人口相对值',
    'REGION_RATING_CLIENT': '地区评级', 'HOUR_APPR_PROCESS_START': '申请小时',
    'DAYS_ID_PUBLISH': '身份证发布时间', 'DAYS_LAST_PHONE_CHANGE': '最后换电话时间',
    'AMT_REQ_CREDIT_BUREAU_YEAR': '年度征信查询次数',
    'OBS_30_CNT_SOCIAL_CIRCLE': '社交圈观察人数',
    'DEF_30_CNT_SOCIAL_CIRCLE': '社交圈违约人数',
}

FEATURE_NAME_MAP_EN = {
    'EXT_SOURCE_3': 'External Credit Score 3', 'EXT_SOURCE_2': 'External Credit Score 2',
    'EXT_SOURCE_1': 'External Credit Score 1', 'DAYS_BIRTH': 'Age',
    'DAYS_EMPLOYED': 'Employment Duration', 'AMT_ANNUITY': 'Monthly Payment',
    'AMT_CREDIT': 'Loan Amount', 'ANNUITY_INCOME_RATIO': 'Payment-to-Income Ratio',
    'CREDIT_INCOME_RATIO': 'Credit-to-Income Ratio', 'ANNUITY_CREDIT_RATIO': 'Payment-to-Credit Ratio',
    'AGE_YEARS': 'Age (Years)', 'AMT_INCOME_TOTAL': 'Annual Income',
    'CNT_FAM_MEMBERS': 'Family Members', 'CNT_CHILDREN': 'Number of Children',
    'CODE_GENDER': 'Gender', 'NAME_EDUCATION_TYPE': 'Education Level',
    'NAME_FAMILY_STATUS': 'Marital Status', 'NAME_INCOME_TYPE': 'Income Type',
    'OCCUPATION_TYPE': 'Occupation Type', 'FLAG_OWN_REALTY': 'Owns Real Estate',
    'FLAG_OWN_CAR': 'Owns Car', 'OWN_CAR_AGE': 'Car Age',
    'NAME_HOUSING_TYPE': 'Housing Type', 'REGION_POPULATION_RELATIVE': 'Region Population',
    'REGION_RATING_CLIENT': 'Region Rating', 'HOUR_APPR_PROCESS_START': 'Application Hour',
    'DAYS_ID_PUBLISH': 'ID Publish Days', 'DAYS_LAST_PHONE_CHANGE': 'Last Phone Change Days',
    'AMT_REQ_CREDIT_BUREAU_YEAR': 'Credit Inquiries (Year)',
    'OBS_30_CNT_SOCIAL_CIRCLE': 'Social Circle Observed',
    'DEF_30_CNT_SOCIAL_CIRCLE': 'Social Circle Defaulted',
}

# ========== 3. 配置与常量 / Configuration & Constants ==========
COLORS = {
    'risk_low': '#22c55e',
    'risk_medium_low': '#84cc16',
    'risk_medium': '#f59e0b',
    'risk_high': '#ef4444',
    'primary': '#3b82f6',
}

SEED = 42
DEFAULT_THRESHOLD = 0.24
TARGET_COL = 'TARGET'

# 路径配置
BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR
HISTORY_DIR = BASE_DIR / 'predictions_history'
HISTORY_FILE = HISTORY_DIR / 'history.json'

# 创建历史目录
HISTORY_DIR.mkdir(exist_ok=True)

# ========== 4. 数据预处理函数 / Data Preprocessing ==========
def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """应用特征工程"""
    out = df.copy()
    if 'DAYS_EMPLOYED' in out.columns:
        out.loc[out['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = np.nan
    if 'DAYS_BIRTH' in out.columns:
        out['AGE_YEARS'] = (-out['DAYS_BIRTH'] / 365.25).clip(lower=18, upper=100)
    eps = 1e-6
    if {'AMT_ANNUITY', 'AMT_INCOME_TOTAL'}.issubset(out.columns):
        out['ANNUITY_INCOME_RATIO'] = out['AMT_ANNUITY'] / (out['AMT_INCOME_TOTAL'] + eps)
    if {'AMT_CREDIT', 'AMT_INCOME_TOTAL'}.issubset(out.columns):
        out['CREDIT_INCOME_RATIO'] = out['AMT_CREDIT'] / (out['AMT_INCOME_TOTAL'] + eps)
    if {'AMT_ANNUITY', 'AMT_CREDIT'}.issubset(out.columns):
        out['ANNUITY_CREDIT_RATIO'] = out['AMT_ANNUITY'] / (out['AMT_CREDIT'] + eps)
    return out

def build_features(df: pd.DataFrame, encoder=None, fit=False):
    """构建特征矩阵"""
    df = add_basic_features(df)
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols and c != TARGET_COL]

    if encoder is None:
        encoder_obj = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1)
        fit = True
        saved_cat_cols = None
    elif isinstance(encoder, tuple):
        encoder_obj, saved_cat_cols = encoder
    else:
        encoder_obj = encoder
        saved_cat_cols = None

    if len(cat_cols) > 0:
        if fit:
            df_cat_encoded = pd.DataFrame(encoder_obj.fit_transform(df[cat_cols]), columns=cat_cols, index=df.index)
            encoder_to_return = (encoder_obj, cat_cols)
        else:
            if saved_cat_cols is not None:
                cat_cols = saved_cat_cols
            for c in cat_cols:
                if c not in df.columns:
                    df[c] = -1
            df_cat_encoded = pd.DataFrame(encoder_obj.transform(df[cat_cols]), columns=cat_cols, index=df.index)
            encoder_to_return = (encoder_obj, saved_cat_cols)
    else:
        df_cat_encoded = pd.DataFrame(index=df.index)
        encoder_to_return = (encoder_obj, cat_cols) if fit else encoder

    df_num = df[num_cols].copy()
    df_final = pd.concat([df_num, df_cat_encoded], axis=1)
    return df_final, encoder_to_return

def preprocess_single_input(input_dict: dict, encoder) -> pd.DataFrame:
    """预处理单条输入数据"""
    df = pd.DataFrame([input_dict])
    df_processed, _ = build_features(df, encoder=encoder, fit=False)
    return df_processed

# ========== 5. 风险等级函数 / Risk Level Functions ==========
def get_risk_level(probability: float, threshold: float = None, lang: str = 'zh') -> dict:
    """根据违约概率获取风险等级"""
    if threshold is None:
        threshold = DEFAULT_THRESHOLD

    t = TRANSLATIONS[lang]

    if probability < threshold * 0.33:
        return {'level': t['risk_low'], 'emoji': '✅', 'color': COLORS['risk_low'],
                'decision': t['decision_approve'], 'description': t['desc_low']}
    elif probability < threshold:
        return {'level': t['risk_medium'], 'emoji': '⚠️', 'color': COLORS['risk_medium'],
                'decision': t['decision_review'], 'description': t['desc_medium']}
    else:
        return {'level': t['risk_high'], 'emoji': '❌', 'color': COLORS['risk_high'],
                'decision': t['decision_reject'], 'description': t['desc_high']}

# ========== 6. 历史记录函数 / History Functions ==========
def load_history() -> list:
    """加载历史记录"""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f).get('predictions', [])
    return []

def save_history(predictions: list):
    """保存历史记录"""
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump({'predictions': predictions}, f, ensure_ascii=False, indent=2)

def add_prediction_record(mode: str, input_summary: str, result: dict):
    """添加预测记录"""
    predictions = load_history()
    record = {'timestamp': datetime.now().isoformat(), 'mode': mode, 'input_summary': input_summary, 'result': result}
    predictions.append(record)
    save_history(predictions)

# ========== 7. 页面函数 / Page Functions ==========
def page_single_prediction(credit_model, lang: str):
    """单客户预测页面"""
    t = TRANSLATIONS[lang]
    st.header(t['single_header'])
    st.subheader(t['enter_info'])

    # 数据映射字典
    if lang == 'zh':
        gender_map = {"男": "M", "女": "F"}
        family_status_map = {"已婚": "Married", "单身": "Single / not married", "未婚同居": "Civil marriage", "丧偶": "Widow", "分居": "Separated"}
        income_type_map = {"在职": "Working", "公务员": "State servant", "退休": "Pensioner", "个体工商户": "Commercial associate"}
        occupation_map = {"劳工": "Laborers", "核心员工": "Core staff", "销售": "Sales staff", "经理": "Managers", "司机": "Drivers", "其他": "Laborers"}
        gender_options = [t['gender_male'], t['gender_female']]
        status_options = [t['status_married'], t['status_single'], t['status_civil'], t['status_widow'], t['status_separated']]
        income_options = [t['income_working'], t['income_state'], t['income_pensioner'], t['income_commercial']]
        occupation_options = [t['occupation_laborers'], t['occupation_core'], t['occupation_sales'], t['occupation_managers'], t['occupation_drivers'], t['occupation_other']]
        housing_options = [t['housing_house'], t['housing_rented'], t['housing_parents'], t['housing_other']]
        age_caption = f"将自动转换为出生天数: {-int(st.session_state.get('age_years', 27) * 365.25)}"
        work_caption = f"将自动转换为工作天数: {-int(st.session_state.get('work_years', 3) * 365)}"
    else:
        gender_map = {"Male": "M", "Female": "F"}
        family_status_map = {"Married": "Married", "Single": "Single / not married", "Civil Marriage": "Civil marriage", "Widow": "Widow", "Separated": "Separated"}
        income_type_map = {"Working": "Working", "State servant": "State servant", "Pensioner": "Pensioner", "Commercial associate": "Commercial associate"}
        occupation_map = {"Laborers": "Laborers", "Core staff": "Core staff", "Sales staff": "Sales staff", "Managers": "Managers", "Drivers": "Drivers", "Other": "Laborers"}
        gender_options = [t['gender_male'], t['gender_female']]
        status_options = [t['status_married'], t['status_single'], t['status_civil'], t['status_widow'], t['status_separated']]
        income_options = [t['income_working'], t['income_state'], t['income_pensioner'], t['income_commercial']]
        occupation_options = [t['occupation_laborers'], t['occupation_core'], t['occupation_sales'], t['occupation_managers'], t['occupation_drivers'], t['occupation_other']]
        housing_options = [t['housing_house'], t['housing_rented'], t['housing_parents'], t['housing_other']]
        age_caption = f"Auto-converted to birth days: {-int(st.session_state.get('age_years', 27) * 365.25)}"
        work_caption = f"Auto-converted to employment days: {-int(st.session_state.get('work_years', 3) * 365)}"

    # 输入表单
    with st.expander(t['basic_info'], expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox(t['gender'], gender_options, index=0)
            age_years = st.number_input(t['age'], value=27, min_value=18, max_value=100, key='age_years')
            st.caption(age_caption)
        with col2:
            family_status = st.selectbox(t['marital_status'], status_options, index=0)
            children = st.number_input(t['children'], value=0, min_value=0, max_value=20)
            family_members = st.number_input(t['family_members'], value=2, min_value=1)

    with st.expander(t['financial_info']):
        col1, col2 = st.columns(2)
        with col1:
            income = st.number_input(t['annual_income'], value=180000, min_value=0, step=10000)
            credit = st.number_input(t['loan_amount'], value=500000, min_value=0, step=10000)
        with col2:
            annuity = st.number_input(t['monthly_payment'], value=25000, min_value=0, step=1000)
            goods_price = st.number_input(t['goods_price'], value=450000, min_value=0, step=10000)

    with st.expander(t['employment_info']):
        col1, col2 = st.columns(2)
        with col1:
            income_type = st.selectbox(t['income_type'], income_options)
            occupation = st.selectbox(t['occupation'], occupation_options)
        with col2:
            work_years = st.number_input(t['employment_years'], value=3, min_value=0, max_value=50, key='work_years')
            st.caption(work_caption)

    with st.expander(t['asset_info']):
        col1, col2 = st.columns(2)
        with col1:
            own_car = st.selectbox(t['own_car'], ["Y", "N"], index=1)
            own_realty = st.selectbox(t['own_realty'], ["Y", "N"], index=0)
            if own_car == "Y":
                car_age = st.number_input(t['car_age'], value=5, min_value=0)
            else:
                car_age = 0
        with col2:
            housing = st.selectbox(t['housing_type'], housing_options)

    with st.expander(t['external_scores']):
        col1, col2, col3 = st.columns(3)
        with col1:
            ext1 = st.slider(t['ext_score_1'], 0.0, 1.0, 0.5)
        with col2:
            ext2 = st.slider(t['ext_score_2'], 0.0, 1.0, 0.5)
        with col3:
            ext3 = st.slider(t['ext_score_3'], 0.0, 1.0, 0.5)

    col1, col2 = st.columns([1, 2])
    with col1:
        predict_btn = st.button(t['start_assessment'], type="primary", use_container_width=True)
    with col2:
        load_sample_btn = st.button(t['load_sample'], use_container_width=True)

    if load_sample_btn:
        st.session_state['sample_loaded'] = True
        st.rerun()

    if predict_btn:
        input_data = {
            'SK_ID_CURR': 999999,
            'CODE_GENDER': gender_map.get(gender, "M"),
            'DAYS_BIRTH': int(-age_years * 365.25),
            'NAME_EDUCATION_TYPE': 'Secondary / secondary special',
            'NAME_FAMILY_STATUS': family_status_map.get(family_status, "Married"),
            'CNT_CHILDREN': children,
            'CNT_FAM_MEMBERS': family_members,
            'AMT_INCOME_TOTAL': income,
            'AMT_CREDIT': credit,
            'AMT_ANNUITY': annuity,
            'AMT_GOODS_PRICE': goods_price,
            'NAME_INCOME_TYPE': income_type_map.get(income_type, "Working"),
            'OCCUPATION_TYPE': occupation_map.get(occupation, "Laborers"),
            'DAYS_EMPLOYED': int(-work_years * 365),
            'FLAG_OWN_CAR': own_car,
            'FLAG_OWN_REALTY': own_realty,
            'OWN_CAR_AGE': car_age if own_car == "Y" else 0,
            'NAME_HOUSING_TYPE': housing,
            'REGION_POPULATION_RELATIVE': 0.02,
            'REGION_RATING_CLIENT': 2,
            'EXT_SOURCE_1': ext1,
            'EXT_SOURCE_2': ext2,
            'EXT_SOURCE_3': ext3,
            'OBS_30_CNT_SOCIAL_CIRCLE': 2,
            'DEF_30_CNT_SOCIAL_CIRCLE': 0,
            'HOUR_APPR_PROCESS_START': 10,
            'DAYS_ID_PUBLISH': -4000,
            'DAYS_LAST_PHONE_CHANGE': -1000,
            'AMT_REQ_CREDIT_BUREAU_YEAR': 1,
        }

        try:
            X = preprocess_single_input(input_data, credit_model.encoder)
            prob = credit_model.predict_proba(X)[0]
            risk_info = get_risk_level(prob, credit_model.threshold, lang)

            # SHAP
            try:
                explainer = shap.TreeExplainer(credit_model.model)
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

                feature_map = FEATURE_NAME_MAP_ZH if lang == 'zh' else FEATURE_NAME_MAP_EN
                feature_names = X.columns.tolist()
                feature_shap = list(zip(feature_names, shap_values[0]))
                feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)

                positive_factors = [(feature_map.get(f, f), v) for f, v in feature_shap if v < 0][:5]
                negative_factors = [(feature_map.get(f, f), v) for f, v in feature_shap if v > 0][:5]
            except:
                positive_factors = []
                negative_factors = []

            # Save history
            input_summary = t['input_summary_format'].format(gender=gender, age=age_years, income=income/1000)
            add_prediction_record('single', input_summary, {
                'probability': float(prob),
                'risk_level': risk_info['level'],
                'decision': risk_info['decision']
            })

            # Results
            st.divider()
            st.subheader(t['results_header'])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(t['default_prob'], f"{prob*100:.2f}%")
            with col2:
                st.markdown(f"### {risk_info['emoji']} {risk_info['level']}")
            with col3:
                st.markdown(f"**{t['decision_suggestion']}** {risk_info['decision']}")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': f"{t['risk_assessment']} ({risk_info['level']})"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': risk_info['color']},
                    'steps': [
                        {'range': [0, credit_model.threshold * 70], 'color': '#f0fdf4'},
                        {'range': [credit_model.threshold * 70, credit_model.threshold * 100], 'color': '#fffbeb'},
                        {'range': [credit_model.threshold * 100, 100], 'color': '#fef2f2'},
                    ],
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

            st.info(t['description'].format(desc=risk_info['description']))

            if positive_factors or negative_factors:
                st.subheader(t['shap_header'])
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{t['positive_factors']}**")
                    for feat, val in positive_factors:
                        st.write(f"• {feat}: {val:.4f}")
                with col2:
                    st.markdown(f"**{t['negative_factors']}**")
                    for feat, val in negative_factors:
                        st.write(f"• {feat}: {val:.4f}")

        except Exception as e:
            st.error(f"{t['prediction_failed']}: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def page_batch_prediction(credit_model, lang: str):
    """批量预测页面"""
    t = TRANSLATIONS[lang]
    st.header(t['batch_header'])

    st.subheader(t['step1'])

    template_data = {
        'SK_ID_CURR': [100001, 100002],
        'CODE_GENDER': ['M', 'F'],
        'DAYS_BIRTH': [-10000, -12000],
        'NAME_EDUCATION_TYPE': ['Secondary / secondary special', 'Higher education'],
        'NAME_FAMILY_STATUS': ['Married', 'Single / not married'],
        'CNT_CHILDREN': [0, 1],
        'CNT_FAM_MEMBERS': [2, 3],
        'AMT_INCOME_TOTAL': [180000, 150000],
        'AMT_CREDIT': [500000, 400000],
        'AMT_ANNUITY': [25000, 20000],
        'AMT_GOODS_PRICE': [450000, 350000],
        'NAME_INCOME_TYPE': ['Working', 'State servant'],
        'OCCUPATION_TYPE': ['Laborers', 'Core staff'],
        'DAYS_EMPLOYED': [-1000, -2000],
        'FLAG_OWN_CAR': ['N', 'Y'],
        'FLAG_OWN_REALTY': ['Y', 'Y'],
        'OWN_CAR_AGE': [0, 5],
        'NAME_HOUSING_TYPE': ['House / apartment', 'House / apartment'],
        'REGION_POPULATION_RELATIVE': [0.02, 0.03],
        'REGION_RATING_CLIENT': [2, 2],
        'EXT_SOURCE_1': [0.5, 0.6],
        'EXT_SOURCE_2': [0.5, 0.7],
        'EXT_SOURCE_3': [0.5, 0.4],
        'OBS_30_CNT_SOCIAL_CIRCLE': [2, 1],
        'DEF_30_CNT_SOCIAL_CIRCLE': [0, 0],
        'HOUR_APPR_PROCESS_START': [10, 14],
        'DAYS_ID_PUBLISH': [-4000, -3000],
        'DAYS_LAST_PHONE_CHANGE': [-1000, -500],
        'AMT_REQ_CREDIT_BUREAU_YEAR': [1, 2],
    }
    template_df = pd.DataFrame(template_data)
    csv = template_df.to_csv(index=False).encode('utf-8-sig')

    st.download_button(t['download_template'], csv, "credit_risk_template.csv", "text/csv", type="primary")

    st.divider()
    st.subheader(t['step2'])
    uploaded_file = st.file_uploader(t['select_csv'], type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(t['loaded_rows'].format(n=len(df)))

            with st.expander(t['data_preview']):
                st.dataframe(df.head(), use_container_width=True)

            st.divider()
            st.subheader(t['step3'])

            if st.button(t['batch_predict'], type="primary"):
                with st.spinner(t['predicting']):
                    X, _ = build_features(df, encoder=credit_model.encoder, fit=False)
                    probabilities = credit_model.predict_proba(X)

                    df['TARGET_PROB'] = probabilities
                    df['RISK_LEVEL'] = df['TARGET_PROB'].apply(lambda p: get_risk_level(p, credit_model.threshold, lang)['level'])
                    df['DECISION'] = df['TARGET_PROB'].apply(lambda p: get_risk_level(p, credit_model.threshold, lang)['decision'])

                    low_count = (df['RISK_LEVEL'] == t['risk_low']).sum()
                    med = (df['RISK_LEVEL'] == t['risk_medium']).sum()
                    high = (df['RISK_LEVEL'] == t['risk_high']).sum()

                    add_prediction_record('batch', f"Batch prediction {len(df)} records", {
                        'total': len(df),
                        'low_risk': int(low_count),
                        'high_risk': int(med + high)
                    })

                    st.success(t['prediction_complete'])

                    col1, col2, col3 = st.columns(3)
                    col1.metric(f"✅ {t['risk_low']}", low_count)
                    col2.metric(f"⚠️ {t['risk_medium']}", med)
                    col3.metric(f"❌ {t['risk_high']}", high)

                    st.subheader(t['prediction_results'])
                    result_cols = [col for col in ['SK_ID_CURR', 'TARGET_PROB', 'RISK_LEVEL', 'DECISION'] if col in df.columns]
                    st.dataframe(df[result_cols], use_container_width=True)

                    result_csv = df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(t['download_results'], result_csv,
                                     f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                     "text/csv", type="primary")

        except Exception as e:
            st.error(f"{t['processing_failed']}: {str(e)}")

def page_model_report(credit_model, lang: str):
    """模型报告页面"""
    t = TRANSLATIONS[lang]
    st.header(t['report_header'])

    st.subheader(t['model_performance'])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(t['auc'], f"{credit_model.auc:.4f}")
    with col2:
        st.metric(t['f1_score'], f"{credit_model.f1:.4f}")
    with col3:
        st.metric(t['best_threshold'], f"{credit_model.threshold:.4f}")

    st.divider()
    st.subheader(t['feature_importance'])

    importances = credit_model.model.feature_importances_
    feature_names = credit_model.model.feature_names_in_
    feature_map = FEATURE_NAME_MAP_ZH if lang == 'zh' else FEATURE_NAME_MAP_EN

    feat_imp = pd.DataFrame({
        'feature': [feature_map.get(f, f) for f in feature_names],
        'importance': importances
    }).sort_values('importance', ascending=False)

    fig = px.bar(feat_imp.head(15), x='importance', y='feature', orientation='h', title=t['top15_importance'])
    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader(t['fairness_analysis'])

    col1, col2 = st.columns(2)
    with col1:
        st.metric(t['gender_di'], "0.8139")
        st.success(t['acceptable']) if 0.8139 >= 0.8 else st.warning(t['needs_improvement'])
    with col2:
        st.metric(t['age_di'], "0.7743")
        st.success(t['acceptable']) if 0.7743 >= 0.8 else st.warning(t['needs_improvement'])

    st.info(t['di_hint'])

def page_history(lang: str):
    """预测历史页面"""
    t = TRANSLATIONS[lang]
    st.header(t['history_header'])

    predictions = load_history()

    if not predictions:
        st.info(t['no_records'])
        return

    total = len(predictions)
    single_count = sum(1 for p in predictions if p['mode'] == 'single')
    batch_count = sum(1 for p in predictions if p['mode'] == 'batch')

    col1, col2, col3 = st.columns(3)
    col1.metric(t['total_records'], total)
    col2.metric(t['single_predictions'], single_count)
    col3.metric(t['batch_predictions'], batch_count)

    st.divider()
    st.subheader(t['history_records'])

    for i, record in enumerate(reversed(predictions[-20:])):
        with st.expander(f"⏰ {record['timestamp'][:19]} - {record['mode']}"):
            st.write(f"**Input Summary:** {record['input_summary']}")
            st.write(f"**Result:** {record['result']}")

    st.divider()
    if st.button(t['clear_history'], type="secondary"):
        save_history([])
        st.success(t['history_cleared'])
        st.rerun()

def page_about(lang: str):
    """关于页面"""
    t = TRANSLATIONS[lang]
    st.header(t['about_header'])

    st.markdown(f"""
    {t['project_title']}

    {t['project_info']}
    {t['project_name']}
    {t['course']}
    {t['data_source']}

    {t['model_info']}
    {t['algorithm']}
    {t['samples']}
    {t['features']}
    {t['validation']}

    {t['core_features']}
    {t['feature1']}
    {t['feature2']}
    {t['feature3']}
    {t['feature4']}
    {t['feature5']}
    {t['feature6']}
    """)

    st.divider()
    st.warning(f"""
    {t['disclaimer']}

    {t['disclaimer_text']}
    """)

# ========== 8. 模型加载 / Model Loading ==========
@st.cache_resource
def load_model_and_encoder():
    """加载模型和编码器（缓存）"""
    model_path = ARTIFACTS_DIR / 'baseline_lgbm.pkl'
    metrics_path = ARTIFACTS_DIR / 'baseline_metrics.csv'
    encoder_path = ARTIFACTS_DIR / 'encoder.pkl'

    class CreditRiskModel:
        def __init__(self):
            from lightgbm import LGBMClassifier
            self.model = joblib.load(model_path)
            self.metrics = pd.read_csv(metrics_path)
            self.threshold = self.metrics['best_threshold'].values[0]
            self.auc = self.metrics['cv_auc_oof'].values[0]
            self.f1 = self.metrics['best_f1'].values[0]

        def predict_proba(self, X):
            return self.model.predict_proba(X)[:, 1]

    credit_model = CreditRiskModel()
    encoder_obj = joblib.load(encoder_path)
    cat_cols = ['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                'NAME_INCOME_TYPE', 'OCCUPATION_TYPE', 'FLAG_OWN_REALTY',
                'FLAG_OWN_CAR', 'NAME_HOUSING_TYPE']
    credit_model.encoder = (encoder_obj, cat_cols)
    credit_model.cat_cols = cat_cols

    return credit_model

# ========== 9. 主函数 / Main Function ==========
def main():
    # 初始化语言
    if 'language' not in st.session_state:
        st.session_state['language'] = 'zh'

    # 加载模型
    try:
        credit_model = load_model_and_encoder()
    except Exception as e:
        st.error(f"{TRANSLATIONS[st.session_state['language']]['model_load_failed']}: {str(e)}")
        return

    # 侧边栏
    lang = st.session_state['language']
    t = TRANSLATIONS[lang]

    st.sidebar.title(t['sidebar_title'])
    st.sidebar.markdown("---")

    # 页面选择
    pages = {
        t['page_single']: "single",
        t['page_batch']: "batch",
        t['page_report']: "report",
        t['page_history']: "history",
        t['page_about']: "about"
    }

    selected_page = st.sidebar.radio(t['select_function'], list(pages.keys()))
    st.sidebar.markdown("---")

    # 阈值调整
    with st.sidebar.expander(t['advanced_settings']):
        # 语言选择
        lang_options = {'🇨🇳 中文': 'zh', '🇺🇸 English': 'en'}
        selected_lang = st.selectbox(t['language'], list(lang_options.keys()),
                                     index=list(lang_options.values()).index(st.session_state['language']))
        if lang_options[selected_lang] != st.session_state['language']:
            st.session_state['language'] = lang_options[selected_lang]
            st.rerun()

        st.markdown("---")

        # 阈值调整
        st.caption(t['threshold_caption'])
        st.caption(t['threshold_hint'])

        custom_threshold = st.slider(
            t['threshold_label'],
            min_value=0.05,
            max_value=0.50,
            value=0.24,
            step=0.01,
            format="%.4f",
            help=t['threshold_help']
        )

        st.caption(f"{t['current_threshold']} {custom_threshold:.4f}")
        credit_model.threshold = custom_threshold

    st.sidebar.markdown("---")

    # 模型信息
    st.sidebar.subheader(t['model_info_header'])
    st.sidebar.write(f"**AUC**: {credit_model.auc:.4f}")
    st.sidebar.write(f"**F1**: {credit_model.f1:.4f}")
    st.sidebar.write(f"**Threshold**: {credit_model.threshold:.4f}")

    # 路由到对应页面
    page = pages[selected_page]

    if page == "single":
        page_single_prediction(credit_model, lang)
    elif page == "batch":
        page_batch_prediction(credit_model, lang)
    elif page == "report":
        page_model_report(credit_model, lang)
    elif page == "history":
        page_history(lang)
    elif page == "about":
        page_about(lang)

# Global variables
if 'sample_loaded' not in st.session_state:
    st.session_state['sample_loaded'] = False

if __name__ == "__main__":
    main()
