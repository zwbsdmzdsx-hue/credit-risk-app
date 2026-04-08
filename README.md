# 信贷风险评估系统

## 启动应用
在线访问：https://credit-risk-app-fhwuegfrnlfqduffuzt2am.streamlit.app/

本地运行：
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行应用
streamlit run app.py
```

应用将在浏览器中打开: http://localhost:8501

## 文件说明

| 文件 | 说明 |
|------|------|
| `app.py` | 主应用程序 |
| `requirements.txt` | Python 依赖包 |
| `baseline_lgbm.pkl` | 训练好的 LightGBM 模型 |
| `baseline_metrics.csv` | 模型性能指标 |
| `encoder.pkl` | 分类特征编码器 |
| `predictions_history/` | 预测历史记录目录（自动创建）|

## 功能

1. **单客户风险评估** - 输入客户信息，获取违约概率和风险等级
2. **批量预测** - 上传 CSV 文件进行批量评估
3. **模型报告** - 查看模型性能和特征重要性
4. **预测历史** - 查看历史预测记录

---

*COMP5572 课程作业 - 信贷风险评估 Demo*
