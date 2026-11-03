# 基于Superstore的零售利润归因与用户价值可视化分析

> 《商业数据分析》期末大作业 | 数据科学学院 | 大数据242
> 
> **作者**：周璐瑶 | **学号**：2024454995304

## 1. 项目背景与简介
随着零售行业的竞争加剧，企业普遍面临“增收不增利”的经营痛点。本项目以 Tableau 官方公开的 Superstore真实零售数据集为基础，运用 Python 全栈技术，完成从数据获取、清洗治理、模型分析到可视化大屏呈现的端到端商业数据分析流程。

本项目旨在挖掘利润流失的根本原因，精准刻画用户价值分层，并借助直观的数据可视化手段赋能管理层决策。

## 2. 数据合规与工具声明
- **数据源名称**：Sample - Superstore（Tableau 官方示例零售数据）。
- **获取方式**：通过官方Tableau 官网直接下载 XLSX 文件。**全程未使用任何网页爬虫技术，完全符合白名单规范。**
- **数据规模**：主表记录数 9,994 条，涵盖 2015-2018 年，包含 21 个核心字段。
- **工具声明**：本项目所有数据处理、机器学习模型、静态图表及交互式大屏，均由 **Python 代码原生实现**。未使用 Tableau、Power BI 等商业 BI 软件出图。

## 3. 核心技术栈
- **数据处理与清洗**：Pandas, NumPy, SQLite3
- **数据建模与算法**：Scikit-learn (K-Means 聚类), Statsmodels (SARIMA/Holt-Winters 预测), Scipy
- **可视化绘图**：Matplotlib , Plotly
- **交互式大屏**：Dash (纯 Python 前端联动框架)

## 4. 项目目录结构
```text
Superstore-Business-Analysis/
├── data/                 
│   ├── raw/              # 原始数据层（存放 orders_raw.csv 与原始 Excel）
│   └── clean/            # 清洗后数据层（存放 1张事实表+4张维度表，星型模型结构）
├── scripts/              
│   ├── 01_load.ipynb     # 数据加载与原始落盘，字段校验
│   ├── 02_clean.ipynb    # 数据清洗、特征派生、星型模型主外键对齐校验
│   ├── 03_analysis.ipynb # 核心业务模型计算（帕累托、RFM、聚类、预测等）
│   ├── 04_visuals.ipynb  # 15种高级暗黑模板复现 + 10张业务模型图绘制
│   └── 05_dashboard.ipynb # 基于 Dash 的交互式商业大屏（含9个联动组件）
├── outputs/              
│   ├── figures/          # 存储所有产出的高清图表（fig01 至 fig25 PNG图片）
│   └── tables/           # 存储模型计算的底层结果表（t30 至 t45 CSV数据表）
├── docs/                 
│   └── 基于Superstore的零售利润与用户分析.pdf  # 最终版完整论文
├── requirements.txt      # 项目依赖库清单
├── run_all.py            # 一键复现脚本入口（自动按顺序执行Jupyter代码）
└── README.md             # 项目说明书（本文件）
