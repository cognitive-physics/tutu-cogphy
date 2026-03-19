# Cognitive Engine / 认知引擎

> 一个把“编码器 → 四维责任模型 → 解码器”三层闭环架构，整理为可阅读文档、可运行原型与可扩展 API 的研究型仓库。

## 项目简介

这个仓库整理了你在 Notion 中的几份核心文档，并将它们映射为一个更清晰的 GitHub 项目结构：

- **理论文档层**：编码器、四维责任模型、解码器、整体公式版
- **核心原型层**：`cognitive_engine/engine.py`
- **接口层**：FastAPI 服务 `cognitive_engine/api.py`
- **交互演示层**：Streamlit 前端 `frontend/app.py`

它的定位不是“已经定型的成熟产品”，而是一个：

- 研究原型（research prototype）
- 理论到工程的映射骨架
- 后续继续扩写、重构、实验和产品化的起点

## 核心结构

### 1. 编码器（Encoder）

输入侧负责把模糊的人类信号翻译为结构化参数估计。

主要特征：

- 多通道采样
- 逆问题求解
- 贝叶斯逐帧更新
- `D_KL` 跨通道降噪
- 收敛判定

### 2. 四维责任模型（Decision Layer）

处理中间层，把事件分为四个正交维度：

- 行为责任
- 教育责任
- 环境责任
- 成长责任

其中第四维负责从分析导出行动：

```math
δA = 0
```

### 3. 解码器（Decoder）

输出侧负责把已经求出的最优路径翻译成接收者能理解的表达方式。

主要特征：

- 结论先行
- 高压缩输出
- 带宽适配
- `η`（解码率）反馈闭环
- 输出压缩率 `c` 的动态调整

## 仓库目录

```text
.
├── docs/
│   ├── encoder.md
│   ├── four-dimensional-responsibility.md
│   ├── decoder.md
│   └── formalism.md
├── cognitive_engine/
│   ├── __init__.py
│   ├── engine.py
│   ├── complexity.py
│   ├── store.py
│   └── api.py
├── frontend/
│   └── app.py
├── README.md
├── README.zh-CN.md
├── LICENSE
├── requirements.txt
├── Dockerfile
└── .gitignore
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动后端 API

```bash
uvicorn cognitive_engine.api:app --reload
```

默认地址：`http://localhost:8000`

### 启动前端演示

```bash
streamlit run frontend/app.py
```

默认地址：`http://localhost:8501`

## 主要接口

- `POST /chat`
- `POST /feedback`
- `GET /profile/{person_id}`
- `DELETE /profile/{person_id}`
- `GET /health`

## 当前状态

当前仓库更适合作为：

- 理论梳理与留档
- 原型验证
- API/前端交互测试
- 后续产品化与论文配套材料的基础

而不是立即视为：

- 已经过系统实验验证的科学软件
- 已适合生产环境部署的正式系统

## 后续建议

- 为 `eta` 增加更稳定的标定方案
- 为 `DecisionEngine` 增加更清晰的目标函数实验
- 将 `complexity.py` 升级为向量模型或更强编码器
- 增加测试、示例 notebook 与更完整的文档说明

## 许可协议

本仓库当前使用 [MIT License](LICENSE)。
