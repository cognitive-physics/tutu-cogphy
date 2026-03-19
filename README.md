# Cognitive Translator / Cognitive Engine

一个从 Notion 整理出来的研究型原型仓库，围绕三层闭环架构展开：

1. **编码器（Encoder）**：把模糊的人类输入转成结构化参数估计  
2. **四维责任模型（Decision Layer）**：把事件拆成正交责任分量，并导出最小耗散行动路径  
3. **解码器（Decoder）**：把最优路径翻译成接收者可理解的输出，并根据反馈自适应压缩率

> 当前状态：**研究原型 / 概念验证（research prototype）**  
> 这里的公式、代码和接口，主要用于表达理论结构与工程映射，不应视为已经过严格实验验证的生产系统。

## 仓库结构

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
├── requirements.txt
├── Dockerfile
└── .gitignore
```

## 文档目录

- [`docs/encoder.md`](docs/encoder.md) — 编码器：贝叶斯-IFD 推断引擎
- [`docs/four-dimensional-responsibility.md`](docs/four-dimensional-responsibility.md) — 四维责任模型
- [`docs/decoder.md`](docs/decoder.md) — 解码器：高压缩输出引擎
- [`docs/formalism.md`](docs/formalism.md) — 三层整体公式化版本

## 运行后端 API

```bash
pip install -r requirements.txt
uvicorn cognitive_engine.api:app --reload
```

默认会在 `http://localhost:8000` 提供服务。

### 主要接口

- `POST /chat`
- `POST /feedback`
- `GET /profile/{person_id}`
- `DELETE /profile/{person_id}`
- `GET /health`

## 运行前端原型

另开一个终端：

```bash
streamlit run frontend/app.py
```

默认打开 `http://localhost:8501`。

## 设计要点

### 1) 编码器
- 多通道采样：语言内容、语言结构、情绪色彩、行为信号、隐性信号、一致性、上下文
- 逆问题求解：从观测输出反推内部参数概率分布
- 贝叶斯逐帧更新：数据越多，不确定性越低
- 跨通道 `D_KL` 降噪：识别并降权不一致信号
- 收敛判定：后验变化足够小时锁定结论

### 2) 四维责任模型
- 行为责任：当前帧，只记事实
- 教育责任：回溯初始条件与耦合历史
- 环境责任：分离外界造成的偏移
- 成长责任：在前三维已知时，求最小耗散行动路径

### 3) 解码器
- 结论先行
- 高压缩输出
- 根据 `eta`（解码率）自适应压缩率 `c`
- 闭环：输出反馈重新进入编码器，对接收带宽做下一轮估计

## 关于这次整理

本仓库内容由以下 Notion 页面整理而成，并做了轻度结构化、文件化与可运行化处理：

- 编码器：贝叶斯-IFD 推断引擎
- 四维责任模型
- 解码器：高压缩输出引擎
- 公式版（编码器 × 四维责任 × 解码器）
- 代码版（Python）
- API 服务
- 前端 app.py

## 下一步建议

- 为 `eta` 增加更稳定的标定数据集
- 把 `complexity.py` 从启发式规则升级到 embedding / encoder 模型
- 给 `DecisionEngine` 增加更清晰的目标函数实验
- 增加测试集与示例 notebook
