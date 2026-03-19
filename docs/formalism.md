# 认知引擎 · 公式版（编码器 × 四维责任 × 解码器）

## 符号表

| 符号 | 含义 |
|---|---|
| `ρ(t)` | 信息密度场 |
| `ρ_0, H_0` | 初始条件 |
| `σ*` | 稳态吸引子 |
| `μ` | 内外不一致度 |
| `λ` | 节点间耦合强度 |
| `D_KL` | 信息不匹配度 |
| `R` | 认知留白 |
| `η` | 解码率 |
| `c` | 压缩率 |
| `g_{ij}` | 认知度量张量 |
| `δA = 0` | 最小作用量原理 |

## 一、编码器

原始输入 `x(t)` 经过多通道采样与逆问题推断，得到结构化参数估计：

```math
\\text{Encoder}(\\mathbf{x}(t)) = \\hat{θ}
= (\\hat{ρ}_0, \\hat{H}_0, \\hat{μ}, \\hat{σ}^*)
```

### 观测合成

```math
ρ_{obs}(t) = \\sum_{k=1}^{K} w_k f_k(x_k(t)),
\\quad \\sum_k w_k = 1
```

### 贝叶斯更新

```math
P(θ \\,|\\, D_{1:n}) =
\\frac{P(D_n \\,|\\, θ) P(θ \\,|\\, D_{1:n-1})}{P(D_n \\,|\\, D_{1:n-1})}
```

### 跨通道降噪

```math
D_{KL}(ρ_{ch_A} \\,\\|\\, ρ_{ch_B}) > θ_{noise}
\\Rightarrow \\text{标记 } μ > 0
```

## 二、四维责任模型

处理层把当前事件分解为四个维度，并输出最优行动：

```math
\\text{Decision}(A_1, A_2, A_3) = a^*
```

其中：

```math
A_1 = ρ(t_n)
```

```math
A_2 = f(ρ_0, H_0, λ_{history})
```

```math
A_3 = Δρ_{ext}
```

最终：

```math
a^* = \\arg\\min_a \\; \\Phi[ρ_a(t)]
\\Longleftrightarrow
δA = 0
```

## 三、解码器

```math
\\text{Decoder}(a^*, c_n) \\to y(t)
```

### 解码率

```math
η =
\\frac{\\text{成功传递的信息量}}
{\\text{总输出信息量}}
\\in [0, 1]
```

### 压缩率自适应更新

```math
c_{n+1} = c_n + α(η_n - η_{target})
```

### 闭环方程

```math
η_{n+1} = \\text{Encoder}(\\text{Decoder}(a^*, c_n))
```

收敛条件：

```math
|η_{n+1} - η_n| < ε
\\Rightarrow c^* = c_n
```

## 四、完整系统

```math
\\mathbf{x}(t)
\\xrightarrow{\\text{编码器}}
\\hat{θ}
\\xrightarrow{\\text{四维责任}}
a^*
\\xrightarrow{\\text{解码器}}
y(t)
```

反馈再进入下一轮：

```math
y(t)
\\xrightarrow{\\text{反馈}}
\\hat{η}
\\xrightarrow{\\text{调整}}
c
\\xrightarrow{\\text{更新}}
\\text{下一轮解码器输入}
```

## 全局目标函数

```math
\\min_{c, a}
\\left[
D_{KL}(ρ_{self} \\,\\|\\, σ^*)
+
D_{KL}(ρ_{other} \\,\\|\\, ρ_{output})
\\right]
\\quad \\text{s.t. } δA = 0
```
