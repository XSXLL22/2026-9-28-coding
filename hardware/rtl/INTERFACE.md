# 最小卷积算子 RTL 接口说明(P5.0a,实现前冻结)

版本:p5.0b,2026-09-28。对应合同 v1.3。本文件先于任何 RTL 编写;RTL 与测试台不得违背此处语义。**算子输出默认取 SiLU 前(SiLU 前的重量化饱和 int8 值);SiLU 属 PS_INT,不在 PL 边界内。**

修订记录:

- **p5.0b(2026-09-28,实现卷积核心时发现)**:权重流原写 "cout·9 个 int8",**漏乘 cin**。3×3 卷积每个输出通道需要 cin·9 个权重,权重张量展平后为 cout·cin·9。此处修正 §2/§3 的权重数量与地址公式。修正发生在**任何向量生成或 RTL 验收之前**(当时只有接口文档与 `mac_array.v`/`requant_sat.v` 两个模块,卷积核心尚未存在),因此不涉及任何已通过产物;但仍按合同修订纪律升版本并记录。
- p5.0a(2026-09-28):首版冻结。

## 1 模块与职责

| 模块 | 职责 |
|---|---|
| `mac_array.v` | 单个有符号 8×8 乘加:64 位累加器,clear/enable;数值上 |acc| 恒 ≤ 2^31−1(合同 §6 逐层校核保证) |
| `requant_sat.v` | 组合逻辑:t = acc·M + half;h = 2^(n−1)(n=0 时 h=0、y=t);算术右移 n;饱和到 [-128,127] |
| `conv3x3_core.v` | 任务控制:配置锁存、输入/权重流接收、逐输出像素 MAC 调度、零填充、输出流、错误状态 |

## 2 接口

```
clk, rst(同步,高有效)
── 配置口(仅 IDLE 态可写;BUSY 写入 → error_code=CONFIG_LOCKED)
config_we, config_addr[7:0], config_data[31:0]
  addr 0: cin   addr 1: cout   addr 2: h   addr 3: w
start(脉冲) ── busy/done/error/error_code[7:0]
── 输入像素流(CHW,零填充在核心内部处理,流中只含真实像素)
in_valid/in_ready/in_data[7:0](有符号 int8,共 cin·h·w 个)
── 权重流(OIHW 展平:cout·cin·9 个 int8,地址 o·cin·9 + ci·9 + i·3 + j)
wt_valid/wt_ready/wt_data[7:0]
param_valid/param_ready/param_data[31:0](每通道 3 个字:qb, M, shift)
── 输出流(CHW:cout·h_out·w_out,post-requant pre-SiLU int8)
out_valid/out_ready/out_data[7:0]
cycle_count[31:0](busy 期间计数)
```

`in_ready` 由核心状态决定;`out_ready` 由下游决定。**数据只在 valid && ready 同时为真时传输**;valid 拉高后 data 保持稳定直到握手完成。

## 3 任务语义

1. IDLE 态写配置 → start:校验 `1 ≤ cin ≤ MAX_CIN`、`1 ≤ cout ≤ MAX_COUT`、`1 ≤ h,w ≤ MAX_HW`、`h·w·cin ≤ 输入 RAM 容量`;不支持 → `error=1, error_code=UNSUPPORTED_CONFIG`,回到 IDLE,无任何输出。
2. LOAD_IN:接收 cin·h·w 像素写入 RAM(地址 ci·h·w + r·w + c)。
3. LOAD_W:接收 cout·cin·9 个权重(OIHW 展平)与每通道 qb/M/shift。
4. COMPUTE:对每个输出像素 (o, r, c)(h_out = h、w_out = w,pad=1,stride=1):清累加器,对 i,j∈[0,2]、ci∈[0,cin):窗口坐标 (r+i−1, c+j−1) 越界则跳过(等价零填充),否则 MAC 收 `in[ci,·,·] × w[o,i,j]`;累加完成后经 `requant_sat`(M=m_ram[o], shift=shift_ram[o])得到输出字节,握手送出。
5. 全部 cout·h·w_out 输出握手完成 → done 脉冲,回 IDLE。成功输出的数量严格等于 cout·h_out·w_out。
6. 任务中复位:立即清状态;复位后无旧结果泄漏,可重新提交任务。
7. 错误码:1 = UNSUPPORTED_CONFIG,2 = CONFIG_LOCKED,3 = PARAM_UNDERFLOW(参数流不足)。

## 4 数值规则(与合同 v1.3 一致)

- 累加:signed int8 × int8 → int64 逐项累加;|acc| ≤ 2^31−1 由包构建的逐层界证明。
- 重量化:t = acc·M + h,n ∈ [0,62];|acc·M + h| ≤ (2^31−1)² + 2^61 < 2^63,int64 充分。
- 舍入:算术右移(floor);n = 0 时 y = t。
- 饱和:clamp [-128,127]。
- 布局:输入/输出 CHW;权重 OIHW;填充值 0。

## 5 首版支持与拒绝

仅支持 kernel=3×3、stride=1、pad=1(内建,无配置项);尺寸上限由参数 `MAX_CIN/MAX_COUT/MAX_HW` 决定。任何越界配置必须 error 拒绝,不得静默按其他配置执行。测试台必须覆盖:合法小张量、真实层切片、非法配置、反压、任务中复位、连续任务、超时(见 P5.0c)。

## 6 验收

每个合成/真实层向量:输出逐字节等于期望(期望由 Python 整数参考的 pre-SiLU 边界导出,**不依赖 RTL 实测结果**);协议检查(握手、数量、超时)由测试台断言,任一失败以非零退出码结束;仿真周期数记录但不作为性能声明。
