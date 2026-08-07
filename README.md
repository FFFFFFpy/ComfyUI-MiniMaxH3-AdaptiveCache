# ComfyUI MiniMax H3 Adaptive Cache

面向 ComfyUI **原生 MiniMax H3** 的独立加速插件。它会在每次采样中重算前部 DiT Block，并在条件允许时复用后部 Block 对目标音视频状态产生的残差。

插件提供两个节点：

- **MiniMax H3 Adaptive Cache**：预设版，适合日常使用。
- **MiniMax H3 Adaptive Cache (Advanced)**：高级版，提供完整参数调节。

插件不会修改 `comfy/ldm/minimax/model.py`，也不需要运行核心补丁脚本。它直接使用 H3 现有的 `("double_block", index)` ModelPatcher 扩展点，并安装一个只在当前线程、当前缓存命中期间生效的预取守卫。因此，被跳过的尾部 Block 不仅不会执行，也不会被 ComfyUI 的动态权重预取搬入显存。

## 安装

将整个仓库放入：

```text
ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-AdaptiveCache
```

重启 ComfyUI。插件本身不增加第三方 Python 依赖。

## 工作流接法

将节点放在 H3 的 `UNETLoader` 后面：

```text
UNETLoader
   ↓
MiniMax H3 Adaptive Cache
   ├─→ BasicScheduler
   └─→ BasicGuider / CFGGuider
```

`BasicScheduler` 与 `BasicGuider` 必须同时使用加速节点输出的 MODEL。只接其中一条支路会得到结构完整、逻辑错误的工作流。

`examples/video_minimax_h3_i2v_adaptive_cache.json` 提供了可直接加载的 I2V 示例。

### 与 Kijai Sol-Attn 组合

本插件可以和 [`kijai/ComfyUI-SolAttn_triton`](https://github.com/kijai/ComfyUI-SolAttn_triton) 同时使用。Sol-Attn 是独立插件，需要另外安装；本仓库只提供组合兼容和 Morton 安全防护，不包含其 Triton 内核。

推荐顺序：

```text
UNETLoader
   ↓
Patch Sol-Attn
   ↓
MiniMax H3 Adaptive Cache
   ├─→ BasicScheduler
   └─→ BasicGuider / CFGGuider
```

Sol-Attn 负责降低实际执行 Block 内的 Attention 成本，Adaptive Cache 负责跳过满足条件的尾部 Block，两者优化层级不同。

组合时必须关闭 Sol-Attn 的 Morton 重排：

```text
morton = false
```

Morton 会在 Block 栈前重排目标视频 token，并在最后一个 Block 后恢复顺序。缓存命中可能跳过尾部 Block，不能依赖该恢复点；完整缓存捕获也不能把 Morton 顺序的 warm state 与恢复顺序后的 final state 相减。

从 `0.1.1` 起，插件会读取 `transformer_options["sol_morton"]`：

- Morton 关闭：Sol-Attn 与 Adaptive Cache 正常叠加；
- Morton 开启：Adaptive Cache 自动旁路，完整执行全部 DiT Block，Sol-Attn 继续工作；
- 旁路期间不会捕获或复用 residual，控制台会给出一次警告和调用统计；
- 不需要导入或依赖 Kijai 插件的私有 Python 模块。

质量优先的组合起点：

```text
Sol-Attn:
  morton = false
  use_tma = false
  int8_qk = false
  sink_conditioning = exact_kv_and_rows

Adaptive Cache:
  preset = balanced
  cache_device = auto
```

`exact_kv`、`exact_kv_and_rows` 与 `int8_qk` 的实际速度会受到 GPU、序列长度、Triton 版本、内核编译缓存和运行顺序影响。建议先用 BF16 确认质量，再单独测试 INT8，不要一次同时改变多种近似参数。

## 预设

| 预设 | 最多缓存的尾部 Block | 缓存窗口 | 连续缓存上限 | 内容保护 |
|---|---:|---:|---:|---|
| Safe | 约 60% | 15%–85% | 1 | 严格检查音频与视频 |
| Balanced | 约 75% | 10%–90% | 2 | 自适应检查音频与视频 |
| Fast | 约 84% | 8%–92% | 3 | 较宽松检查音频与视频 |
| Sigma Only | 约 75% | 10%–90% | 2 | 关闭，只看 sigma |

建议从 `Balanced + auto` 开始。该算法属于有损近似，即使 seed 相同，输出也不会与官方完整计算逐像素一致。

以下场景优先使用 `Safe`：

- 人脸或角色身份必须稳定；
- 高速攻击、爆炸和剧烈运动；
- 硬切镜头；
- 对白口型或音效时间点很重要；
- 多参考图、参考视频较复杂的 Ref2VA。

## 算法流程

以 50 层 H3、Balanced 预设为例：

1. 完整调用执行全部 50 层。
2. 执行约第 12 层后，只截取目标音频和目标视频的 hidden state。
3. 第 50 层结束后保存：

   ```text
   tail_residual = final_target - warm_target
   ```

4. 下一次缓存候选仍正常执行前 12 层。
5. 从目标音视频中抽样少量 token 和 channel，计算对称相对变化量。
6. 同时满足以下条件才命中缓存：
   - 位于允许的采样窗口；
   - 相邻 sigma 变化低于阈值；
   - 内容变化没有超过动态 EMA 阈值与硬上限；
   - 没有超过连续缓存次数。
7. 命中后把 residual 加到当前目标音视频状态，跳过第 13–50 层。
8. 预取守卫同步阻止这些尾部 Block 的权重传输。

插件只缓存最终输出层会读取的目标音频与目标视频区间。文本、首尾帧条件、参考图片、参考视频与参考音频的尾部 hidden state 不会保存。Ref2VA 条件较多时，这可以减少缓存显存和传输量。

## 内容保护

内容变化使用音频和视频 warm state 的抽样探针，不扫描整条 packed sequence。变化量采用对称相对差：

```text
mean(abs(current - previous))
────────────────────────────────────────────
0.5 × (mean(abs(current)) + mean(abs(previous)))
```

判断同时使用：

- 移动平均阈值：适应不同分辨率、提示词和采样阶段；
- 硬上限：避免第一次缓存候选因为尚未建立 EMA 而被无条件接受；
- 音视频加权：默认视频 80%，音频 20%。

## 缓存位置

- `gpu`：读取最快，但较高分辨率下 residual 可能占用数百 MB 显存。
- `cpu`：存放在 pinned host memory，CUDA 环境下会尽量在前部 Block 计算期间异步搬回 GPU。
- `auto`：只有 residual 加安全余量能放进当前空闲显存时才选择 GPU，否则选择 CPU。

对 16GB 显卡，建议优先使用 `auto`。

## 性能说明

### Block 级理论收益

以 50 层模型为例，Balanced 缓存步只执行约 12 层。单个缓存命中调用可以减少约 76% 的 Block 执行与尾部权重搬运。

这不等于端到端耗时下降 76%。文本编码、VAE、视频封装、完整刷新调用、必须执行的前部 Block、显存调度和磁盘/内存缓存仍会占用时间。

### RTX 5090 组合实测：20 步 / Turbo-4 / Turbo-8

这组测试比下面的 RTX 4090 初测更完整：同一套工作流连续测试 18 种组合，每种配置生成 2 次并取平均。输出为 `1376×768`、124 帧、24 FPS（约 5.17 秒视频）。

测试环境：

- GPU：NVIDIA GeForce RTX 5090 32GB；
- PyTorch：`2.12.1+cu130`；
- ComfyUI：`0.30.0`；
- MiniMax H3 DiT：W4A8 混合量化测试环境；
- Adaptive Cache：`Balanced`，本轮 residual 落在 GPU；
- EasyCache：`threshold=0.2, start_percent=0.15, end_percent=0.95`。

| 配置 | 两次总耗时 | 平均 | 相对 20 步原版 |
|---|---:|---:|---:|
| 原版 20 步 | 303.72 / 290.91 s | **297.32 s** | 1.00× |
| Sol-Attn | 184.73 / 176.93 s | **180.83 s** | **1.64×** |
| Adaptive Cache | 186.36 / 186.11 s | **186.24 s** | **1.60×** |
| Sol-Attn + Adaptive Cache | 121.98 / 121.79 s | **121.89 s** | **2.44×** |
| EasyCache | 194.54 / 180.58 s | **187.56 s** | **1.59×** |
| Sol-Attn + EasyCache | 137.64 / 137.44 s | **137.54 s** | **2.16×** |
| Turbo-4 | 73.21 / 73.17 s | **73.19 s** | **4.06×** |
| Sol-Attn + Turbo-4 | 48.53 / 48.71 s | **48.62 s** | **6.12×** |
| Adaptive Cache + Turbo-4 | 51.08 / 51.57 s | **51.33 s** | **5.79×** |
| Sol-Attn + Adaptive Cache + Turbo-4 | 39.61 / 39.18 s | **39.40 s** | **7.55×** |
| EasyCache + Turbo-4 | 73.57 / 73.08 s | **73.33 s** | **4.05×** |
| Sol-Attn + EasyCache + Turbo-4 | 48.87 / 48.60 s | **48.74 s** | **6.10×** |
| Turbo-8 | 131.62 / 131.80 s | **131.71 s** | **2.26×** |
| Sol-Attn + Turbo-8 | 82.41 / 82.17 s | **82.29 s** | **3.61×** |
| Adaptive Cache + Turbo-8 | 87.71 / 87.42 s | **87.57 s** | **3.40×** |
| Sol-Attn + Adaptive Cache + Turbo-8 | 56.99 / 57.14 s | **57.07 s** | **5.21×** |
| EasyCache + Turbo-8 | 131.74 / 131.70 s | **131.72 s** | **2.26×** |
| Sol-Attn + EasyCache + Turbo-8 | 82.36 / 82.36 s | **82.36 s** | **3.61×** |

这轮数据里有几个比较稳定的结论：

- **Sol-Attn 与 Adaptive Cache 可以有效叠加。** 20 步原版从 297.32 秒降到 121.89 秒；Sol-Attn 降低实际执行 Block 的 Attention 成本，Adaptive Cache 减少实际执行的尾部 Block，优化层级不同。
- **Adaptive Cache 单独也有明确收益。** 20 步从 297.32 秒降到 186.24 秒，约 1.60×；Balanced 典型统计为 `full=10, cache=10, executed=620/1000 blocks, block reduction=38%`。
- **Turbo-8 与 Sol-Attn + Adaptive Cache 在性能上仍能继续叠加。** 本轮最快的 Turbo-8 组合为 57.07 秒，相对 20 步原版约 5.21×。
- **Turbo-4 的极限性能更高。** `Sol-Attn + Adaptive Cache + Turbo-4` 达到 39.40 秒，约 7.55×；但 Turbo 相关画质必须单独评估，不能只看耗时。
- **EasyCache 在正常 20 步下有效，但在本轮 Turbo-4 / Turbo-8 下没有实际跳步。** 日志分别出现 `skipped 0/4 steps` 与 `skipped 0/8 steps`，加入 EasyCache 后总耗时也与对应 Turbo 基线几乎一致，因此不建议为了 Turbo 组合额外挂 EasyCache。

#### Turbo + W4A8 画质说明

本轮 Turbo 测试使用 W4A8 混合量化环境。部分 Turbo-4 样本出现明显画质异常，尤其在继续叠加缓存近似时更容易暴露，但**当前数据不足以证明 Adaptive Cache 与 Turbo 本身不兼容**。

目前更合理的待验证假设是：Turbo 的少步采样 / LoRA 与 W4A8 量化近似之间存在兼容性或误差放大问题；Adaptive Cache 可能进一步放大已有误差，但未必是根因。要区分这些因素，需要在完全相同 seed、提示词和输入下至少补充 BF16/FP16 或其他量化格式的 Turbo 对照。

因此当前建议是：

- 使用 Turbo 前先单独验证所用量化模型的画质；
- W4A8 + Turbo 组合暂时视为实验性配置，不把异常直接归因于 Adaptive Cache；
- 若要评估 Adaptive Cache 与 Turbo 的质量兼容性，应先在非 W4A8 基线下做同 seed A/B；
- 速度结果可以参考上表，但不能把这组 W4A8 Turbo 样本直接当成通用质量结论。

另外，Turbo 将采样器本身压缩到很短以后，端到端瓶颈开始转向固定开销。以最快的 Turbo 组合为例，sampler 约 23.8 秒，而总工作流约 39.4 秒，说明约 15～16 秒已经来自 VAE、封装和其他前后处理。继续只优化 DiT 的边际收益会越来越小。

### RTX 4090 初步实测

以下数据来自同一套 MiniMax H3 工作流，分辨率为 `960×640`，单位为秒。它们是实际使用过程中的单次记录，不是严格控制预热、运行顺序和重复次数的正式 benchmark。

| 编号 | Sol-Attn 条件保护 | INT8 QK | Adaptive Cache | 耗时 | 相对原生 |
|---:|---|:---:|:---:|---:|---:|
| 1 | `exact_kv` | 开 | 开 | 164.53 | -0.4% |
| 2 | `exact_kv` | 关 | 开 | 124.04 | +24.3% |
| 3 | `exact_kv_and_rows` | 开 | 开 | 112.58 | +31.3% |
| 4 | `exact_kv_and_rows` | 关 | 开 | 114.67 | +30.0% |
| 5 | 关闭 | - | 开 | 159.56 | +2.6% |
| 6 | 关闭 | - | 关 | 163.89 | 基准 |
| 7 | `exact_kv` | 开 | 关 | 148.25 | +9.5% |
| 8 | `exact_kv` | 关 | 关 | 150.71 | +8.0% |
| 9 | `exact_kv_and_rows` | 开 | 关 | 150.22 | +8.3% |
| 10 | `exact_kv_and_rows` | 关 | 关 | 156.99 | +4.2% |

相同测试过程中，Balanced 的控制台统计通常为：

```text
full=10, cache=10, executed=620/1000 blocks,
block reduction=38.0%, content rejects=1, residual=gpu
```

这组数据能支持的结论有限：

- Sol-Attn 单独运行在该环境中获得了约 4%～10% 的单次端到端收益；
- Adaptive Cache 单独一轮只表现出约 2.6% 的端到端收益，尽管 Block 工作量下降了 38%；
- Sol-Attn 与 Adaptive Cache 组合的较好单次结果为约 24%～31%；
- `exact_kv + INT8 + Cache` 的第一次结果明显偏慢，很可能受到首次 Triton 编译、autotune、模型加载或运行顺序影响；
- `exact_kv_and_rows` 理论计算量更高，却在部分单次记录中更快，现有样本不足以把它解释为稳定规律；
- INT8 开关在当前样本中没有显示出稳定、可重复的速度优势。

较高分辨率 `1152×928` 下，`exact_kv + INT8 + Cache` 连续两次记录为：

| 次数 | 耗时 |
|---:|---:|
| 第一次 | 322.67 秒 |
| 第二次 | 241.46 秒 |

同一配置第二次快约 25%，说明新的分辨率或形状可能触发显著的 Triton 编译、autotune 和缓存建立成本。该分辨率没有原生对照，因此不能据此计算加速比例。

### 为什么单次测试容易误导

以下状态会跨运行保留或随运行顺序变化：

- Windows 文件系统缓存；
- 模型权重在内存、显存和动态加载队列中的状态；
- CUDA、TorchInductor 与 Triton 编译缓存；
- Triton autotune 结果；
- ComfyUI Manager 等后台任务；
- 不同分辨率、帧数和 packed sequence 长度对应的内核形状。

因此，“单独 Cache 收益很低”可能是这次运行顺序和已有缓存状态造成的，也可能反映该工作流中非 Block 开销占比较高。现有数据不足以区分两者。

更可靠的测试方式：

1. 固定模型、工作流、seed、提示词、分辨率、帧数和采样参数；
2. 每种配置先运行一次预热，不计成绩；
3. 再连续运行至少 3 次，记录中位数；
4. 等待 ComfyUI Manager 等后台更新完成；
5. 同时记录总耗时、Adaptive Cache 统计、Sol-Attn verbose 日志和峰值显存；
6. 对人脸、手部、快速运动、口型和音频同步进行固定 seed 质量对比。

性能数字应被视为当前环境的观察值，不应直接外推到其他 GPU、分辨率、帧数、Triton 版本或采样器。软件最擅长的事情之一，就是让一个看似精确的秒数拥有非常模糊的含义。

## 控制台统计

一次采样结束后，插件按条件通道打印：

- 完整调用与缓存调用次数；
- 实际执行和跳过的 Block 数；
- Block 工作量减少比例；
- 内容保护拒绝次数；
- 平均内容变化量；
- 插件包围的模型调用耗时；
- residual 最终存放位置；
- Sol-Attn Morton 导致的安全旁路调用数。

`block reduction` 不是总工作流速度提升。

## 高级节点参数

| 参数 | 作用 |
|---|---|
| `cache_depth` | 命中时跳过的尾部 Block 比例 |
| `sigma_threshold` | 相邻 sigma 最大允许变化 |
| `window_start/end` | 允许缓存的采样进度区间 |
| `max_consecutive` | 连续缓存次数上限 |
| `quality_guard` | 音视频检查、仅视频或关闭 |
| `content_multiplier` | 相对 EMA 的宽松程度，越低越保守 |
| `content_ceiling` | 内容变化硬上限 |
| `cache_device` | auto/gpu/cpu |
| `gpu_safety_mb` | auto 模式保留的显存安全余量 |

## 兼容性

插件依赖 ComfyUI 原生 H3 具备以下内部接口：

- `MiniMaxH3Model.blocks`；
- 每层调用 `("double_block", index)` 替换点；
- Block wrapper 参数包含 `img`、`mod_segments` 与 `transformer_options`；
- `transformer_options` 包含 `sigmas`、`sample_sigmas`、`cond_or_uncond`；
- ModelPatcher 提供 `OUTER_SAMPLE` wrapper API。

当前实现依据 ComfyUI 的 MiniMax H3 支持版本编写。若未来 ComfyUI 改变这些内部接口，插件会明确报错。

可以与使用 `optimized_attention_override` 的 Attention 后端组合，包括 Kijai Sol-Attn。Sol-Attn 的 Morton 模式会被自动识别并安全旁路 Adaptive Cache。其他会替换 MiniMax H3 `double_block` 的缓存、跳层或 Block Patch 节点仍不建议叠加，相同扩展点只能有一套最终实现。

Turbo LoRA / Turbo sampler 在执行路径上可以与 Adaptive Cache 同时工作，5090 实测也能继续获得性能叠加；但当前 W4A8 测试中的 Turbo 画质存在未解决变量，暂不把它列为“已验证无损兼容”。质量结论请参考上面的 **Turbo + W4A8 画质说明**。

## 测试

在插件目录执行：

```bash
python -m unittest discover -s tests -v
```

当前测试覆盖：

- 尾部 Block 确实被跳过；
- residual 只作用于目标音视频区间；
- 不同条件通道缓存隔离；
- 内容突变强制完整刷新；
- 关闭缓存时完整执行；
- 节点注册 50 个 Block hook 与采样生命周期 hook；
- 预取守卫阻止被跳过 Block 的预取；
- Sol-Attn Morton 开启时完整执行全部 Block；
- Morton 旁路不会保存或复用 residual；
- 从 Morton 返回正常 token 顺序后强制完整刷新。

## 版本

当前版本：`0.1.1`

## 许可证

GPL-3.0-or-later。
