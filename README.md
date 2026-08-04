# ComfyUI MiniMax H3 Adaptive Cache

面向 ComfyUI **原生 MiniMax H3** 的独立加速插件。它会在每次采样中重算前部 DiT Block，并在条件允许时复用后部 Block 对目标音视频状态产生的残差。

插件提供两个全新节点：

- **MiniMax H3 Adaptive Cache**：预设版，日常使用。
- **MiniMax H3 Adaptive Cache (Advanced)**：高级版，完整参数调节。

它不会修改 `comfy/ldm/minimax/model.py`，不需要运行补丁脚本。插件直接使用 H3 现有的 `("double_block", index)` ModelPatcher 扩展点，并安装一个只在当前线程、当前缓存命中期间生效的预取守卫。因此，被跳过的尾部 Block 不仅不执行，也不会被 ComfyUI 的动态权重预取搬入显存。

## 安装

把整个文件夹复制到：

```text
ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-AdaptiveCache
```

重启 ComfyUI。插件不增加任何第三方 Python 依赖。

## 工作流接法

将节点放在 H3 的 `UNETLoader` 后面：

```text
UNETLoader
   ↓
MiniMax H3 Adaptive Cache
   ├─→ BasicScheduler
   └─→ BasicGuider / CFGGuider
```

`BasicScheduler` 与 `BasicGuider` 必须同时使用加速节点输出的 MODEL。只接其中一条支路会得到一个结构完整、逻辑荒唐的工作流，ComfyUI 在这方面一向很尊重用户的自由意志。

`examples/video_minimax_h3_i2v_adaptive_cache.json` 已根据本次提供的 I2V 工作流改好，可直接加载。

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

插件只缓存最终输出层会读取的目标音频与目标视频区间。文本、首尾帧条件、参考图片、参考视频与参考音频的尾部 hidden state 不会保存。Ref2VA 条件很多时，这一点可以显著减少缓存显存和传输量。

## 内容保护

内容变化使用音频和视频 warm state 的抽样探针，不扫描整条 packed sequence。变化量采用对称相对差，理论范围更稳定：

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
- `cpu`：放在 pinned host memory，CUDA 环境下会尽量在前部 Block 计算期间异步搬回 GPU。
- `auto`：只有 residual 加安全余量能放进当前空闲显存时才选择 GPU，否则选择 CPU。

对 16GB 显卡，`auto` 比“我觉得应该塞得下”这种传统显存管理策略可靠一点。

## 预计性能

以 50 层模型为例，Balanced 缓存步只执行约 12 层，单个命中调用可减少约 76% 的 Block 计算与尾部权重搬运。

实际总工作流提升取决于缓存命中率、分辨率、时长、权重卸载、Qwen 编码和 VAE 解码：

| 场景 | 相对官方完整流程的预期总耗时下降 |
|---|---:|
| Safe | 约 20%–35% |
| Balanced | 约 30%–45% |
| Fast、稳定慢镜头 | 约 40%–52% |
| 高动态内容 | 命中率下降，主要收益是避免错误缓存 |

以上是按执行层数和 H3 权重流式加载结构推算的工程区间，不是当前包在 RTX 5080 上完成的实测成绩。本开发环境没有 H3 权重与 CUDA GPU，因此只完成 CPU 逻辑、节点注册和预取守卫测试。需要在真实 ComfyUI 环境中做固定 seed 的官方节点对照测试。

## 控制台统计

一次采样结束后，插件按条件通道打印：

- 完整调用与缓存调用次数；
- 实际执行和跳过的 Block 数；
- Block 工作量减少比例；
- 内容保护拒绝次数；
- 平均内容变化量；
- 插件包围的模型调用耗时；
- residual 最终存放位置。

`block reduction` 不是总工作流速度提升。Qwen 编码、VAE 解码、视频封装和必须执行的前部 Block 仍然存在，宇宙没有突然开始发放免费算力。

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

当前实现依据 ComfyUI 的 MiniMax H3 支持版本编写。若未来 ComfyUI 改变这些内部接口，插件会明确报错，而不是礼貌地显示“已加速”然后什么都没做。

不建议与其他会替换 MiniMax H3 `double_block` 的缓存、跳层或 Block Patch 节点叠加。相同扩展点只能有一套最终实现，节点堆叠并不会形成复利。

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
- 预取守卫阻止被跳过 Block 的预取。

## 版本

当前版本：`0.1.0`

## 许可证

GPL-3.0-or-later。
