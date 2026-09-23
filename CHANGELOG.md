# Changelog

## 0.1.2

- 升级为 prefetch guard v3，保留 `*args/**kwargs` 透传，兼容原生 H3 的 `malloc_scope="block"` 调用。
- 缓存命中时先调用 ComfyUI 的无队列路径推进分配图作用域，再清理已预取权重，保持原生执行顺序与队列对齐。
- 携带 `core`、图选项、额外位置参数或未知关键字参数时保守回退原函数，不吞掉回调或扩展行为。
- 安装检查同时核对实际函数；升级时解开本插件旧 guard，避免旧 marker、嵌套补丁或过期原函数引用阻止修复。
- 保留第三方函数包装，阻止旧版插件重新覆盖新版 guard，并输出补丁版本和实际加载路径。
- 保留单参数与双参数 cleanup API 的兼容性；新增 24 项无需 GPU 的独立回归测试。
- 新增[预取接口兼容与升级说明](docs/prefetch-compatibility.md)。本次不修改缓存算法、节点输入或模型权重。

## 0.1.1

- 新增 Kijai `ComfyUI-SolAttn_triton` 组合兼容层。
- Sol-Attn Morton 关闭时，Attention 稀疏与尾部 Block 缓存可以同时工作。
- 检测到 `transformer_options["sol_morton"]` 时自动旁路 Adaptive Cache，完整执行全部 DiT Block，同时保留 Sol-Attn。
- Morton 旁路不会捕获或复用 residual，避免 token 重排状态与最终恢复状态混算。
- 从 Morton 返回正常 token 顺序时强制重新建立缓存。
- 新增 Morton 旁路警告、调用统计与单元测试。
- 更新节点说明与组合工作流文档。

## 0.1.0

- 新增 `MiniMaxH3AdaptiveCache` 预设节点。
- 新增 `MiniMaxH3AdaptiveCacheAdvanced` 高级节点。
- 使用现有 `double_block` hook，无需修改 ComfyUI 核心文件。
- 新增目标音视频区间 residual，不缓存文本和参考条件尾部状态。
- 新增内容变化 EMA、硬上限和音视频加权保护。
- 新增按条件通道和设备隔离的缓存状态。
- 新增 GPU、pinned CPU 与自动 residual 放置。
- 新增线程局部 Block 预取抑制。
- 新增 OUTER_SAMPLE 生命周期清理，支持中断后的安全重置。
- 新增 I2V 示例工作流和 CPU 单元测试。
