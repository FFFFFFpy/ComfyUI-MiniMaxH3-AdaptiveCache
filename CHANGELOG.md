# Changelog

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
