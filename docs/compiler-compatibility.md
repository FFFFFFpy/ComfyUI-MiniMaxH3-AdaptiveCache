# Comfy 编译器与缓存张量生命周期（0.1.3）

## 问题与证据

本次针对的场景是：0.1.2 的 prefetch guard v3 已加载，原来的 `malloc_scope` 参数错误已消失，但原生 MiniMax H3 在缓存收尾路径发生 `Fatal Python error: Aborted`；同一工作流添加 `--disable-comfy-compiler` 后可以完成。

代码中可以确认一个分配图边界缺陷：`warm_snapshot` 跨 Block 存活，`previous_probes` 和 residual 跨模型调用存活，但插件在 Comfy 的单个 Block 分配图作用域中创建、保存和清理这些张量，没有隔离插件缓存的生命周期。上游同一版本的 H3 FunControl 已使用 `pause_malloc_graph()` 将控制状态放在基本 Block 的分配图之外。

上述缺陷由本次补丁修正。用户日志没有原生故障回溯；CPU 测试不能证明该缺陷是该 Windows/CUDA 崩溃的唯一原因，也不能代替实际 H3 生成验收。

## 修复范围

`_cache_tensor_operation` 使用 Comfy 公开的 `pause_malloc_graph()` 上下文，只包裹控制器的六个缓存操作入口：`start_sampling_run`、`end_sampling_run`、`begin_call`、`after_warm`、`finish_call`、`abort_call`。缓存快照、探针、残差的创建、传输、注入和释放不再被算作基本 Block 的分配序列。

原始 `original_block` 调用没有包裹在暂停上下文中；上下文退出后，模型继续正常使用编译器。插件不修改 `disable_comfy_compiler`、`disable_cuda_graphs`、DynamicVRAM 或其他全局选项，不结束、销毁或重建模型分配图。0.1.2 的预取 guard 及其作用域推进逻辑保持不变。

暂停状态使用线程局部嵌套保护。AIMDO 的暂停接口本身是布尔值，插件内部嵌套缓存操作不能提前恢复外层暂停。旧版 Comfy 缺少该 API、或在独立 CPU 环境使用缓存控制器时，保持原来的执行方式；公开 API 的真实错误会继续向外传播，不以静默禁用编译器掩盖错误。

另修复一项异步残差生命周期问题：预取张量在侧 CUDA 流上分配、在计算流上参与残差加法。仅等待复制完成事件不能保护计算流上尚未结束的使用。`materialize()` 现在等待生产事件后，为每个预取张量调用 `record_stream(consumer)`，避免清理 Python 引用后内存过早复用。

缓存阈值、命中判定、目标音视频行、节点输入和 Sol-Attn Morton 既有保护规则均不改变。该补丁没有对第三方 `torch.compile` / 整个模型 CUDA Graph 捕获作出新的兼容保证。

## 已执行的验证

执行环境为 Linux、PyTorch 2.10.0+cpu，没有 CUDA GPU。

```console
python -m unittest discover -s tests -p test_compiler_compat.py -v
python -m unittest discover -s tests -p test_adaptive_cache.py -v
```

新增 23 项边界回归测试、原有 5 项缓存算法测试，合计 28 项通过；修改文件通过 Python 3.10 语法解析检查。新测试使用真实 CPU 张量追踪存储分配所在的作用域，以及 Comfy 暂停 API 和 CUDA 流的替身，覆盖：

- 跨 Block / 跨模型调用的快照、探针、残差存储；完整计算、缓存命中、强制刷新和残差存储复用。
- 目标行数值、内容拒绝、形状变化、sigma 重置、条件通道隔离。
- 正常清理、采样中断、Block 异常、Sol-Attn Morton 状态切换。
- 原始 Block 保持编译器作用域、全局开关不变、线程隔离、嵌套暂停、旧版与独立环境。
- 预取复制事件与消费流 `record_stream` 调用顺序。

未执行真实 AIMDO/CUDA 分配图测试、H3 视频生成、显存或速度基准；本次也未重跑仓库其他原有测试文件。不能将上述 CPU 结果写成“全部 GPU 测试通过”或“性能无损”。

## 本地验收

停止 ComfyUI 后端，在插件目录拉取 `main`，确认 `pyproject.toml` 版本为 `0.1.3`。移除临时的 `--disable-comfy-compiler`，完整重启后端，运行原来的 H3 工作流。

本次修复在 `adaptive_cache.py`，预取安装日志继续显示 `prefetch guard v3` 是正常现象，不代表未更新。

验收应检查：编译器开启时生成完成且不再 `Aborted`；符合阈值时仍有 cache 命中；CPU / GPU residual 放置分别运行；变更种子后连续生成、取消后重新生成均正常。`graph breaks` 和 `rogues` 是诊断统计，不将它们必须为零作为验收条件。任何速度结论必须另做同条件实测。

## 源码依据

- ComfyUI 用户版本：`2255709aa0be2deade91c7c80cda49d31b73906f`。
- [H3 模型分配图作用域](https://github.com/Comfy-Org/ComfyUI/blob/2255709aa0be2deade91c7c80cda49d31b73906f/comfy/ldm/minimax/model.py)。
- [H3 FunControl 的公开暂停边界](https://github.com/Comfy-Org/ComfyUI/blob/2255709aa0be2deade91c7c80cda49d31b73906f/comfy_extras/nodes_minimax_h3.py)。
- [Comfy 暂停上下文实现](https://github.com/Comfy-Org/ComfyUI/blob/2255709aa0be2deade91c7c80cda49d31b73906f/comfy/model_prefetch.py)。
- [AIMDO 暂停与分配实现](https://github.com/Comfy-Org/comfy-aimdo/blob/3b8e8c162efeb9470d912609a7a6e7a2b1c693ec/src/malloc-graph.c)。
- [PyTorch record_stream 文档](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)。
