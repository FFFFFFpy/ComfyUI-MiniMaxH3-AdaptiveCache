# 预取接口兼容与升级

## 适用错误

```text
install_prefetch_guard.<locals>.guarded_prefetch_queue_pop() got an unexpected keyword argument 'malloc_scope'
```

原生 H3 向 `comfy.model_prefetch.prefetch_queue_pop` 传入 `malloc_scope="block"`。旧版三参数 guard 不接受该参数。2026-09-04 合并的 `868ecc5` 已加入 `*args/**kwargs`；仍出现上述原样错误，说明运行中的函数不是那份新版 guard，不能仅凭磁盘文件已更新判断后端已加载修复。

0.1.2 在已有参数透传修复上进一步处理旧补丁安装状态，并补上缓存命中分支的分配图作用域处理。

## 行为

正常执行、未启用预取、缺少 skip 回调以及非 H3 调用均透传原函数的参数与返回值。

对于原生 H3 的普通预取调用，缓存命中时先以 `queue=None` 调用原函数，让 ComfyUI 自己处理 `malloc_scope`，再同步流、释放已预取状态并推进队列。这样不会预取被跳过 Block 的权重，也无需复制分配图内部逻辑。单参数与双参数 `cleanup_prefetched_modules` 均受支持。

带 `core`、图选项、额外位置参数或未知关键字参数的调用保守走原函数。这些调用不抑制预取，但不会吞掉回调、参数或图行为。缓存控制器本身未改变。

安装时只解开可识别的本插件旧 guard，保留其他插件的外层包装。旧 marker 不再是唯一安装判断依据。无法识别的第三方内部补丁及所有重复安装组合不在保证范围内，完整重启仍是升级后的标准步骤。

## 本地更新

停止 ComfyUI 后端进程，然后在 PowerShell 中执行。路径按实际安装位置调整：

```powershell
$plugin = "D:\MyProjects\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3-AdaptiveCache"
git -C "$plugin" status --short
git -C "$plugin" pull --ff-only origin main
```

如有冲突或无法快进，应先保存和处理本地修改，不要使用 `reset --hard` 强行覆盖。非 Git 安装应替换原插件目录中的文件，不要将新旧副本同时放在 `custom_nodes` 内。

重新启动后执行包含 Adaptive Cache 节点的工作流。节点安装补丁时应出现以下日志，并显示实际的 `runtime_patch.py` 路径：

```text
MiniMax H3 Adaptive Cache: prefetch guard v3 installed from ...runtime_patch.py
```

仅刷新网页不会替换后端内存中的 Python 函数。若仍报同一个参数错误，检查日志中的加载路径，以及 `custom_nodes` 下是否存在重复插件目录。

## 验证

在插件目录运行独立接口回归测试，不需要安装 ComfyUI、PyTorch 或 CUDA：

```console
python -m unittest discover -s tests -p test_runtime_patch.py -v
```

24 项测试覆盖普通/跳过/无队列路径、`malloc_scope` 顺序、流同步、两种 cleanup 签名、回调与扩展参数透传、旧 guard 替换、热重载、第三方外层包装、线程局部状态与诊断日志。

这些测试使用替身验证接口与队列行为，不等同于真实 CUDA 分配图回放或 MiniMax H3 视频生成验证。更新后仍应使用原工作流进行实际生成测试。
