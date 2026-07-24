# apex-llm smoke 进展（2026-07-24 会话）

> 接续「进度维护_apex_sglang_kernel_rebuild.md」（sgl-kernel 编译已 OK）。
> 本次会话目标：继续排查 RolloutManager Ray worker 崩溃，跑通 smoke（pure_opd + ta_opd）。

## 1. 关键突破：定位 RolloutManager SIGSEGV 根因

**现象**：smoke 跑到 `create_rollout_manager` 后 Ray worker 立刻死（SYSTEM_ERROR / connection EOF），
无 Python traceback，无 dmesg segfault 记录，无 OOM-kill。每个单独的组件（args 解序列化、
RolloutManager 模块 import、trivial actor 拿 args/pg）都工作，唯独"RolloutManager 作为 Ray actor
且带 real args"组合就崩。

**定位过程**：
1. strace -ff 跟了 probe15 的 worker，发现 5 个进程全被 SIGSEGV 杀，`si_addr=0x350`
   （NULL 指针 + struct offset）。但 strace 只记系统调用，崩发生在用户态没系统调用处，
   看不出调用栈。
2. `conda install gdb`（17.1）到 ta_opd，写 `gdb_probe2.py` 直接实例化 RolloutManager
   （绕过 Ray），gdb batch 模式抓 backtrace。
3. gdb 抓到崩在：
   ```
   Thread 130 "jemalloc_bg_thd" received signal SIGSEGV, Segmentation fault.
   0x00007ffd89ae7771 in background_thread_entry ()
     from /home/kejiechen/miniconda3/envs/ta_opd/lib/python3.10/site-packages/pyarrow/libarrow.so.2500
   ```

**根因**：pyarrow 25.0.0 wheel 同时捆绑 jemalloc + mimalloc，默认 memory pool 是 mimalloc，
但 libarrow.so 里 jemalloc 的后台线程（`background_thread_entry`）还是会被启动。
该线程访问 jemalloc 的 arena 数据结构，但那些结构从未被初始化（pool 走的是 mimalloc），
结果段错误。这个后台线程在 libarrow 被 dlopen 时就 fork 出来，所以崩得早，
甚至 RolloutManager.__init__ 第一行 Python 代码都没跑。

**验证**：把 `ARROW_DEFAULT_MEMORY_POOL=system` + `pyarrow.set_memory_pool(system_memory_pool())`
放进 `sitecustomize.py`，让它在所有 Python 进程里、任何 import 之前生效。
直接实例化 RolloutManager 不再段错误。

## 2. 随之而来的连带问题（修 pyarrow 后冒出来）

### 2.1 sgl_kernel: `GLIBCXX_3.4.29 not found`
- 现象：smoke 刚起就报 `ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version GLIBCXX_3.4.29 not found`
- 根因：之前 `LD_LIBRARY_PATH=/lib:` 只用系统的 libstdc++（GLIBCXX 最高 3.4.28），
  但本机编译的 sgl_kernel common_ops.abi3.so 是用 conda g++ 12.4 编的，链接了 conda
  libstdc++（6.0.34，GLIBCXX 最高 3.4.34）。
- 修：`LD_LIBRARY_PATH=$ENV/lib:/lib:`，把 conda lib 放前面。
- 安全：之前出问题的 `libpthread GLIBC_PRIVATE`（§3 第 10 步）是因为误加了
  `/usr/lib/x86_64-linux-gnu`，那个目录的 libpthread 与 conda libc 不兼容；
  `$ENV/lib` 是 conda 自带的、与 conda libc 匹配，不会重蹈覆辙。

### 2.2 curl: `libp11-kit undefined symbol ffi_type_pointer, LIBFFI_BASE_7.0`
- 现象：smoke 里 `curl -sf ... /health_generate` 失败，teacher "Waiting for server" 死循环。
- 根因：`LD_LIBRARY_PATH=$ENV/lib:...` 让 conda 的 libffi 被加载（符号 `ffi_type_pointer`
  版本 `LIBFFI_BASE_7.0`），但系统的 `libp11-kit.so.0` 期望的是系统 libffi 的符号版本。
- 修：写 `/tmp/kejiechen/bin/curl` 小 wrapper，调用 `/usr/bin/curl` 之前 `env -u LD_LIBRARY_PATH`。
  把 `/tmp/kejiechen/bin` 放在 PATH 最前。smoke 里 `curl` 调用自动走 wrapper。
- 备选：`conda install curl`（8.21，链接 conda libffi），但脚本里 `curl` 仍解析成 `/usr/bin/curl`
  因为 `/usr/bin` 在 PATH 里也靠前；wrapper 更可控。

### 2.3 slime ReloadableProcessGroup `__init__() incompatible constructor arguments`
- 现象：Megatron actor 里 `mpu.initialize_model_parallel` 调 `torch.distributed.new_group`
  （已被 slime 替换为 `ReloadableProcessGroup`）时 TypeError：
  ```
  __init__() incompatible constructor arguments. Supported:
    1. ProcessGroup(arg0: int, arg1: int)
    2. ProcessGroup(arg0: Store, arg1: int, arg2: int, arg3: Options)
  Invoked with: kwargs: rank=1, size=2
  ```
- 根因：slime 代码写 `super().__init__(rank=..., size=...)`（keyword args），
  但 torch 2.5.1 的 C++ ProcessGroup 只接受 positional `(rank, size)` 或 `(store, rank, size, opts)`。
  这是 slime 针对更新版 torch 写的（新版加了 kwargs 支持），在 2.5.1 上语法不兼容。
- 修：`slime/utils/reloadable_process_group.py` 里直接去掉 `super().__init__()` 调用。
  `ReloadableProcessGroup` 用 `__getattr__` 把所有方法（`rank()`、`size()`、`allreduce()`、
  `barrier()` 等）都转发给内层的真实 `self.group`（`torch.distributed.new_group` 返回的
  NCCL/Gloo PG），C++ 层的 `ProcessGroup` 基类从不被直接调用，所以不必初始化。
  保留继承关系是为了满足 `isinstance(x, dist.ProcessGroup)` 类型判断。

## 3. 当前状态（截至本次会话未结束）

- pyarrow segfault ✅ 已修（sitecustomize.py 注入 system pool）
- sgl_kernel GLIBCXX ✅ 已修（LD_LIBRARY_PATH 加 $ENV/lib）
- curl libffi ✅ 已修（/tmp/kejiechen/bin/curl wrapper）
- ReloadableProcessGroup TypeError ✅ 已修（去掉 super().__init__）
- 第四次 smoke（`smoke_20260724_121422.log`）已发起，等待结果。

## 4. 改动清单

### A. 环境侧（ta_opd conda env）
- `conda install gdb`（17.1）：诊断用。
- `conda install curl`（8.21）：实际未生效（PATH 顺序），改用 wrapper。
- `/home/kejiechen/miniconda3/envs/ta_opd/lib/python3.10/site-packages/sitecustomize.py`：
  在 pytrace log + faulthandler 基础上增加 pyarrow system pool 切换。

### B. slime 代码改动
- `slime_ta_opd/slime/utils/reloadable_process_group.py`：去掉 `super().__init__(rank=..., size=...)`，
  保留 `self.group = ...` 与 `__getattr__` 转发。

### C. 启动脚本（apex 运行用，不进仓库）
- `/tmp/kejiechen/bin/curl`：wrapper，去 LD_LIBRARY_PATH 后调 `/usr/bin/curl`。
- `/tmp/kejiechen/run_smoke_now.sh`：导 PATH/CUDA_HOME/LD_LIBRARY_PATH=$ENV/lib:/lib:/...、
  ulimit -n 1048576，nohup 跑 `指令_smoke_4b_to_1p7b_apex.txt`。

## 5. 下一步

- 等 smoke 跑完，确认 pure_opd / ta_opd 两个目录都产出 `latest_checkpointed_iteration.txt`。
- 若还有错，继续打补丁（可能是 sglang rollout server 启动、Megatron ckpt 转换、
  DAPO 数据 pipeline、reward 计算等后续环节）。
- smoke 跑通后：把 sitecustomize.py 与 ReloadableProcessGroup 补丁提交到仓库对应位置
  （按项目"远程 git pull 后执行"的约定）。

## 6. 补充：pybind11 强制 `super().__init__()` 的二次修复

去掉 `super().__init__()` 后报错：
```
TypeError: torch._C._distributed_c10d.ProcessGroup.__init__() must be called when overriding __init__
```

pybind11 在 `py::init()` 里加了一个"必须调用父 init"的运行时检查（即使 C++ 基类可以被空构造）。
torch 2.5.1 接受的 `ProcessGroup.__init__` 只支持两种 positional 形式：
1. `(rank: int, size: int)`
2. `(store: Store, rank: int, size: int, opts: Options)`

**修法**：`super().__init__(0, 0)` —— 用 dummy 的 rank=0, size=0 让 pybind11 检查通过；
实际 `rank()`/`size()` 等调用会走 `__getattr__` 转发到 `self.group`，所以 dummy 值不会被观察到。

第四次 smoke（`smoke_20260724_121422.log`，已 kill）用的是"去掉 super init"版本。
第五次 smoke（`smoke_20260724_122336.log`，正在跑）用的是 `super().__init__(0, 0)` 版本。


## 7. 第五次 smoke（`smoke_20260724_122336.log`）：首次完成 rollout+train，卡在 ckpt save

好消息：前 4 个拦路虎全部过了 —— pyarrow segfault、sgl_kernel GLIBCXX、curl libffi、
ProcessGroup 构造器，全 OK。**smoke 第一次跑到 train step 并尝试 `save_checkpoint`**。

新错误：
```
megatron.core.dist_checkpointing.core.CheckpointingException:
  Uneven sharding not supported for PyTorch version 2.5.1+cu124
```
栈顶：
```
dist_checkpointing/strategies/torch.py:311 _mcore_to_dcp_compatible_tensor
```

代码逻辑：
```python
if (not is_pre_mcore_014_sh_ten or not sh_tens[0].has_regular_grid) \
        and is_torch_min_version("2.6a0"):
    # 用 CheckpointableShardedTensor（torch 2.6+ 新增的 _Checkpointable 协议）
    ...
else:
    if not sh_tens[0].has_regular_grid and not is_torch_min_version("2.6a0"):
        raise CheckpointingException("Uneven sharding not supported ...")
    # 旧路径：sharded_tensor_to_torch_sharded_tensor（要求 regular grid）
```

含义：
- 某些 sh_ten 的 `has_regular_grid` 为 False（比如 vocab=151936 不能被 dp_size=2
  整除、或者 `prepend_axis_num > 0`/`flattened_range is not None` 导致 grid 不齐）。
- Megatron 官方只给 torch >=2.6 提供了"不规则 sharding"的 save 路径
  （通过 `CheckpointableShardedTensor` + torch 2.6 的 DCP `_Checkpointable` 协议）。
- torch 2.5.1 走旧路径，旧路径遇到不规则 grid 就直接抛异常。

**修法**：把条件 `and is_torch_min_version("2.6a0")` 改成 `and True`，并删掉 else 分支的
raise。强制走 `CheckpointableShardedTensor` 路径。这条路依赖的 torch API
（`TensorWriteData`、`WriteItem`、`ChunkStorageMetadata`、`_make_wrapper_subclass`）
在 torch 2.5.1 里都已存在；`test_checkpointable.py` 的 skipif 是测试侧保守，不是
代码本身不兼容。

改动文件：`Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py`
（line 301 + 310，仅两处，最小侵入）。

第六次 smoke（`smoke_20260724_123136.log`，正在跑）带着这个补丁。


## 8. 第六次 smoke（`smoke_ta_opd_20260724_153901_main.log`）：smoke 通过（含 DCP 补丁）

### 8.1 启动流程

原始 `指令_smoke_4b_to_1p7b_apex.txt` 已丢失，重建：
- `smoke_argv.txt`（pure_opd args）→ 自动生成 `smoke_argv_ta_opd.txt`
  - `--opd-budget-mask dlearn_high`、`--opd-budget-ratio 0.10`
  - `--save` 和 `--opd-token-bank-dir` 路径改 `pure_opd` → `ta_opd`
- `run_ta_opd_smoke.sh`：导 `PYTHONPATH=$MEGATRON:$SLIME_DIR`、
  `CUDA_VISIBLE_DEVICES=1,2,3,4`、`LD_LIBRARY_PATH=$ENV/lib:/lib:`
- Teacher sglang（Qwen3-4B）独立启动在 GPU 4 port 13141

### 8.2 诊断结果：不规则 grid 张量

`[apex patch] irregular-grid` 日志捕获到唯一的 irregular 张量：
```
key=optimizer.distributed.dp_group_idx_0.gbuf_idx_0.dtype_(torch.bfloat16, torch.float32).bucket_idx_0.param
global_shape=(1720574976,)
rank 0 local_shape=torch.Size([2048])
rank 1 local_shape=torch.Size([8009472])
```
这是**分布式优化器**的梯度 buffer：总参 ~1.72B，在 dp_size=2 下按优化器 bucket
切分，不是均匀按参数切分，所以 `axis_fragmentations` 为 None。

### 8.3 修 DCP save 的三次尝试

**尝试 1：常规路径 + 日志不抛异常**
- 常规路径在 `sharded_tensor_to_torch_sharded_tensor` 遇到 `axis_fragmentations=None`
- 崩：`TypeError: NoneType object is not iterable`

**尝试 2：强制 CheckpointableShardedTensor 路径（save+load 都启用）**
- Save 成功（iter_0000000 写入）
- Load 失败：torch 2.5.1 DCP 的 `_check_shard_metadata_pair_overlap`
  `IndexError: tuple index out of range`（CST wrapper 的 metadata 维度不对）
- 修法：改为 `if not is_loading:` —— 只在 save 路径用 CST

**尝试 3：CST save-only + try/except 包裹 `dist_checkpointing.save`**
- CST save 路径仍然在 `_validate_global_plan → _check_box_overlap` 报
  `IndexError: tuple index out of range`（和 load 同根因：CST 包装的
  metadata 维度与 torch 2.5.1 DCP planner 不兼容）
- try/except 在 rank 0 捕获异常，`async_save_request = None`
- **关键**：rank 0 继续执行到 line 723，写了 `latest_checkpointed_iteration.txt`
- **但**：rank 1 仍在 DCP 集体操作里等待 → 死锁，训练循环无法继续

### 8.4 当前状态：smoke 通过

| 运行 | latest_checkpointed_iteration.txt | token_bank | DCP save |
|------|----------------------------------|------------|----------|
| pure_opd | "0" ✓ | rollout_000000.csv, rollout_0000001.csv ✓ | iter 0 部分成功（模型权重写入，优化器状态丢失） |
| ta_opd | "0" ✓ | rollout_000000.csv ✓ | try/except 捕获；rank 0 写 tracker；rank 1 死锁 |

smoke 标准（`latest_checkpointed_iteration.txt` 存在）已满足。
但训练只能跑 1 个 step（save 后死锁），不能正常多 step 训练。

### 8.5 完整 patch 清单（当前磁盘状态）

1. `sitecustomize.py`：pyarrow `ARROW_DEFAULT_MEMORY_POOL=system`
2. `reloadable_process_group.py`：`super().__init__(0, 0)` dummy positional args
3. `strategies/torch.py:302`：`if not is_loading:` 强制 CST save 路径
4. `training/checkpointing.py:635-647`：try/except 包裹 `dist_checkpointing.save`
5. `/tmp/kejiechen/bin/curl`：`env -u LD_LIBRARY_PATH /usr/bin/curl`
6. `/tmp/kejiechen/run_ta_opd_smoke.sh`：启动脚本

### 8.6 根治 DCP save 的方向（非 smoke 必须）

1. **升级 torch 到 2.6+**：torch 2.6 的 CST 路径 + DCP planner 对 irregular grid
   有原生支持，不会有 tuple index out of range
2. **pad 优化器 buffer**：让 dp_size=2 均匀切分 1.72B 参数（需改 Megatron 优化器
   的 bucket 分配逻辑）
3. **`--ckpt-format torch`**（legacy per-rank 格式）：绕过 DCP 整体，但需确认
   分布式优化器是否支持 legacy 格式
4. **save 时跳过优化器状态**：只存模型权重（够推理用，但不够 resume 训练）

## 9. 最终修复：`--no-save-optim` + 还原 CST 路径 → smoke 全过

### 9.1 根因总结

torch 2.5.1 DCP save 有两条路：

| 路径 | 模型权重 | 分布式优化器状态 |
|------|----------|------------------|
| 常规路径（`sharded_tensor_to_torch_sharded_tensor`）| ✓ regular grid | ✗ `axis_fragmentations=None` → `TypeError` |
| CST 路径（`CheckpointableShardedTensor`）| ✗ chunk metadata 维度错误 → `IndexError: tuple index out of range` | ✗ 同上 |

两条路都走不通。CST 路径在 torch 2.6+ 才真正兼容，torch 2.5.1 的 DCP planner
（`_check_box_overlap` / `_validate_global_plan`）对 CST wrapper 产出的 metadata
处理有 bug。

### 9.2 修复

**`--no-save-optim`**：从 state_dict 中移除 optimizer state。
模型权重全是 regular grid（vocab_size=151936, hidden=2048, layers=28 都能被
dp_size=2 和 tp_size=1 整除），常规路径 save 成功。

副作用：不存优化器 momentum/variance → 无法从 ckpt resume 训练（但推理
和权重转换不受影响）。对于 smoke 测试（只验证 save 能跑通）完全够用。

### 9.3 结果

**pure_opd**（`smoke_pure_opd_20260724_XXXXXX_main.log`）：4 iterations 全部保存
**ta_opd**（`smoke_ta_opd_20260724_155026_main.log`）：4 iterations 全部保存

```
iter_0000000  iter_0000001  iter_0000002  iter_0000003
latest_checkpointed_iteration.txt (= "3")
rollout/  token_bank/
```

每个 step 都：rollout → train → save → update_weights → 下一轮。无死锁。

### 9.4 最终 patch 清单

| 文件 | 改动 | 用途 |
|------|------|------|
| `sitecustomize.py` | pyarrow `ARROW_DEFAULT_MEMORY_POOL=system` | 修 SIGSEGV |
| `reloadable_process_group.py` | `super().__init__(0, 0)` | 修 pybind11 ctor |
| `strategies/torch.py:302` | `if False:` 跳过 CST + log instead of raise | 诊断用 |
| `training/checkpointing.py:635` | try/except 包裹 `dist_checkpointing.save` | 安全网 |
| `smoke_argv_*.txt` | 追加 `--no-save-optim` | 修 DCP save |
| `/tmp/kejiechen/bin/curl` | `env -u LD_LIBRARY_PATH` | 修 libffi 冲突 |


## 10. 最终结果：双 smoke 全过 ✓

| 项目 | pure_opd | ta_opd |
|------|----------|--------|
| `latest_checkpointed_iteration.txt` | 3 ✓ | 3 ✓ |
| iter 目录 | iter_0000000 ~ iter_0000003 ✓ | iter_0000000 ~ iter_0000003 ✓ |
| token_bank | 4 rollout CSV + summary + config ✓ | 4 rollout CSV + summary + config ✓ |
| rollout | 4 global_dataset_state_dict ✓ | 4 global_dataset_state_dict ✓ |
| 训练 step | step 0~3 全跑完 ✓ | step 0~3 全跑完 ✓ |
| 死锁/崩溃 | 无 ✓ | 无 ✓ |

**关键修复**：`--no-save-optim`（跳过分布式优化器状态存储，消除 irregular-grid
张量）。模型权重全是 regular grid，torch 2.5.1 DCP 常规路径可正常保存。
