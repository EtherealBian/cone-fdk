# AGENTS.md — cone-fdk 代码气质与防御性编程规范

本文件约束所有对 src/、bench/、examples/ 的代码改动。
experimental/ 下的验收工具不受此约束。

## 核心原则

1. 校验在边界，计算在 kernel，错误检查在同步点，验证在工具里。
2. hot path 上只允许出现算法必需的分支。任何“为了显得严谨”的检查都是噪音。
3. CUDA 项目里 segfault 是可接受的失败模式（可用 compute-sanitizer / cuda-gdb
   定位）。不要为了“优雅报错”在 hot path 堆砌防御代码。
4. 删除一个检查的判断标准：如果删掉它，正常输入下行为完全不变，异常输入下
   只是 segfault 而不是“优雅报错”——那就删。

## 必须删除的防御性代码（禁止出现在 src/ 和 bench/）

- 每个函数入口的 `if (ptr == nullptr) return ERR;`。指针来源由调用方保证，
  只在 API 边界（fdk_reconstruct 入口、文件读入、cudaMalloc 返回值）检查一次。
- 每个 CUDA API 调用后跟 `printf + exit(1)` 式错误处理。改为统一的 CUDA_CHECK
  宏（debug build 生效，release build 编译掉）+ 同步点 CUDA_CHECK_LAST()。
- kernel 内对 blockIdx / gridDim / 编译期常量的范围校验。
- 对内部函数参数做调用方已保证的范围检查。
- 用 try/catch 包裹 CUDA Runtime API 调用（它不抛异常）。
- 为从未被拷贝的结构体手写拷贝/移动构造（遵循 Rule of Zero）。
- 在 src/ 的重建流程中嵌入 checkpoint / resume / bitwise-verify / snapshot /
  hash 记录逻辑。这些属于 experimental/。

## 必须保留但必须收敛的（CUDA 特有）

- kernel launch 后的错误检查：CUDA 错误是异步的，必须查，但只在 debug build
  的 CUDA_CHECK_LAST() 中查，release build 编译掉。
- cudaMalloc / cudaMallocAsync 返回值检查：OOM 是真实场景，检查一次即可。
- host 端输入一致性校验：投影数 × 探测器尺寸 × 角度数匹配、体素尺寸 > 0 等，
  全部收敛到 validate_inputs()，在 fdk_reconstruct() 入口调用一次。
  内部函数不再重复校验。
- __syncthreads() 前后的边界处理：属于算法正确性，保留。

## 禁止新增的代码

- 新的“为了未来可能用到”的参数、开关、分支。
- 新的日志、进度打印、中间状态落盘（除非在 experimental/）。
- 新的散落 if (err != cudaSuccess) 检查，必须走 CUDA_CHECK 宏。
- 为单个调用点抽象出的“通用工具函数”。

## kernel 代码规范

- kernel 签名只包含算法必需参数：输入指针、输出指针、维度、投影数。
  禁止传入 config 结构体、checkpoint 指针、验证开关。
- 每个 kernel 上方必须有结构化注释块，模板：

  /**
   * <kernel_name> — <一句话职责>
   *
   * Thread mapping: <线程映射方式与理由>
   * Memory: <全局/共享/常量/纹理内存的使用与理由>
   * Occupancy: <实测 occupancy、寄存器数、瓶颈；无法实测填 <待实测>>
   * Known bottleneck: <已知瓶颈与对应 optimization-log 条目>
   */

- kernel 内唯一允许的边界检查是 `if (idx >= N) return;` 形式。
- __restrict__ 必须用在所有不 alias 的指针参数上。

## host 代码规范

- 重建 API 签名必须干净：
  void fdk_reconstruct(const Projections& proj, const Geometry& geo, Volume& out);
  禁止把 checkpoint / verify / snapshot 相关参数塞进主 API。
- 校验集中在 validate_inputs()，错误检查集中在同步点的 CUDA_CHECK_LAST()。
- 禁止在 hot path 上打印日志或写中间文件。

## 代码审查自问清单（每次提交前）

1. 这段检查删掉，正常路径会变吗？不会 → 删。
2. 这个参数/开关/分支，今天有调用方吗？没有 → 删。
3. 这个错误处理，是在边界还是在 hot path？hot path → 收敛到同步点。
4. 这个 kernel 参数，是算法必需还是验收需要？验收需要 → 移出。
5. 这段代码让 src/ 更接近“最简 CUDA 实现”还是更远离？远离 → 重写。
