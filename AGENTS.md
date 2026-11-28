# cone-fdk 开发约定

- 主线只实现单 GPU、单网格 FDK。输入是已校正、有限的 FP32 线积分；入口检查几何与元素数，CLI 检查读写结果。
- Debug 和 Release 都检查 CUDA/cuFFT 操作。每次 kernel launch 查提交错误，在已有的同步拷贝处查执行错误，不逐 kernel 同步。
- 保护尾块和探测器外的零延拓；改变插值或运算顺序前，先跑独立 FP64 数值测试。
- README 性能只引用当前 CLI 的真实核桃测量；端到端墙钟、GPU 阶段时间与逻辑有效带宽分开报告。
