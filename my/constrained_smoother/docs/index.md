# Constrained Smoother 文档

本站点记录 `my/constrained_smoother` 独立实验包的运动学后端设计、构建方式和稳定错误语义。

## 文档入口

<div class="grid cards" markdown>

- [包使用指南](package-guide.md)

  ---

  整体能力、公开接口、参数说明、失败传播路径和构建方式。

- [运动学平滑器设计](KINEMATIC_SMOOTHER_DESIGN.md)

  ---

  状态展开、问题构建、残差定义、边界约束和后验校验的详细设计。

- [错误码参考](error-codes.md)

  ---

  稳定错误码、失败原因枚举和各层错误表面映射。

</div>

## 建议阅读顺序

1. 第一次接触：先看 [包使用指南](package-guide.md)。
2. 准备改动求解器实现：再看 [运动学平滑器设计](KINEMATIC_SMOOTHER_DESIGN.md)。
3. 准备对齐异常和失败返回：最后看 [错误码参考](error-codes.md)。

## 本地预览

```bash
./run_docs.sh
```
