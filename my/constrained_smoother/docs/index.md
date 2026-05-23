# Constrained Smoother 文档

这个站点现在只记录 `my/constrained_smoother` 的运动学后端、构建方式和稳定错误语义。

## 阅读入口

<div class="grid cards" markdown>

- [Package Guide](package-guide.md)

  ---

  先看整体能力、公开接口、失败传播路径和构建方式。

- [Kinematic Smoother Design](KINEMATIC_SMOOTHER_DESIGN.md)

  ---

  理解状态展开、问题构建、边界约束和后验校验。

- [Error Codes](error-codes.md)

  ---

  查询稳定错误码和失败语义。

</div>

## 建议阅读顺序

1. 第一次接触这个独立版包：先看 [Package Guide](package-guide.md)。
2. 准备改动求解器实现：再看 [Kinematic Smoother Design](KINEMATIC_SMOOTHER_DESIGN.md)。
3. 准备对齐异常和失败返回：最后看 [Error Codes](error-codes.md)。

## 本地预览

```bash
./run_docs.sh
```