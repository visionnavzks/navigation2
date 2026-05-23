# Constrained Smoother 文档

这个站点把 `my/constrained_smoother` 当前最重要的设计文档、接口约定和参考资料集中到一个 Material for MkDocs 站点里，方便按主题浏览，而不是在仓库里手动跳转 Markdown 文件。

## 阅读入口

<div class="grid cards" markdown>

- [Package Guide](package-guide.md)

  ---

  先看整体能力、主要接口约定、失败传播路径和构建方式。

- [Geometric Smoother I/O](SMOOTHER_INPUT_OUTPUT.md)

  ---

  先确认输入输出语义、失败返回形式和对外契约。

- [Geometric Smoother Design](SMOOTHER_DESIGN.md)

  ---

  理解几何版 smoother 的分层、残差连接、cusp 处理和后验校验。

- [Kinematic Smoother Design](KINEMATIC_SMOOTHER_DESIGN.md)

  ---

  理解运动学版 smoother 的状态展开、问题构建和边界约束。

- [MPC Cost Notes](mpc_cost.md)

  ---

  对照两类 smoother 的代价、约束和未来演进方向。

- [Error Codes](error-codes.md)

  ---

  查询稳定错误码和失败语义。

</div>

## 建议阅读顺序

1. 如果你是第一次接触这个独立版包，先看 [Package Guide](package-guide.md) 和 [Geometric Smoother I/O](SMOOTHER_INPUT_OUTPUT.md)。
2. 如果你准备修改几何版实现，再看 [Geometric Smoother Design](SMOOTHER_DESIGN.md)。
3. 如果你准备修改运动学版实现，再看 [Kinematic Smoother Design](KINEMATIC_SMOOTHER_DESIGN.md)。
4. 如果你在对齐约束表达或规划后续 MPC 能力，再看 [MPC Cost Notes](mpc_cost.md)。

## 本地预览

在 `my/constrained_smoother` 目录下运行：

```bash
./run_docs.sh
```

默认监听 `127.0.0.1:8000`。如果需要改地址或端口，可以设置 `CS_DOCS_HOST` / `CS_DOCS_PORT`。

构建静态站点：

```bash
uvx --with mkdocs-material mkdocs build -f mkdocs.yml
```