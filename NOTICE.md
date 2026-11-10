# 第三方组件声明

本仓库的**源代码**以 MIT 许可发布（见 `LICENSE`）。
以下第三方组件**不随本仓库分发**，运行管线时需自行获取并遵守其各自许可：

| 组件 | 提供方 | 许可 | 用途 | 获取方式 |
| :--- | :--- | :--- | :--- | :--- |
| **MHR** (Momentum Human Rig) | Meta | Apache-2.0 | 127 关节骨骼、LOD0 细分网格、姿态修正 | 官方仓库 + 官方 `assets` 包（`lod0.fbx`、`compact_v6_1.model`、`corrective_blendshapes_lod*.npz`） |
| **SAM 3D Body (DINOv3)** | Meta | 见官方模型卡 | 单图人体姿态与体态先验 | 权重受各自许可约束，**不得再分发** |
| **MediaPipe Selfie Segmenter** | Google | Apache-2.0 | 人像前景分割 | `.tflite` 模型文件 |
| **GPEN 人像增强** | ModelScope (`damo/cv_gpen_image-portrait-enhancement`) | 见模型卡 | 输入照片 2× 面部增强与超分 | ModelScope 模型库 |
| **Blender** | Blender Foundation | GPL | 物理布料模拟与 Cycles 渲染 | blender.org |

被 `.gitignore` 排除的目录即对应上述资产：`envs/`、`mamba/`、`models/`、`repos/`。

## 输入素材与肖像

`work/**/scene/*.png`（输入实拍照片）及其派生的 mask、`fit/`、`kpts/` 均**未纳入仓库**。
`work/q004-base/renders/` 下的渲染图由真实人物照片重建而来，包含该人物的面部形象；
仓库维护者已就公开该形象取得授权。若你 fork 本仓库并替换为自己的输入照片，
请自行确保对所用照片及生成结果拥有合法权利。
