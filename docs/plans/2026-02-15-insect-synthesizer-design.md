# Insect Synthesizer 设计文档

**目标：** 构建合成数据集工具，用干净背景的昆虫图片与复杂背景合成，用于训练昆虫分类/检测模型

**架构：** 两个独立脚本（segment.py 扣图 + synthesize.py 合成），支持 CPU/GPU/MPS 推理，后续迭代添加更多物理控制

**技术栈：** Python, PyTorch, SAM3 (CPU/GPU/MPS 自动适配), OpenCV, scikit-image, tqdm (进度条), Pillow (PNG/JPG 压缩)

---

## 1. 扣图脚本 (`segment.py`)

### 功能
- 加载 SAM3 模型（自动适配 CPU/GPU/MPS）
- 自动分割昆虫或其它目标类型（支持文本提示）
- 多种分割方法可选（SAM3/Otsu/GrabCut/SAM3-fill）
- 可选自动补洞（多种策略）
- 保存提取的干净昆虫图片（带 alpha 通道或仅 bbox）
- 保存元数据（COCO 格式标注、掩码面积、外接框、原图路径等）
- 记录 log.txt 日志（包含 tqdm 进度信息）

### 命令行
```bash
python segment.py \
  --input_dir images/clean_insects/ \
  --out_dir outputs/insects_clean/ \
  --sam3-checkpoint models/sam3_hq_vit_h.pt \
  --segmentation-method sam3 \
  --device auto \
  --hint "insect" \
  --repair-strategy sam3-fill \
  --out-image-format png \
  --threads 12
```

### 参数说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_dir` | 输入图片目录 | 必填 |
| `--out_dir` | 输出目录 | 必填 |
| `--sam3-checkpoint` | SAM3 模型权重路径 | 必填 |
| `--device` | 推理设备 (auto/cpu/cuda/mps) | auto |
| `--hint` | 分割提示文本 | "insect" |
| `--input_dir` | 输入图片目录 | 必填 |
| `--out_dir` | 输出目录 | 必填 |
| `--segmentation-method` | 分割方法 (sam3/sam3-bbox/otsu/grabcut) | sam3 |
| `--sam3-checkpoint` | SAM3 模型权重路径 | 必填 (仅 sam3/sam3-bbox) |
| `--device` | 推理设备 (auto/cpu/cuda/mps) | auto |
| `--confidence-threshold` | 置信度阈值 (0.0-1.0) | 0.0 |
| `--repair-strategy` | 修复策略 (opencv/sam3-fill/black-mask/lama) | None |
| `--padding-ratio` | 边框填充比例 (0.0-0.5) | 0.0 |
| `--out-image-format` | 输出格式 (png/jpg) | png |
| `--threads` | 并行线程数 | 8 |

### 核心逻辑
```python
# 1. 加载 SAM3 模型（自动适配 device）
model = build_sam3_checkpoint(
    checkpoint=sam3_checkpoint,
    model_type="vit_h",
    device=device,
    dtype=torch.float32,
    video_mode=False  # 关闭视频分割
)

# 2. 对每张图片并行处理
def process_image(img_path, repair_strategy):
    img = load_image(img_path)
    
    # SAM3 分割（支持文本提示）
    masks = sam3.predict(img, text_prompt=hint)
    
    # 处理多只昆虫情况
    if len(masks) > 1:
        # 多只昆虫，分别保存
        for i, mask in enumerate(masks):
            # 提取干净昆虫（带 alpha 通道）
            clean_insect = apply_mask_with_alpha(img, mask)
            
            # 获取原图文件名（不含扩展名）
            base_name = Path(img_path).stem
            
            # 保存多昆虫变体：原文件名_001.jpg, _002.jpg
            output_path = f"{base_name}_{i+1:03d}.png"
            save_clean_insect(clean_insect, output_path)
            
            # 计算元数据
            mask_area = count_nonzero(mask)
            bbox = get_bounding_box(mask)
            
            # 保存 COCO 格式标注
            save_metadata({
                'original_image': str(img_path),
                'output_file': output_path,
                'mask_area': mask_area,
                'bbox': bbox,  # [x, y, width, height]
                'segmentation': mask_to_polygon(mask)
            })
    else:
        # 单只昆虫，直接保存
        insect_mask = masks[0]
        clean_insect = apply_mask_with_alpha(img, insect_mask)
        
# 补洞（可选，仅当指定策略时）
            if repair_strategy:
                if strategy == "opencv":
                    # 多孔补洞（支持多 mask）
                    if len(masks) > 1:
                        combined_mask = combine_masks(masks)
                        inpaint_mask = 255 - combined_mask
                    else:
                        inpaint_mask = 255 - insect_mask
                    result = cv2.inpaint(img, inpaint_mask, 3, cv2.INPAINT_NS)
                elif strategy == "sam3-fill":
                    filled = fill_holes_with_texture(img, insect_mask)
                elif strategy == "black-mask":
                    filled = fill_holes_with_black(img, insect_mask)
                elif strategy == "lama":
                    filled = fill_holes_with_lama(img, insect_mask)
                # 保存修复后的图片
                save_filled_image(result, mask)
        
        # 总是保存干净提取的昆虫
        save_clean_insect(clean_insect, mask)
```

### 输出
- **干净昆虫图片**（PNG/JPG，带 alpha 通道或仅 bbox）- 保存到 `cleaned_images/`
  - 单昆虫：`filename.png` 或 `filename.jpg`
  - 多昆虫：`filename_1.png`, `filename_2.png`, ... (无前导零)
- **元数据文件**（JSON）- 包含 COCO 格式标注、掩码面积、外接框、原图路径
- **修复后图片**（仅当指定 --repair-strategy 时）- 保存到 `repaired_images/`，每张完整原图的修复版本
- **log.txt** - 记录完整命令和日志信息（包含 tqdm 进度）

### 修复策略说明
- **opencv**: 对整张原图进行 OpenCV Inpainting 修复，输出到 `repaired_images/` (PNG/JPG 格式，与输出格式一致)，多个目标合并为一张修复图。`cleaned_images/` 仍保留原分割结果（带 alpha）
- **sam3-fill**: 使用 SAM3 增强的修复策略（基于 OpenCV + SAM3 引导），输出到 `repaired_images/`。`cleaned_images/` 仍保留原分割结果（带 alpha）
- **black-mask**: 纯黑色填充 [0,0,0]，用于后续合成时避开该区域。输出到 `repaired_images/`。`cleaned_images/` 仍保留原分割结果（带 alpha）
- **LaMa**: 基于傅里叶卷积的大掩码修补（WACV 2022），专为大掩码设计，支持高分辨率。需单独下载模型文件（~400MB for big-lama）。输出到 `repaired_images/`。`cleaned_images/` 仍保留原分割结果（带 alpha）

### 分割方法说明
- **sam3**: SAM3 分割 + 透明 alpha 通道（PNG 保留 alpha，JPG 自动转 RGB）
- **sam3-bbox**: SAM3 分割 + 裁剪外接框，无透明通道
- **otsu**: Otsu 二值化分割，无需 SAM3 checkpoint
- **grabcut**: GrabCut 算法分割，无需 SAM3 checkpoint

---

## 2. 合成脚本 (`synthesize.py`)

### 功能
- 从目录随机选择背景图
- 控制尺度（基于面积占比）
- 控制位置（随机但避免边缘）
- 光照/颜色匹配（LAB 空间直方图匹配）

### 命令行
```bash
python synthesize.py \
  --target_dir outputs/insects_clean/ \
  --background_dir images/backgrounds/ \
  --out_dir outputs/synthesized/ \
  --num-syntheses 100 \
  --scale-min 0.10 \
  --scale-max 0.50 \
  --color-match-strength 0.5 \
  --out-image-format png \
  --threads 12
```

### 核心逻辑
```python
# 1. 加载所有干净昆虫和背景
target_images = load_images(target_dir)  # 带 alpha 通道
backgrounds = load_images(backgrounds_dir)

# 2. 合成循环：每个干净昆虫合成 num_syntheses 次
def synthesize_with_background(insect, background, scale, x, y, color_match_strength):
    # 尺度控制（基于面积占比）
    target_area = background.shape[0] * background.shape[1] * scale
    scale_factor = math.sqrt(target_area / insect.mask_area)
    resized_target = resize(insect, scale_factor)
    
    # 位置控制：优先纹理密度高的区域
    x, y = random_position_with_texture_constraint(background, resized_target)
    
    # 合成
    blended = paste_with_alpha(background, resized_target, x, y)
    
    # 光照匹配（可选，仅对昆虫区域，非整图）
    if color_match_strength > 0:
        blended = match_lab_histograms_per_region(
            blended, background, resized_target,
            strength=color_match_strength  # 0.0-1.0
        )
    
    return blended

# 并行合成，带进度条
results = parallel_process(
    targets=target_images,
    num_syntheses=num_syntheses,
    progress_bar=True,  # 显示进度条
    num_workers=threads
)

for insect, bg, scale, x, y in results:
    blended = synthesize_with_background(insect, bg, scale, x, y, color_match_strength)
    save(blended, format='png', quality=90)  # PNG 中等压缩
```

### 输出
- 合成图片（PNG/JPG）
- COCO 格式标注文件（bbox + segmentation）
- log.txt - 记录完整命令和输出

---

## 3. 更新日志

### 2026-02-15
- ✅ 修复分割方法：sam3, sam3-bbox, otsu, grabcut 四种
- ✅ 修复 RGBA as JPEG 报错 - 保存 JPEG 时自动转换为 RGB
- ✅ 修复 repaired_images 文件夹 - 现在可以正常创建和保存
- ✅ 修复多个目标修复逻辑：合并所有掩码后修复整张图，仅输出一张修复图
- ✅ 修复 GrabCut 断言错误：使用 RECT 初始化方式
- ✅ Otsu/GrabCut 方法无需 SAM3 checkpoint 参数
- ✅ 新增 sam3-fill 修复策略
- ✅ 修复 sam3 + repair 策略问题：现在 sam3 方法即使启用 repair 策略也**仍然**保存带 alpha 的结果到 cleaned_images/，同时额外修复整张图到 repaired_images/
- ✅ 重新排序 CLI 参数：hint, input_dir, out_dir, segmentation-method, sam3-checkpoint, ...
- ✅ 修复文件命名：`filename_026.png` → `filename_26.png` (去除前导零)
- ✅ 修复输出格式：`--out-image-format jpg` 现在正确保存为 JPG
- ✅ 修复文件命名：`filename_026.png` → `filename_26.png` (去除前导零)
- ✅ 修复输出格式：`--out-image-format jpg` 现在正确保存为 JPG

---

## 3. 后续迭代方向

### Phase 2
- 多种补洞策略（OpenCV + SAM3-fill + black-mask + LaMa）
- 景深模糊（bokeh based on depth map）
- 更精细的光照控制（阴影、高光）
- 接触/依附关系（检测昆虫足部接触点）

### Phase 3
- 批量处理脚本（自动化整个流程）
- 质量验证脚本（可视化检查）
- LaMa 高级修复选项（支持超大掩码）
- Web UI（可选）

---

## 4. 文件结构

```
entomokit/
├── segment.py      # 扣图脚本
├── synthesize.py   # 合成脚本
├── configs/
│   └── default.yaml        # 默认配置
├── models/                 # SAM3 模型（用户提供）
│   └── sam3_hq_vit_h.pt
├── data/
│   ├── clean_insects/      # 输入：干净昆虫图
│   ├── backgrounds/        # 输入：背景图
│   ├── insects_clean/      # 输出：干净提取的昆虫（带 alpha）
│   ├── insects_repaired/   # 输出：修复后的图片（可选）
│   └── synthesized/        # 输出：合成图
└── docs/
    └── plans/              # 设计文档
```

---

## 5. 关键技术细节

### SAM3 Device Auto-Adaptation
```python
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

# 使用时
device = get_device() if args.device == "auto" else args.device

# SAM3 视频分割功能关闭（减少内存占用）
model = build_sam3_checkpoint(
    ...,
    video_mode=False  # 关闭视频分割
)
```

### 补洞策略（可选）

**策略 1：OpenCV 修复**
```python
# 基于 Navier-Stokes 方程的图像修复
inpaint_mask = 255 - segmentation_mask
result = cv2.inpaint(img, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
```
优点：速度快，无需额外模型  
缺点：复杂纹理可能不自然

**策略 2：SAM3 分割 + 纹理填充**
```python
def fill_holes_with_texture(img, mask):
    # 1. 用 mask 提取昆虫
    insect = apply_mask(img, mask)
    
    # 2. 对空洞区域做纹理合成
    filled = texture_synthesis(img, mask)
    
    return filled
```
优点：精度高，纹理自然  
缺点：速度慢，需要额外处理

---

## 6. 实现优先级

1. **Phase 1 (Minimal Version)**
    - SAM3 自动设备适配（CPU/GPU/MPS）
    - 文本提示分割（--hint）
    - OpenCV 补洞策略
    - 基础合成（尺度+位置）
    - COCO 格式标注
    - 位置约束（纹理密度高的区域）
    - 分区域颜色匹配（--color-match-strength）

2. **Phase 2 (迭代)**
   - SAM3-fill 补洞策略
   - black-mask 补洞策略（纯黑色填充，用于后续合成避开该区域）
   - LaMa 补洞策略（大掩码修补，WACV 2022）
   - 光照匹配（LAB 直方图）
   - 更多合成策略

3. **Phase 3 (后续)**
   - 景深模糊
   - 接触/依附关系
   - 批量处理脚本

---

## 7. 测试策略

### 扣图脚本测试
```python
def test_sam3_device_auto():
    model = load_sam3_auto("models/sam3.pt")
    assert model.device in ["cpu", "cuda", "mps"]

def test_hints():
    model = load_sam3_auto("models/sam3.pt")
    img = load_image("test_data/insect.png")
    mask = model.predict(img, text_prompt="insect")
    assert mask.shape == img.shape[:2]

def test_repair_optional():
    img = load_image("test_data/insect.png")
    mask = load_mask("test_data/insect_mask.png")
    
    # 不指定 repair-strategy，仅提取
    clean = extract_clean_insect(img, mask)
    assert clean.shape[2] == 4  # RGBA
    
    # 指定 repair-strategy，修复+提取
    result = opencv_repair(img, mask)
    assert result.shape == img.shape
```

### 合成脚本测试
```python
def test_scale_control():
    background = load_image("test_data/background.jpg")
    insect = load_image("test_data/insect.png")
    
    result = synthesize(background, insect, scale=0.10)
    
    insect_area = count_nonzero(result.alpha)
    total_area = background.shape[0] * background.shape[1]
    actual_ratio = insect_area / total_area
    
    assert 0.09 <= actual_ratio <= 0.11

def test_color_matching():
    bg = load_image("test_data/bg.jpg")
    insect = load_image("test_data/insect.png")
    
    result = synthesize(bg, insect, match_colors=True)
    
    assert lab_distance(result, bg) < threshold
```

---

## 8. 性能预期

### SAM3 推理（各设备）
- **CPU**: 单张 5-10 秒，1000 张约 2-3 小时
- **MPS**: 单张 2-5 秒，1000 张约 1-2 小时
- **CUDA**: 单张 0.5-2 秒，1000 张约 10-30 分钟

### 合成速度
- 单张合成：0.1-0.5 秒
- 1000 张合成（100 insects × 10 syntheses，12线程）：约 1-3 分钟

### 优化建议
- 并行处理（--threads 12）
- 批量处理（batch_size=4~8）
- PNG 中等压缩（quality=90）
- 视频分割模式关闭（减少内存）
- 自动设备选择

---

## 9. 用户手册（简化）

### 安装依赖
```bash
pip install torch torchvision opencv-python scikit-image numpy tqdm pillow
```

### 准备数据
```
data/
├── clean_insects/      # 放干净的昆虫图片（JPG/PNG）
└── backgrounds/        # 放背景图片
```

### 运行扣图
```bash
# 最小配置（仅提取，不修复）
python segment.py \
  --input_dir data/clean_insects/ \
  --out_dir data/insects_clean/ \
  --sam3-checkpoint models/sam3_hq_vit_h.pt

# 带修复的完整流程
python segment.py \
  --input_dir data/clean_insects/ \
  --out_dir data/insects_clean/ \
  --sam3-checkpoint models/sam3_hq_vit_h.pt \
  --repair-strategy opencv \
  --device auto \
  --hint "insect" \
  --out-image-format png
```

### 运行合成
```bash
python synthesize.py \
  --target_dir data/insects_clean/ \
  --background_dir data/backgrounds/ \
  --out_dir data/synthesized/ \
  --num-syntheses 100 \
  --scale-min 0.10 \
  --scale-max 0.50 \
  --color-match-strength 0.5 \
  --out-image-format png \
  --threads 12
```

### 查看结果
```bash
# 打开 COCO 标注文件
cat data/synthesized/annotations.json
```

---

## 10. 注意事项

1. **SAM3 模型文件**：用户需要自行下载（如 sam3_hq_vit_h.pt），脚本不包含模型
2. **设备自动适配**：默认 auto 模式自动选择 cuda > mps > cpu
3. **内存占用**：
   - SAM3 CPU 模式约 4-6GB RAM
   - SAM3 CUDA 模式约 2-4GB VRAM
   - 视频分割模式关闭可显著减少内存
4. **背景选择**：建议背景图中不含昆虫，否则需要手动补洞
5. **num_syntheses**: 每个干净昆虫合成的数量，默认 100，可根据需求调整
6. **log.txt**: 每次执行自动生成日志文件，记录完整命令和输出
7. **位置约束算法**: Phase 1 采用简单纹理密度检测（Sobel 边缘密度），可实现但精度有限， Phase 2 可升级为深度学习显著性检测

---

## 11. FAQ

**Q: SAM3 模型在哪下载？**  
A: 可以从 HuggingFace 下载，如 `https://huggingface.co/lmzq/sam3`

**Q: CPU 推理太慢怎么办？**  
A: 建议使用 GPU/MPS，或调整 num_syntheses 减少合成数量，或增加 --threads 并行数

**Q: 日志文件在哪？**  
A: 每次执行会在输出目录生成 log.txt，记录完整命令和输出

**Q: 如何调整昆虫大小范围？**  
A: 使用 --scale-min 和 --scale-max，推荐近距离特写：0.10-0.50

**Q: 输出的 COCO 格式可以用于训练吗？**  
A: 可以，标准 COCO 格式，兼容 PyTorch DataLoader

**Q: num_syntheses 100 意味着什么？**  
A: 每个干净昆虫会被合成 100 次，每次随机选择背景、位置、尺度，生成 100 个不同变体

**Q: 如何只提取不修复？**  
A: 不指定 --repair-strategy 参数即可，只会保存干净提取的昆虫图片

**Q: black-mask 修复策略有什么用？**  
A: 使用纯黑色 [0,0,0] 填充掩码区域，用于后续合成时作为遮罩避开该区域

**Q: LaMa 修复策略是什么？需要额外配置吗？**  
A: LaMa (Large Mask Inpainting, WACV 2022) 是基于傅里叶卷积的大掩码修补算法，支持高分辨率。需单独下载模型文件（~400MB for big-lama）或使用ISAT包。CPU 推理约 4-6GB RAM

**Q: 合成时如何控制颜色匹配强度？**  
A: 使用 --color-match-strength 0.0-1.0，0=不匹配，1=完全匹配，默认 0.5，仅对昆虫区域匹配，保留关键纹理

**Q: 多只昆虫怎么办？**  
A: 自动检测多只昆虫，分别保存为 filename_001.png, filename_002.png, ... 对应元数据也分别记录

**Q: 如何控制合成并行数？**  
A: 使用 --threads 12 控制并行线程数，提升合成速度

**Q: 合成时昆虫会贴在什么位置？**  
A: 优先选择背景纹理密度高的区域（如叶片/树干），避免天空/模糊区域

---

**版本：** v0.5 (2026-02-16 - Added black-mask and LaMa repair strategies)  
**最后更新：** 2026-02-16

---

## 最近更新 (2026-02-16)

### 新增功能

1. **新增 repair-strategy 选项**
   - `black-mask`: 纯黑色填充 [0,0,0]，用于后续合成时避开该区域
   - `LaMa`: 基于傅里叶卷积的大掩码修补（WACV 2022），支持高分辨率，需下载模型文件 (~400MB)

2. **输出格式修复**
   - `--out-image-format jpg` 现在正确保存为 `.jpg` 格式
   - 强制文件扩展名与格式参数匹配

3. **修复策略实现**
   - `--repair-strategy opencv` 现在会生成修复后的图片
   - 保存到 `output/repaired_images/` 子目录

4. **置信度过滤**
   - `--confidence-threshold` 正确过滤低质量分割结果
   - 过滤在处理循环之前完成

5. **文件名编号**
   - 输出格式：`XX_1.jpg` (不是 `XX_001.jpg`)
   - 移除前导零

6. **边界框填充**
   - `--padding-ratio` 参数控制边界框周围填充
   - 默认 0.0，示例：0.1 = 10% 填充

7. **分割掩码选项**
   - `--use-segmentation-mask` 标志保存带透明背景的分割图
   - 默认保存边界框裁剪内容

8. **日志格式**
   - `--disable-tqdm` 移除 log.txt 中的进度条
   - 仅记录信息性消息

### 更新的命令行参数

```bash
segment.py:
  --confidence-threshold FLOAT    # 分割置信度阈值
  --padding-ratio FLOAT           # 边界框填充比例
  --use-segmentation-mask         # 使用分割掩码（透明背景）
  --repair-output-format FORMAT   # 修复后图片格式
  --disable-tqdm                  # 禁用进度条（日志中）

synthesize.py:
  [保持不变]
```

### 目录结构

```
output/
├── cleaned_images/       # 分割后的昆虫图片
├── repaired_images/      # 修复后的图片 (可选)
└── annotations.json      # COCO 格式标注
```

### 待实现功能

1. **Otsu 分割方法**
   - 简单的阈值分割
   - 参考: `detect_bounding_box.py`

2. **GrabCut 分割方法**
   - OpenCV GrabCut 算法
   - 需要用户交互或自动初始化

3. **LaMa 修复策略**
    - 基于傅里叶卷积的大掩码修补（WACV 2022）
    - 支持高分辨率，需额外下载模型文件 (~400MB)
4. **black-mask 修复策略**
    - 纯黑色填充 [0,0,0]，用于后续合成时避开该区域
    - 零开销，完全可控
