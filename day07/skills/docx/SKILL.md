---
name: docx
description: 使用专业格式创建和操作Word文档（.docx文件）。
---
 
# DOCX Skill
 
当用户需要创建或操作 Word 文档时，使用 python-docx 库。
 
## 安装
```bash
pip install python-docx
```
 
## 核心用法
 
### 创建文档

```python
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
 
doc = Document()
 
# 添加标题
doc.add_heading('文档标题', level=0)
 
# 添加段落
p = doc.add_paragraph('正文内容')
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
 
# 设置字体
run = p.runs[0]
run.font.size = Pt(12)
run.font.name = '宋体'
 
# 保存
doc.save('output.docx')
```
 
## 最佳实践
- 始终使用样式系统，不要手动设置每处格式
- 表格数据使用 add_table() 而非手动排版
- 图片使用 add_picture() 并指定宽度
- 输出文件保存到 /file/outputs/