---
name: xlsx
description: 使用openpyxl或pandas创建和操作Excel电子表格。
---
 
# XLSX Skill
 
当用户需要创建或操作 Excel 文件时，使用 openpyxl（精细控制）或 pandas（数据处理）。
 
## 安装
```bash
pip install openpyxl pandas
```
 
## 使用 openpyxl 创建样式化表格
```python
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
 
wb = Workbook()
ws = wb.active
ws.title = "数据表"
 
# 写入标题行（加粗 + 背景色）
headers = ["姓名", "部门", "薪资"]
for col, header in enumerate(headers, 1):
    cell = ws.cell(row=1, column=col, value=header)
    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = PatternFill(fill_type="solid", fgColor="4472C4")
    cell.alignment = Alignment(horizontal="center")
 
wb.save("output.xlsx")
```
 
## 使用 pandas 处理数据
```python
import pandas as pd
 
df = pd.read_csv("data.csv")
df_summary = df.groupby("部门")["薪资"].mean()
df_summary.to_excel("summary.xlsx", index=True)
```