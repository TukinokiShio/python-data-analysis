# Python 数据分析课程作业

本仓库用于保存课程作业、分析代码、数据说明和可复现结果。作业按周整理：

- [第二周作业](第二周作业/)：题目、作答、Carseats 数据、分析脚本及模型结果。
- [第三四周作业](第三四周作业/)：正则化回归与 SVD 课件，以及本次作业产物。
- [04 作业](04作业/)：基于 UCI Spambase 数据集的 SVM 垃圾邮件分类。

项目使用 Python 3.12。创建虚拟环境后，在仓库根目录安装共享依赖：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

运行第二周分析：

```powershell
.\.venv\Scripts\python.exe 第二周作业\carseats_analysis.py
```

运行 04 作业：

```powershell
.\.venv\Scripts\python.exe 04作业\svm_spambase.py
```


