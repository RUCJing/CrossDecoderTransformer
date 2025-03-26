# 概述

本项目（CrossDecoderTransformer）基于 “基于多模态数据的在线问诊患者满意度预测研究” 展开。核心模型架构在model.py中定义，通过model_train.py进行训练。训练好的模型保存在models目录，方便后续使用model_eval.py对其在测试数据上的性能进行评估。

# 项目结构

## 文件夹
* dataset：该目录用于存放项目所需的数据集（https://huggingface.co/datasets/FireflyLiu/C_Online_Diagnose/blob/main/README.md）。
* font：存储与字体相关的文件，在可视化中使用。
* image：主要存放图像文件，这些图像是在模型评估过程中产生的可视化图像等。
* models：用于保存训练好的模型文件。在项目中，训练完成的模型会被存储在此目录下，方便后续的模型加载和评估等操作。

## 文件
* model_eval.py：模型评估脚本，加载训练好的模型并评估，绘制混淆矩阵和ROC曲线。
* model_train.py：模型训练脚本，保存训练好的模型到 models 目录。
* model.py：CrossDecoderTransformer 模型的具体实现。
* params.py：存放项目的各种参数设置，方便统一管理和修改参数。
* text_featuring.py：特征处理，并生成适用于pytorch训练的dataset。
* preprocessing.py：对于中文文本，进行字符切分，并保存中文字符与token、标签与数值的对应关系；对于英文文本，可使用其他定制好的分词器。


# 训练流程

请首先执行
```
pip install -r requirements.txt
```

中文文本训练前进行：
```
python preprocessing.py
```

模型训练代码：
```
python model_train.py
```

模型评估代码：
```
python model_eval.py
```

