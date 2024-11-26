# BUPT CS 研一 24 秋季推荐系统课程作业
# 2. Amazon数据集的TopK推荐任务

## 评分预测任务
https://github.com/SkyKingL/RsAmazonPredict

## Env
- Python version: 3.6.10
- Pytorch version: 1.10.1
```
conda create -n rsamazon python=3.6.10
```

下载对应版本torch的whl文件：https://download.pytorch.org/whl/cu113/torch-1.10.1%2Bcu113-cp36-cp36m-linux_x86_64.whl

再执行pytorch环境安装:
```
conda activate rsamazon
```
```
pip install torch-1.10.1+cu113-cp36-cp36m-linux_x86_64.whl
```
其余环境比如numpy,tqdm,就遇到情况了再安装就可以啦

## Dataset
在下面链接中下载好5-score的文件并解压：
http://jmcauley.ucsd.edu/data/amazon/

然后执行
```
python json2csv.py
```
得到对应数据集的csv文件

main.py代码中需要将csv文件的路径替换为对应的路径。再执行：
```
python main.py
```
