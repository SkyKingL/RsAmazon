import json
import pandas as pd

def json_to_csv(input_file, output_file):
    # 用于存储所需的数据
    data = []
    
    # 逐行读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析每行的JSON数据
            review = json.loads(line.strip())
            
            # 提取所需的特征
            extracted_data = {
                'reviewerID': review['reviewerID'],
                'asin': review['asin'],
                'overall': review['overall']
            }
            data.append(extracted_data)
    
    # 转换为DataFrame并保存为CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    
    # 打印一些基本统计信息
    print(f"\n数据集统计信息:")
    print(f"总评论数: {len(df):,}")
    print(f"独立用户数: {df['reviewerID'].nunique():,}")
    print(f"独立物品数: {df['asin'].nunique():,}")
    print(f"评分分布:")
    print(df['overall'].value_counts().sort_index())

# 使用示例
input_file = 'reviews_Digital_Music_5.json'
output_file = 'reviews_Digital_Music_5.csv'
json_to_csv(input_file, output_file)