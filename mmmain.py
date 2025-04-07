import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

class DataAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_data(self):
        print(f"\n{'=' * 50}")
        print(f"开始加载 30G Parquet数据集...")

        start_time = datetime.now()

        #cols = ['user_name', 'chinese_name', 'age', 'income', 'credit_score', 'gender', 'country']
        parquet_pattern = os.path.join(self.data_dir, "*.parquet")
        df = dd.read_parquet(parquet_pattern, chunksize="300MB")

        load_time = datetime.now() - start_time
        print(f"数据加载完成，耗时: {load_time}")
        #print(res_chunk.shape)
        print("字段如下：", df.columns)

        return df

    #抽样
    def sample_data(self, df):
        print(f"\n{'=' * 50}")
        print(f"开始抽样")
        start_time = datetime.now()

        #total_size_bytes = 30 * 1024 ** 3
        #sample_frac = (2 * 1024 ** 3) / total_size_bytes
        sample_df = df.sample(frac=0.001).compute()

        load_time = datetime.now() - start_time
        print(f"数据加载完成，耗时: {load_time}")
        #print("抽样后数据量：", sample_df.shape)

        return sample_df

    #数据预处理
    def preprocess_data(self, df):
        print(f"\n{'=' * 50}")
        print(f"开始数据预处理")
        start_time = datetime.now()
        #df = df["age", "income", "credit_score", "gender", "country", "user_name", "chinese_name"]

        '''数据缺失检测与处理'''
        print(f"数据缺失检测与处理")

        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_report = pd.DataFrame({
            '缺失值数量': missing,
            '缺失值比例(%)': [round(pct, 2) for pct in missing_pct]
        })
        print(missing_report.sort_values('缺失值比例(%)', ascending=False))

        df = df.dropna(subset=["age", "income", "credit_score", "gender"])

        '''数据异常检测与处理'''
        print(f"数据异常检测与处理：采用四分位距异常值检测法")

        numeric_cols = ["age", "income", "credit_score"]
        outlier_report = pd.DataFrame()

        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            print("异常值边界：", lower_bound, upper_bound)

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_pct = (len(outliers) / len(df)) * 100
            outlier_report.loc[col, "异常值比例(%)"] = round(outlier_pct, 2)

            # 移除异常值
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        print(outlier_report)
        clean_size = len(df)
        print("\nIQR 处理完成，清洗后数据量：", clean_size)

        '''处理完毕'''
        load_time = datetime.now() - start_time
        print(f"数据预处理完成，耗时: {load_time}")

        return df

    def visualize_data(self, df):
        print(f"{'=' * 50}")
        print(f"开始数据可视化")
        print(type(df))
        start_time = datetime.now()

        '''年龄分布图'''
        print(f"\n{'=' * 50}")
        print(f"计算年龄分布图")
        age_data = df["age"]
        plt.figure(figsize=(8, 5))
        plt.hist(age_data, bins=30, edgecolor='black')
        plt.title("年龄分布图")
        plt.xlabel("年龄")
        plt.ylabel("人数")
        plt.show()

        '''国家分布图'''
        print(f"\n{'=' * 50}")
        print(f"国家分布图")
        country_counts = df['country'].value_counts()

        top_countries = country_counts.head(10)
        other_countries = country_counts.tail(-10).sum()
        if other_countries != 0:
            top_countries["其他国家"] = other_countries

        plt.figure(figsize=(8, 8))
        plt.pie(top_countries, labels=top_countries.index, autopct='%1.1f%%', startangle=90,
                colors=plt.cm.Paired.colors)
        plt.title("按国家分布的用户占比")
        plt.axis('equal')
        plt.show()

        '''收入分布图'''
        print(f"\n{'=' * 50}")
        print(f"收入分布图")
        income_data = df["income"]
        plt.figure(figsize=(8, 5))
        plt.hist(income_data, bins=30, edgecolor='black')
        plt.title("收入分布图")
        plt.xlabel("收入")
        plt.ylabel("人数")
        plt.show()

        load_time = datetime.now() - start_time
        print(f"数据可视化完成，耗时: {load_time}")

    def recognize_highused(self, df):
        income_thresh = df["income"].quantile(0.75)
        credit_thresh = df["credit_score"].quantile(0.75)

        high_value_users = df[
            (df["income"] > income_thresh) &
            (df["credit_score"] > credit_thresh)
        ]
        print(f"识别到高价值用户数量：{high_value_users.shape[0]}")
        print("展示前5个高价值用户：")
        print(high_value_users[["user_name","chinese_name", "age", "income", "credit_score"]].head(5))

    def run(self):
        try:
            # 1.数据加载和采样
            df = self.load_data()

            # 2.数据采样
            df_sample = self.sample_data(df)

            # 3.数据预处理
            df_clean = self.preprocess_data(df_sample)

            # 4.数据可视化
            self.visualize_data(df_clean)

            # 5.高价值用户识别
            self.recognize_highused(df_clean)

        except Exception as e:
            print(str(e))


if __name__ == "__main__":
    # 数据集目录
    data_dirs = './data/30G_data'

    analyzer = DataAnalyzer(data_dirs)
    analyzer.run()
