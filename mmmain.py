import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import tqdm
import glob
from datetime import timedelta

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

class DataAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_data(self):
        print(f"\n{'=' * 50}")
        print(f"开始加载 30G Parquet数据集...")

        start_time = datetime.now()

        parquet_pattern = os.path.join(self.data_dir, "*.parquet")
        df = dd.read_parquet(parquet_pattern, chunksize="1024MB")

        load_time = datetime.now() - start_time
        print(f"数据加载完成，耗时: {load_time}")
        #print(res_chunk.shape)
        #print("字段如下：", df.columns)

        return df

    #抽样
    def sample_data(self, df):
        print(f"\n{'=' * 50}")
        print(f"开始抽样")
        start_time = datetime.now()

        sample_df = df.sample(frac=0.01).compute()

        load_time = datetime.now() - start_time
        print(f"数据加载完成，耗时: {load_time}")
        #print("抽样后数据量：", sample_df.shape)

        return sample_df

    #数据预处理
    def preprocess_data(self, df, i):
        #print(f"\n{'=' * 50}")
        print(f"开始数据预处理")
        start_time = datetime.now()
        df = df.compute()

        '''数据缺失检测与处理'''
        #print(f"数据缺失检测与处理")

        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_report = pd.DataFrame({
            '缺失值数量': missing,
            '缺失值比例(%)': [round(pct, 2) for pct in missing_pct]
        })
        missing_report.to_csv("missing_report.csv",encoding='utf-8', index=False, mode='a', header=False)
        if i == 0:
            missing_report.to_csv("missing_report.csv", encoding='utf-8', index=False, mode='w')
        else:
            with open("missing_report.csv", "a", encoding="utf-8") as f:
                f.write(f"\n### 分块 {i + 1} ###\n")
            missing_report.to_csv("missing_report.csv", encoding='utf-8', index=False, mode='a', header=True)

        df = df.dropna(subset=["age", "income",  "gender"])

        '''数据异常检测与处理'''
        #print(f"数据异常检测与处理")

        numeric_cols = ["age", "income"]
        outlier_report = pd.DataFrame()

        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            #print("异常值边界：", lower_bound, upper_bound)

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            #outliers = df[(df[col] < 0) | (df[col] > 90)]
            outlier_pct = (len(outliers) / len(df)) * 100
            outlier_report.loc[col, "异常值比例(%)"] = round(outlier_pct, 2)
            # 移除异常值
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            #df = df[(df[col] >= 0) & (df[col] <= 90)]

        if i == 0:
            outlier_report.to_csv("outlier_report.csv", encoding='utf-8', index=False, mode='w')
        else:
            with open("outlier_report.csv", "a", encoding="utf-8") as f:
                f.write(f"\n### 分块 {i + 1} ###\n")
            outlier_report.to_csv("outlier_report.csv", encoding='utf-8', index=False, mode='a', header=True)

        '''处理完毕'''
        load_time = datetime.now() - start_time
        print(f"数据预处理完成，耗时: {load_time}")

        return df, load_time

    def visualize_all_data(self):
        print(f"\n{'=' * 50}\n开始低内存整体数据可视化（逐块加载）")
        start_time = datetime.now()

        age_bins = list(range(0, 101, 5))
        income_bins = list(range(0, 200001, 10000))
        age_counts = [0] * (len(age_bins) - 1)
        income_counts = [0] * (len(income_bins) - 1)
        registration_counts = {}

        for file_path in glob.glob("temp_clean_chunks/clean_chunk_*.csv"):
            chunk = pd.read_csv(file_path)

            # 年龄统计
            if "age" in chunk.columns:
                age_hist = pd.cut(chunk["age"], bins=age_bins, right=False).value_counts(sort=False)
                for idx, count in age_hist.items():
                    age_counts[idx.left // 5] += count

            # 收入统计
            if "income" in chunk.columns:
                income_hist = pd.cut(chunk["income"], bins=income_bins, right=False).value_counts(sort=False)
                for idx, count in income_hist.items():
                    income_counts[idx.left // 10000] += count

            # 注册时间统计
            if "registration_date" in chunk.columns:
                chunk["registration_date"] = pd.to_datetime(chunk["registration_date"], errors="coerce")
                monthly_counts = chunk["registration_date"].dt.to_period("M").value_counts()
                for month, count in monthly_counts.items():
                    month_str = str(month)
                    registration_counts[month_str] = registration_counts.get(month_str, 0) + count

        # 绘图
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        plt.subplots_adjust(hspace=0.4)

        # 年龄分布
        axs[0, 0].bar([f"{age_bins[i]}-{age_bins[i + 1] - 1}" for i in range(len(age_counts))], age_counts,
                      color='skyblue', edgecolor='black')
        axs[0, 0].set_title("年龄分布")
        axs[0, 0].set_xlabel("年龄区间")
        axs[0, 0].set_ylabel("人数")
        axs[0, 0].tick_params(axis='x', rotation=45)

        # 收入分布
        axs[0, 1].bar([f"{income_bins[i]}-{income_bins[i + 1] - 1}" for i in range(len(income_counts))], income_counts,
                      color='lightgreen', edgecolor='black')
        axs[0, 1].set_title("收入分布")
        axs[0, 1].set_xlabel("收入区间")
        axs[0, 1].set_ylabel("人数")
        axs[0, 1].tick_params(axis='x', rotation=45)

        # 注册时间趋势
        if registration_counts:
            sorted_months = sorted(registration_counts.items())
            months, counts = zip(*sorted_months)
            axs[1, 0].bar(months, counts, color='orange')
            axs[1, 0].set_title("月度注册用户数")
            axs[1, 0].set_xlabel("月份")
            axs[1, 0].set_ylabel("注册人数")
            axs[1, 0].tick_params(axis='x', rotation=45)

        plt.suptitle("整体数据可视化", y=1.02)
        plt.tight_layout()
        plt.show()

        elapsed = datetime.now() - start_time
        print(f"可视化完成，用时：{elapsed}")

    # def visualize_all_data(self, df):
    #     print(f"\n{'=' * 50}")
    #     print(f"开始整体数据可视化")
    #     start_time = datetime.now()
    #
    #     plt.figure(figsize=(15, 10))
    #     plt.subplots_adjust(hspace=0.4)
    #
    #     # 年龄分布
    #     plt.subplot(2, 2, 1)
    #     plt.hist(df["age"], bins=30, edgecolor='black', color='skyblue')
    #     plt.title("年龄分布")
    #     plt.xlabel("年龄")
    #     plt.ylabel("人数")
    #
    #     # 收入分布
    #     plt.subplot(2, 2, 2)
    #     plt.hist(df["income"], bins=30, edgecolor='black', color='lightgreen')
    #     plt.title("收入分布")
    #     plt.xlabel("收入")
    #     plt.ylabel("人数")
    #
    #     # 注册时间趋势
    #     plt.subplot(2, 2, 3)
    #     if 'registration_date' in df.columns:
    #         df["registration_date"] = pd.to_datetime(df["registration_date"], errors='coerce')
    #         df = df.dropna(subset=["registration_date"])
    #         monthly_counts = df["registration_date"].dt.to_period("M").value_counts().sort_index()
    #         monthly_counts.index = monthly_counts.index.astype(str)
    #         monthly_counts.plot(kind="bar", color='orange')
    #         plt.title("月度注册用户数")
    #         plt.xticks(rotation=45)
    #         plt.xlabel("月份")
    #         plt.ylabel("用户数")
    #
    #     plt.suptitle("整体数据可视化", y=1.02)
    #     plt.tight_layout()
    #     plt.show()
    #
    #     load_time = datetime.now() - start_time
    #     print(f"可视化完成，耗时: {load_time}")

    def recognize_highused(self, df):
        #print(f"\n{'=' * 50}")
        print("识别高价值用户")
        start_time = datetime.now()

        current_date = pd.to_datetime(datetime.now().date())
        df['registration_date'] = pd.to_datetime(df['registration_date'])
        df['membership_days'] = (current_date - df['registration_date']).dt.days

        def normalize(series, reverse=False):
            if reverse:  # 用于注册时长（越久越好）
                return (series.max() - series) / (series.max() - series.min() + 1e-10)
            return (series - series.min()) / (series.max() - series.min() + 1e-10)

        # 计算各项评分
        df['income_score'] = normalize(df['income'])
        df['age_score'] = df['age'].apply(lambda x: max(0, 1 - abs((x - 30) / 20)))  # 30岁为峰值
        df['loyalty_score'] = normalize(df['membership_days'], reverse=True)  # 注册越久分越高

        # 权重
        df['hv_score'] = (
                0.65 * df['income_score'] +  # 收入权重65%
                0.05 * df['age_score'] +  # 年龄权重5%
                0.3 * df['loyalty_score']  # 忠诚度权重30%
        )

        # 5. 获取前10名高价值用户
        top_10_users = df.sort_values('hv_score', ascending=False).head(10)

        # 6. 打印结果
        #print("\n排名前10的高价值用户：")
        #print(top_10_users[['user_name', 'fullname', 'age', 'income', 'registration_date', 'hv_score']].to_string(index=False))

        load_time = datetime.now() - start_time
        print(f"高价值用户识别完成，耗时: {load_time}")
        print(f"\n{'=' * 50}")
        return top_10_users, load_time

    def run(self):
        try:
            # 1. 数据加载 - 保持惰性加载
            df = self.load_data()
            progress_bar = tqdm(total=df.npartitions, desc="处理分块")
            clean_len = 0
            pre_time = timedelta()
            reg_time = timedelta()

            os.makedirs("temp_clean_chunks", exist_ok=True)

            # 2. 分块处理数据
            for i, partition in enumerate(df.partitions):
                print(f"\n{'=' * 50}")
                #print(f"正在处理第 {i + 1} 个数据分块...")

                try:
                    # 预处理当前分块
                    df_clean, pre_time_part = self.preprocess_data(partition, i)  # 显式计算当前分块
                    clean_len += len(df_clean)
                    pre_time += pre_time_part
                    # 检查处理后的数据是否为空
                    if len(df_clean) == 0:
                        print(f"第 {i + 1} 个分块清洗后无数据，跳过")
                        continue

                    df_clean.to_csv(f"temp_clean_chunks/clean_chunk_{i}.csv",encoding='utf-8', index=False)

                    # 识别高价值用户
                    high_value_users, reg_time_part = self.recognize_highused(df_clean)
                    reg_time += reg_time_part

                    # 保存当前分块的高价值用户
                    if i == 0:
                        high_value_users.to_csv("high_value_users.csv", index=False, mode='w')  # 首次写入创建文件
                    else:
                        high_value_users.to_csv("high_value_users.csv", index=False, mode='a', header=False)  # 追加写入

                    # 手动释放内存
                    del df_clean, high_value_users
                    progress_bar.update(1)

                except Exception as e:
                    print(f"第 {i + 1} 个分块处理异常: {str(e)}")
                    continue
            progress_bar.close()
            print("清洗后的长度：", clean_len)
            print("预处理总用时：", pre_time)
            print("识别总用时：", reg_time)
            self.visualize_all_data()

        except Exception as e:
            print(f"程序运行失败: {str(e)}")


if __name__ == "__main__":
    # 数据集目录
    data_dir = './data/30G_data/30G_data_new'
    #data_dir = './data/test'
    analyzer = DataAnalyzer(data_dir)
    analyzer.run()
