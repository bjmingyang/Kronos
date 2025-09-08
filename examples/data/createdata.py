import pandas as pd
import os
from datetime import datetime, timedelta

def convert_and_merge_kline(root_dir, stock_code):
    """
    遍历指定目录，合并指定股票代码的所有日期数据，并转换为5分钟K线。
    此版本已优化，以兼容每天240条数据的情况。

    Args:
        root_dir (str): 数据根目录，例如 '/Users/your_username/.quant1x/minutes'
        stock_code (str): 股票代码，例如 'sh600380'

    Returns:
        pd.DataFrame: 合并后的多日5分钟K线数据
    """
    all_data_frames = []

    # 遍历根目录下的所有年文件夹
    for year_dir in sorted(os.listdir(root_dir)):
        year_path = os.path.join(root_dir, year_dir)
        if os.path.isdir(year_path):
            # 遍历年文件夹下的所有日期文件夹
            for date_dir in sorted(os.listdir(year_path)):
                date_path = os.path.join(year_path, date_dir)
                if os.path.isdir(date_path) and len(date_dir) == 8: # 确保是日期文件夹
                    file_path = os.path.join(date_path, f'{stock_code}.csv')

                    if not os.path.exists(file_path):
                        print(f"警告: {file_path} 未找到，跳过。")
                        continue

                    print(f"正在处理文件: {file_path}")

                    try:
                        df = pd.read_csv(file_path)
                    except Exception as e:
                        print(f"读取文件 {file_path} 失败: {e}")
                        continue

                    # 检查数据条数，如果不是240条则跳过并警告
                    if len(df) != 240:
                        print(f"警告: 文件 {file_path} 数据条数不为240，跳过。")
                        continue

                    # --- 5分钟K线转换逻辑 ---
                    timestamps = []

                    # 上午时段：9:30 - 11:30 (共120条数据)
                    morning_start_time = datetime.strptime(f'{date_dir}0930', '%Y%m%d%H%M')
                    for i in range(120):
                        timestamps.append(morning_start_time + timedelta(minutes=i))

                    # 下午时段：13:00 - 15:00 (共120条数据)
                    afternoon_start_time = datetime.strptime(f'{date_dir}1300', '%Y%m%d%H%M')
                    for i in range(120):
                        timestamps.append(afternoon_start_time + timedelta(minutes=i))

                    df['timestamps'] = timestamps
                    df['amount'] = df['Price'] * df['Vol']

                    # 按5分钟进行聚合
                    five_min_df = df.resample('5T', on='timestamps').agg({
                        'Price': ['first', 'max', 'min', 'last'],
                        'Vol': 'sum',
                        'amount': 'sum'
                    })

                    five_min_df.columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
                    five_min_df = five_min_df.dropna()

                    all_data_frames.append(five_min_df)

    if not all_data_frames:
        print("未找到任何数据文件，无法生成K线数据。")
        return None

    # 合并所有数据
    merged_df = pd.concat(all_data_frames)

    # 格式化时间戳并重置索引
    merged_df.index.name = 'timestamps'
    merged_df.reset_index(inplace=True)
    merged_df['timestamps'] = merged_df['timestamps'].dt.strftime('%Y-%m-%d %H:%M:%S')

    return merged_df

# 示例用法
# 请将 '/path/to/.quant1x/minutes' 替换为你实际的根目录路径
root_directory = '/root/.quant1x/minutes'
target_stock = 'sh600380'

output_dataframe = convert_and_merge_kline(root_directory, target_stock)

if output_dataframe is not None:
    # 打印前几行查看结果
    print(output_dataframe.head())
    print(f"\n合并后的数据总条数：{len(output_dataframe)}")

    # 导出为新的CSV文件
    output_filename = f'XSHG_5min_merged_{target_stock[2:]}.csv'
    output_dataframe.to_csv(output_filename, index=False)
    print(f"\n转换和合并完成，数据已保存到 {output_filename}")
