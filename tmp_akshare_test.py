import akshare as ak

print("Testing stock_info_global_sina...")
try:
    df = ak.stock_info_global_sina()
    print(df.head(3))
except Exception as e:
    print(f"Error: {e}")

print("Testing news_economic_baidu...")
try:
    df = ak.news_economic_baidu()
    print(df.head(3))
except Exception as e:
    print(f"Error: {e}")
