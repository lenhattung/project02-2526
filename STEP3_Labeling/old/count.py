import pandas as pd

# Đọc dữ liệu
df_comments = pd.read_csv("STEP3_Labeling/output_labeled_posts_vsfc_phobert.csv")
df_posts = pd.read_csv("STEP3_Labeling/output_labeled_comments_vsfc_phobert.csv")

# Thêm cột để biết dữ liệu đến từ comments hay posts
df_comments["data_type"] = "comment"
df_posts["data_type"] = "post"

# Gộp 2 dataframe
df_all = pd.concat([df_comments, df_posts], ignore_index=True)

# Đếm tổng số dòng sau khi gộp
print("Tổng số lượng dữ liệu:", len(df_all))

# Đếm số lượng từng sentiment_raw_label
sentiment_counts = df_all["sentiment_raw_label"].value_counts(dropna=False)

print("\nTổng sentiment_raw_label:")
print(sentiment_counts)