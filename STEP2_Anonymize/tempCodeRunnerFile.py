if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Ẩn danh tên người trong content (CRF - underthesea)")
#     parser.add_argument("--input", required=True, help="Đường dẫn file input (csv/xlsx)")
#     parser.add_argument("--output", required=True, help="Đường dẫn file output (csv/xlsx)")
#     parser.add_argument("--content_col", default="content", help="Tên cột chứa nội dung")
#     parser.add_argument("--min_len", type=int, default=20, help="Độ dài tối thiểu để giữ lại")
#     parser.add_argument("--audit_size", type=int, default=200, help="Số dòng mẫu để audit thủ công")
#     parser.add_argument(
#         "--no_extend", action="store_true",
#         help="Tắt fallback mở rộng sang token kế tiếp (ưu tiên Precision, giảm Recall)"
#     )
#     args = parser.parse_args()

#     run(
#         input_path=args.input,
#         output_path=args.output,
#         content_col=args.content_col,
#         min_len=args.min_len,
#         audit_sample_size=args.audit_size,
#         extend_to_next_token=not args.no_extend,
#     )