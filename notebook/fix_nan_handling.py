# ============================================================
# CHECK VÀ XỬ LÝ CÁC GIÁ TRỊ NaN - ĐIỀN GIÁ TRỊ THAY VÌ XÓA
# ============================================================

print("="*60)
print("TRƯỚC KHI XỬ LÝ NaN")
print("="*60)
print(f"Kích thước DataFrame: {df.shape}")
print(f"\nTổng số NaN trong mỗi cột:")
nan_counts = df.isnull().sum()
print(nan_counts[nan_counts > 0])

print(f"\nSố hàng có ít nhất 1 NaN: {df.isnull().any(axis=1).sum()}")
print(f"Tổng số NaN trong toàn bộ DataFrame: {df.isnull().sum().sum()}")

# ============================================================
# ĐIỀN GIÁ TRỊ VÀO CÁC CỘT NaN (KHÔNG XÓA HÀNG)
# ============================================================

print("\n" + "="*60)
print("XỬ LÝ NaN - ĐIỀN GIÁ TRỊ")
print("="*60)

df_filled = df.copy()

# 1. Điền 0 vào cột Z_err
print("1. Điền giá trị 0 vào cột Z_err...")
df_filled['Z_err'] = df_filled['Z_err'].fillna(0)
print(f"   ✓ Đã điền 0 vào {nan_counts['Z_err']} giá trị NaN trong Z_err")

# 2. Điền MEDIAN vào các cột số còn lại (an toàn hơn mean vì không bị ảnh hưởng bởi outliers)
print("\n2. Điền MEDIAN vào các cột số còn lại...")
numerical_cols = df_filled.select_dtypes(include=['float64', 'int64']).columns
numerical_cols = [col for col in numerical_cols if col not in ['Z_err', 'target']]

filled_count = 0
for col in numerical_cols:
    if df_filled[col].isnull().sum() > 0:
        median_value = df_filled[col].median()
        df_filled[col] = df_filled[col].fillna(median_value)
        filled_count += 1

print(f"   ✓ Đã điền MEDIAN vào {filled_count} cột")

# 3. Kiểm tra xem còn NaN không
print("\n3. Kiểm tra lại...")
remaining_nan = df_filled.isnull().sum().sum()
if remaining_nan > 0:
    print(f"   ⚠️ Còn {remaining_nan} NaN (có thể ở cột object)")
    # Điền giá trị 'Unknown' vào cột object nếu còn NaN
    object_cols = df_filled.select_dtypes(include=['object']).columns
    for col in object_cols:
        if col != 'object_id':  # Không điền vào cột ID
            df_filled[col] = df_filled[col].fillna('Unknown')
    print(f"   ✓ Đã điền 'Unknown' vào các cột text")
else:
    print(f"   ✓ Không còn NaN")

print("\n" + "="*60)
print("SAU KHI XỬ LÝ NaN")
print("="*60)
print(f"Kích thước DataFrame: {df_filled.shape} (KHÔNG thay đổi - KHÔNG XÓA HÀNG)")
print(f"Số hàng đã xóa: 0 ✓")
print(f"\nCòn NaN trong DataFrame không? {df_filled.isnull().any().any()}")
print(f"Tổng số NaN còn lại: {df_filled.isnull().sum().sum()}")

# Hiển thị thống kê cột Z_err sau khi điền
print(f"\nThống kê cột Z_err sau khi điền 0:")
print(f"  - Min: {df_filled['Z_err'].min()}")
print(f"  - Max: {df_filled['Z_err'].max()}")
print(f"  - Mean: {df_filled['Z_err'].mean():.4f}")
print(f"  - Số giá trị = 0: {(df_filled['Z_err'] == 0).sum()}")

# Cập nhật df với bản đã điền
df = df_filled.copy()

print("\n✓ DataFrame đã được cập nhật!")
print(f"✓ Kích thước hiện tại: {df.shape}")
print(f"✓ Đã GIỮ NGUYÊN tất cả {len(df)} hàng dữ liệu!")




