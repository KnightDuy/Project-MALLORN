# Mallorn Data Loader - Quick Start Guide

## Cài đặt nhanh

```bash
cd /Users/kaiser_1/Downloads/Mallorn/mallorn-astronomical-classification-challenge
pip install -r requirements.txt
```

## Sử dụng cơ bản

### 1. Import và khởi tạo

```python
from load_mallorn_data import MallornDataLoader

loader = MallornDataLoader("./")
```

### 2. Load metadata

```python
# Training data với labels
train_meta = loader.load_train_metadata()

# Test data không có labels
test_meta = loader.load_test_metadata()
```

### 3. Load lightcurve data

```python
# Load từ một split
split1_data = loader.load_split_lightcurves(1, mode='train')

# Load tất cả splits (mất nhiều thời gian)
all_data = loader.load_all_lightcurves(mode='train')

# Load cho một object cụ thể
obj_lc = loader.load_object_lightcurve('Dornhoth_fervain_onodrim', mode='train')
```

### 4. Extract features

```python
features = loader.create_features_from_lightcurve(obj_lc)
```

## Chạy demo

```bash
# Demo đầy đủ các tính năng
python load_mallorn_data.py

# Ví dụ đơn giản với visualization
python example_usage.py

# Jupyter notebook để explore
jupyter notebook explore_mallorn.ipynb
```

## Thông tin dataset

- **Training objects**: 3,043 (Target 0: 2,895, Target 1: 148)
- **Test objects**: 7,135
- **Number of splits**: 20
- **Filters**: u, g, r, i, z, y
- **Task**: Binary classification (transient vs non-transient)

## Các file chính

- `load_mallorn_data.py` - Class chính để load data
- `example_usage.py` - Ví dụ đơn giản
- `explore_mallorn.ipynb` - Jupyter notebook để explore
- `README.md` - Documentation đầy đủ
- `requirements.txt` - Dependencies

## Tips

1. **Memory**: Load tất cả lightcurves cùng lúc sẽ dùng nhiều RAM. Load theo split hoặc theo object nếu cần.

2. **Features**: Bắt đầu với features cơ bản từ `create_features_from_lightcurve()`, sau đó thêm features phức tạp hơn (period analysis, spectral features, etc.)

3. **Imbalanced data**: Target distribution rất imbalanced (95% class 0, 5% class 1). Cân nhắc:
   - Class weights
   - Oversampling/undersampling
   - SMOTE
   - Focal loss

4. **Cross-validation**: Sử dụng 20 splits có sẵn để cross-validate

5. **Missing filters**: Không phải object nào cũng có đủ 6 filters. Handle NaN values cẩn thận.

## Example: Build a simple model

```python
from load_mallorn_data import MallornDataLoader
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Load data
loader = MallornDataLoader("./")
train_meta = loader.load_train_metadata()

# 2. Extract features for all training objects
features_list = []
for obj_id in train_meta['object_id']:
    try:
        lc = loader.load_object_lightcurve(obj_id, mode='train')
        features = loader.create_features_from_lightcurve(lc)
        features['object_id'] = obj_id
        features_list.append(features)
    except Exception as e:
        print(f"Error: {obj_id}")

# 3. Create feature matrix
features_df = pd.DataFrame(features_list)
train_data = train_meta.merge(features_df, on='object_id')

# 4. Prepare X, y
feature_cols = [col for col in features_df.columns if col != 'object_id']
X = train_data[feature_cols].fillna(0)
y = train_data['target']

# 5. Train model
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# 6. Evaluate
print(f"Training accuracy: {model.score(X_train, y_train):.4f}")
print(f"Validation accuracy: {model.score(X_val, y_val):.4f}")
```

## Liên hệ

Nếu có vấn đề gì, check lại:
1. Đường dẫn đến data folder
2. Đã cài đặt requirements chưa
3. Python version >= 3.8

