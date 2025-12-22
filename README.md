# Mallorn Astronomical Classification Challenge - Data Loader

Python code để load và xử lý dữ liệu Mallorn astronomical classification challenge.

## Cấu trúc dữ liệu

Dataset bao gồm:

- **train_log.csv**: Metadata và labels cho training data (3,044 objects)
- **test_log.csv**: Metadata cho test data (7,136 objects)
- **sample_submission.csv**: Format file submission mẫu
- **split_01 đến split_20**: 20 folders chứa lightcurve data
  - `train_full_lightcurves.csv`: Time-series data cho training
  - `test_full_lightcurves.csv`: Time-series data cho testing

### Metadata Columns

**Training:**
- `object_id`: ID của object
- `Z`: Redshift
- `Z_err`: Redshift error
- `EBV`: Extinction (E(B-V))
- `SpecType`: Spectroscopic type (AGN, SN Ia, SN II, SN Ib, TDE)
- `English Translation`: Tên object đã được dịch
- `split`: Split number
- `target`: Label (0 hoặc 1) - **đây là target cần predict**

**Testing:**
- Giống như training nhưng không có column `target`

### Lightcurve Columns

- `object_id`: ID của object
- `Time (MJD)`: Thời gian quan sát (Modified Julian Date)
- `Flux`: Giá trị flux
- `Flux_err`: Flux measurement error
- `Filter`: Photometric filter (u, g, r, i, z, y)

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng

### 1. Load metadata

```python
from load_mallorn_data import MallornDataLoader

# Khởi tạo loader
loader = MallornDataLoader("./")

# Load training metadata
train_meta = loader.load_train_metadata()
print(train_meta.head())

# Load test metadata
test_meta = loader.load_test_metadata()
print(test_meta.head())
```

### 2. Load lightcurve data

```python
# Load lightcurves từ một split cụ thể
split1_train = loader.load_split_lightcurves(split_num=1, mode='train')

# Load lightcurves từ tất cả splits
all_train_lc = loader.load_all_lightcurves(mode='train')

# Load lightcurve của một object cụ thể
object_id = "Dornhoth_fervain_onodrim"
lc = loader.load_object_lightcurve(object_id, mode='train')
```

### 3. Trích xuất features

```python
# Trích xuất statistical features từ lightcurve
features = loader.create_features_from_lightcurve(lc)
print(features)
```

### 4. Dataset summary

```python
# Xem tổng quan về dataset
summary = loader.get_dataset_summary()
print(summary)
```

## Chạy ví dụ

```bash
# Demo đầy đủ các chức năng
python load_mallorn_data.py

# Ví dụ đơn giản với visualization
python example_usage.py
```

## Class MallornDataLoader

### Methods

- `load_train_metadata()`: Load training metadata
- `load_test_metadata()`: Load test metadata
- `load_sample_submission()`: Load sample submission file
- `load_split_lightcurves(split_num, mode)`: Load lightcurves từ một split
- `load_all_lightcurves(mode)`: Load lightcurves từ tất cả splits
- `load_object_lightcurve(object_id, mode)`: Load lightcurve của một object
- `get_dataset_summary()`: Lấy thông tin tổng quan về dataset
- `create_features_from_lightcurve(lightcurve)`: Trích xuất features từ lightcurve

## Target Variable

**Binary classification task:**
- `target = 0`: Non-transient objects
- `target = 1`: Transient objects

Distribution trong training data sẽ được hiển thị khi load metadata.

## Filters

Dataset sử dụng 6 photometric filters:
- **u**: Ultraviolet
- **g**: Green
- **r**: Red
- **i**: Near-infrared
- **z**: Infrared
- **y**: Far-infrared

## Notes

- Mỗi object có thể có số lượng observations khác nhau
- Không phải tất cả objects đều có observations ở tất cả filters
- Time series data không đều đặn (irregular sampling)
- Một số giá trị flux có thể âm (do noise)

## Ví dụ workflow

```python
from load_mallorn_data import MallornDataLoader
import pandas as pd

# 1. Initialize
loader = MallornDataLoader("./")

# 2. Load metadata
train_meta = loader.load_train_metadata()
test_meta = loader.load_test_metadata()

# 3. Build feature matrix từ lightcurves
features_list = []
for obj_id in train_meta['object_id']:
    try:
        lc = loader.load_object_lightcurve(obj_id, mode='train')
        features = loader.create_features_from_lightcurve(lc)
        features['object_id'] = obj_id
        features_list.append(features)
    except Exception as e:
        print(f"Error processing {obj_id}: {e}")

# 4. Create feature DataFrame
features_df = pd.DataFrame(features_list)

# 5. Merge với metadata
train_data = train_meta.merge(features_df, on='object_id')

# 6. Train model
# X = train_data.drop(['object_id', 'target', ...], axis=1)
# y = train_data['target']
# model.fit(X, y)

# 7. Make predictions on test data
# ...
```

## License

Data từ Mallorn Astronomical Classification Challenge.

