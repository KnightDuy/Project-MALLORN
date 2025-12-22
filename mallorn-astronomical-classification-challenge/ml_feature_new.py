import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from scipy import stats
from scipy.signal import find_peaks
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class MallornFeatureExtractor:
    """Extract comprehensive features from Mallorn astronomical data."""
    
    def __init__(self, data_dir: str = "./", cache_lightcurves: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            data_dir: Path to the directory containing Mallorn data
            cache_lightcurves: Whether to cache loaded lightcurves for efficiency
        """
        self.data_dir = Path(data_dir)
        self.filters = ['u', 'g', 'r', 'i', 'z', 'y']
        self.splits = list(range(1, 21))
        self.cache_lightcurves = cache_lightcurves
        self._lightcurve_cache = {}
        
    def load_metadata(self, mode: str = 'train') -> pd.DataFrame:
        """Load metadata file."""
        file_path = self.data_dir / f"{mode}_log.csv"
        return pd.read_csv(file_path)
    
    def load_split_lightcurves(self, split_num: int, mode: str = 'train') -> pd.DataFrame:
        """Load all lightcurves for a split (for caching efficiency)."""
        cache_key = f"{mode}_split_{split_num}"
        
        if self.cache_lightcurves and cache_key in self._lightcurve_cache:
            return self._lightcurve_cache[cache_key]
        
        split_dir = self.data_dir / f"split_{split_num:02d}"
        lc_path = split_dir / f"{mode}_full_lightcurves.csv"
        
        df = pd.read_csv(lc_path)
        df = df.sort_values(['object_id', 'Time (MJD)']).reset_index(drop=True)
        
        if self.cache_lightcurves:
            self._lightcurve_cache[cache_key] = df
            
        return df
    
    def load_object_lightcurve(self, object_id: str, split_name: str, mode: str = 'train') -> pd.DataFrame:
        """Load lightcurve for a specific object."""
        split_num = int(split_name.split('_')[1])
        
        # Load entire split and filter
        split_df = self.load_split_lightcurves(split_num, mode)
        object_lc = split_df[split_df['object_id'] == object_id].copy()
        
        return object_lc.sort_values('Time (MJD)').reset_index(drop=True)
    
    def extract_filter_features(self, lightcurve: pd.DataFrame, filter_name: str) -> Dict:
        """
        Extract comprehensive features for a single filter.
        
        Args:
            lightcurve: Full lightcurve DataFrame for an object
            filter_name: Filter name (u, g, r, i, z, y)
            
        Returns:
            Dictionary of features for this filter
        """
        filt_data = lightcurve[lightcurve['Filter'] == filter_name].copy()
        
        features = {}
        prefix = f'{filter_name}_'
        
        if len(filt_data) == 0:
            return self._get_nan_filter_features(prefix)
        
        filt_data = filt_data.sort_values('Time (MJD)').reset_index(drop=True)
        
        times = filt_data['Time (MJD)'].values
        fluxes = filt_data['Flux'].values
        flux_errors = filt_data['Flux_err'].values if 'Flux_err' in filt_data.columns else None
        
        # Basic counting statistics
        features[f'{prefix}n_obs'] = len(filt_data)
        features[f'{prefix}time_span'] = times[-1] - times[0] if len(times) > 1 else 0
        
        # Observation cadence features
        if len(times) > 1:
            time_diffs = np.diff(times)
            features[f'{prefix}cadence_mean'] = np.mean(time_diffs)
            features[f'{prefix}cadence_std'] = np.std(time_diffs)
            features[f'{prefix}cadence_median'] = np.median(time_diffs)
            features[f'{prefix}cadence_min'] = np.min(time_diffs)
            features[f'{prefix}cadence_max'] = np.max(time_diffs)
        else:
            for stat in ['mean', 'std', 'median', 'min', 'max']:
                features[f'{prefix}cadence_{stat}'] = np.nan
        
        # Flux statistics
        features[f'{prefix}flux_mean'] = np.mean(fluxes)
        features[f'{prefix}flux_std'] = np.std(fluxes)
        features[f'{prefix}flux_min'] = np.min(fluxes)
        features[f'{prefix}flux_max'] = np.max(fluxes)
        features[f'{prefix}flux_median'] = np.median(fluxes)
        features[f'{prefix}amplitude'] = np.max(fluxes) - np.min(fluxes)
        
        # Percentile features
        features[f'{prefix}flux_q25'] = np.percentile(fluxes, 25)
        features[f'{prefix}flux_q75'] = np.percentile(fluxes, 75)
        features[f'{prefix}flux_iqr'] = features[f'{prefix}flux_q75'] - features[f'{prefix}flux_q25']
        
        # Signal-to-noise features
        if flux_errors is not None:
            snr = fluxes / np.clip(flux_errors, 1e-10, None)
            features[f'{prefix}snr_mean'] = np.mean(snr)
            features[f'{prefix}snr_median'] = np.median(snr)
            features[f'{prefix}snr_max'] = np.max(snr)
        else:
            features[f'{prefix}snr_mean'] = np.nan
            features[f'{prefix}snr_median'] = np.nan
            features[f'{prefix}snr_max'] = np.nan
        
        # Distribution features
        if len(fluxes) >= 3:
            features[f'{prefix}flux_skew'] = stats.skew(fluxes)
            features[f'{prefix}flux_kurtosis'] = stats.kurtosis(fluxes)
        else:
            features[f'{prefix}flux_skew'] = np.nan
            features[f'{prefix}flux_kurtosis'] = np.nan
        
        # Weighted mean features (if errors available)
        if flux_errors is not None and np.all(flux_errors > 0):
            weights = 1.0 / (flux_errors ** 2)
            features[f'{prefix}flux_weighted_mean'] = np.average(fluxes, weights=weights)
        else:
            features[f'{prefix}flux_weighted_mean'] = features[f'{prefix}flux_mean']
        
        # Shape/dynamics features
        if len(filt_data) >= 2:
            peak_idx = np.argmax(fluxes)
            t_first, t_last = times[0], times[-1]
            t_peak = times[peak_idx]
            
            features[f'{prefix}t_peak'] = t_peak - t_first
            features[f'{prefix}rise_time'] = t_peak - t_first
            features[f'{prefix}decay_time'] = t_last - t_peak
            
            # Rise/decay ratio
            if features[f'{prefix}rise_time'] > 0:
                features[f'{prefix}rise_decay_ratio'] = features[f'{prefix}decay_time'] / features[f'{prefix}rise_time']
            else:
                features[f'{prefix}rise_decay_ratio'] = np.nan
            
            # Normalized peak position
            if features[f'{prefix}time_span'] > 0:
                features[f'{prefix}peak_position_norm'] = features[f'{prefix}rise_time'] / features[f'{prefix}time_span']
            else:
                features[f'{prefix}peak_position_norm'] = np.nan
            
            # Slope features
            dt = np.diff(times)
            dflux = np.diff(fluxes)
            valid_mask = dt > 0
            
            if np.any(valid_mask):
                slopes = np.zeros_like(dt)
                slopes[valid_mask] = dflux[valid_mask] / dt[valid_mask]
                
                features[f'{prefix}max_slope_up'] = np.max(slopes[slopes > 0]) if np.any(slopes > 0) else 0
                features[f'{prefix}max_slope_down'] = np.min(slopes[slopes < 0]) if np.any(slopes < 0) else 0
                features[f'{prefix}mean_slope'] = np.mean(slopes[valid_mask])
                features[f'{prefix}std_slope'] = np.std(slopes[valid_mask])
            else:
                for stat in ['max_slope_up', 'max_slope_down', 'mean_slope', 'std_slope']:
                    features[f'{prefix}{stat}'] = np.nan
            
            # Area under curve
            features[f'{prefix}auc'] = np.trapz(fluxes, times)
            
            # Peak detection (number of local maxima)
            try:
                peaks, _ = find_peaks(fluxes, prominence=np.std(fluxes) * 0.5)
                features[f'{prefix}n_peaks'] = len(peaks)
            except:
                features[f'{prefix}n_peaks'] = np.nan
            
        else:
            for stat in ['t_peak', 'rise_time', 'decay_time', 'rise_decay_ratio', 
                        'peak_position_norm', 'max_slope_up', 'max_slope_down',
                        'mean_slope', 'std_slope', 'auc', 'n_peaks']:
                features[f'{prefix}{stat}'] = np.nan
        
        # Variability features
        if len(fluxes) > 1:
            # Beyond 1 sigma (percentage of points)
            flux_std = np.std(fluxes)
            flux_mean = np.mean(fluxes)
            beyond_1sigma = np.sum(np.abs(fluxes - flux_mean) > flux_std) / len(fluxes)
            features[f'{prefix}beyond_1sigma'] = beyond_1sigma
            
            # Coefficient of variation
            features[f'{prefix}coeff_variation'] = flux_std / np.abs(flux_mean) if flux_mean != 0 else np.nan
        else:
            features[f'{prefix}beyond_1sigma'] = np.nan
            features[f'{prefix}coeff_variation'] = np.nan
        
        return features
    
    def _get_nan_filter_features(self, prefix: str) -> Dict:
        """Return NaN features when no data available for a filter."""
        feature_names = [
            'n_obs', 'time_span', 'cadence_mean', 'cadence_std', 'cadence_median',
            'cadence_min', 'cadence_max', 'flux_mean', 'flux_std', 'flux_min',
            'flux_max', 'flux_median', 'amplitude', 'flux_q25', 'flux_q75', 'flux_iqr',
            'snr_mean', 'snr_median', 'snr_max', 'flux_skew', 'flux_kurtosis',
            'flux_weighted_mean', 't_peak', 'rise_time', 'decay_time',
            'rise_decay_ratio', 'peak_position_norm', 'max_slope_up', 'max_slope_down',
            'mean_slope', 'std_slope', 'auc', 'n_peaks', 'beyond_1sigma', 'coeff_variation'
        ]
        
        return {f'{prefix}{name}': (0 if name == 'n_obs' else np.nan) 
                for name in feature_names}
    
    def extract_cross_filter_features(self, filter_features: Dict) -> Dict:
        """Extract cross-filter features (colors, correlations, time differences)."""
        cross_features = {}
        
        # Define filter pairs for color indices
        filter_pairs = [
            ('u', 'g'), ('g', 'r'), ('r', 'i'), ('i', 'z'), ('z', 'y'),
            ('u', 'r'), ('g', 'i'), ('r', 'z'), ('u', 'i'), ('g', 'z')
        ]
        
        # Color features (flux differences and ratios)
        for f1, f2 in filter_pairs:
            mean1 = filter_features.get(f'{f1}_flux_mean', np.nan)
            mean2 = filter_features.get(f'{f2}_flux_mean', np.nan)
            
            cross_features[f'color_mean_{f1}_{f2}'] = mean1 - mean2
            
            # Color ratio
            if not np.isnan(mean2) and mean2 != 0:
                cross_features[f'color_ratio_{f1}_{f2}'] = mean1 / mean2
            else:
                cross_features[f'color_ratio_{f1}_{f2}'] = np.nan
        
        # Peak time differences
        for f1, f2 in filter_pairs:
            t_peak1 = filter_features.get(f'{f1}_t_peak', np.nan)
            t_peak2 = filter_features.get(f'{f2}_t_peak', np.nan)
            cross_features[f't_peak_diff_{f1}_{f2}'] = t_peak1 - t_peak2
        
        # Rise time differences
        for f1, f2 in filter_pairs[:5]:  # Just main sequential pairs
            rise1 = filter_features.get(f'{f1}_rise_time', np.nan)
            rise2 = filter_features.get(f'{f2}_rise_time', np.nan)
            cross_features[f'rise_time_diff_{f1}_{f2}'] = rise1 - rise2
        
        # Amplitude ratios
        for f1, f2 in filter_pairs[:5]:
            amp1 = filter_features.get(f'{f1}_amplitude', np.nan)
            amp2 = filter_features.get(f'{f2}_amplitude', np.nan)
            if not np.isnan(amp2) and amp2 != 0:
                cross_features[f'amplitude_ratio_{f1}_{f2}'] = amp1 / amp2
            else:
                cross_features[f'amplitude_ratio_{f1}_{f2}'] = np.nan
        
        return cross_features
    
    def extract_global_features(self, lightcurve: pd.DataFrame, filter_features: Dict) -> Dict:
        """Extract features across all filters."""
        global_features = {}
        
        # Total number of observations
        global_features['total_n_obs'] = len(lightcurve)
        
        # Total time span
        if len(lightcurve) > 0:
            times = lightcurve['Time (MJD)'].values
            global_features['global_time_span'] = np.max(times) - np.min(times)
            global_features['global_time_first'] = np.min(times)
            global_features['global_time_last'] = np.max(times)
        else:
            global_features['global_time_span'] = np.nan
            global_features['global_time_first'] = np.nan
            global_features['global_time_last'] = np.nan
        
        # Number of filters with data
        n_filters_with_data = sum(1 for f in self.filters 
                                  if filter_features.get(f'{f}_n_obs', 0) > 0)
        global_features['n_filters_with_data'] = n_filters_with_data
        
        # Filter coverage fraction
        global_features['filter_coverage'] = n_filters_with_data / len(self.filters)
        
        return global_features
    
    def extract_object_features(self, object_id: str, split_name: str, 
                                metadata_row: pd.Series, mode: str = 'train') -> Dict:
        """Extract all features for a single object."""
        features = {'object_id': object_id}
        
        # Add metadata features
        features['Z'] = metadata_row['Z']
        features['Z_err'] = metadata_row['Z_err']
        features['EBV'] = metadata_row['EBV']
        
        # Add target if in training mode
        if mode == 'train' and 'target' in metadata_row:
            features['target'] = metadata_row['target']
        
        try:
            # Load lightcurve
            lightcurve = self.load_object_lightcurve(object_id, split_name, mode)
            
            # Extract per-filter features
            filter_features = {}
            for filter_name in self.filters:
                filt_feats = self.extract_filter_features(lightcurve, filter_name)
                filter_features.update(filt_feats)
            
            features.update(filter_features)
            
            # Extract global features
            global_features = self.extract_global_features(lightcurve, filter_features)
            features.update(global_features)
            
            # Extract cross-filter features
            cross_features = self.extract_cross_filter_features(filter_features)
            features.update(cross_features)
            
        except Exception as e:
            print(f"Error processing {object_id}: {e}")
            # Return features with NaNs if lightcurve loading fails
            for filter_name in self.filters:
                features.update(self._get_nan_filter_features(f'{filter_name}_'))
        
        return features
    
    def create_feature_dataset(self, mode: str = 'train', 
                              output_file: Optional[str] = None,
                              chunk_size: Optional[int] = None) -> pd.DataFrame:
        """
        Create complete feature dataset for all objects.
        
        Args:
            mode: 'train' or 'test'
            output_file: Optional path to save CSV file
            chunk_size: If specified, process and save in chunks (memory efficient)
            
        Returns:
            DataFrame with all features
        """
        print(f"Creating feature dataset for {mode} data...")
        
        # Load metadata
        metadata = self.load_metadata(mode)
        print(f"Loaded metadata for {len(metadata)} objects")
        
        if chunk_size is not None:
            # Process in chunks and save incrementally
            return self._create_features_chunked(metadata, mode, output_file, chunk_size)
        
        # Extract features for each object
        all_features = []
        
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), 
                            desc=f"Extracting features ({mode})"):
            object_id = row['object_id']
            split_name = row['split']
            
            features = self.extract_object_features(object_id, split_name, row, mode)
            all_features.append(features)
        
        # Create DataFrame
        features_df = pd.DataFrame(all_features)
        
        print(f"\nFeature extraction complete!")
        print(f"Shape: {features_df.shape}")
        print(f"Columns: {len(features_df.columns)}")
        
        # Save to file if specified
        if output_file:
            output_path = self.data_dir / output_file
            features_df.to_csv(output_path, index=False)
            print(f"Saved to: {output_path}")
        
        return features_df
    
    def _create_features_chunked(self, metadata: pd.DataFrame, mode: str, 
                                output_file: str, chunk_size: int) -> pd.DataFrame:
        """Process features in chunks for memory efficiency."""
        output_path = self.data_dir / output_file
        
        for i in range(0, len(metadata), chunk_size):
            chunk_metadata = metadata.iloc[i:i+chunk_size]
            chunk_features = []
            
            for idx, row in tqdm(chunk_metadata.iterrows(), 
                               total=len(chunk_metadata),
                               desc=f"Chunk {i//chunk_size + 1}"):
                features = self.extract_object_features(
                    row['object_id'], row['split'], row, mode
                )
                chunk_features.append(features)
            
            chunk_df = pd.DataFrame(chunk_features)
            
            # Append to file
            if i == 0:
                chunk_df.to_csv(output_path, index=False, mode='w')
            else:
                chunk_df.to_csv(output_path, index=False, mode='a', header=False)
        
        return pd.read_csv(output_path)


class MissingDataHandler:
    """Handle missing data with various imputation strategies."""
    
    def __init__(self):
        self.imputers = {}
        self.feature_groups = {}
        self.missing_indicators = {}
        
    def analyze_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze missing data patterns."""
        print("\n" + "="*70)
        print("PHÂN TÍCH DỮ LIỆU BỊ THIẾU")
        print("="*70)
        
        missing_stats = pd.DataFrame({
            'column': df.columns,
            'missing_count': df.isnull().sum().values,
            'missing_percent': (df.isnull().sum() / len(df) * 100).values
        })
        
        missing_stats = missing_stats[missing_stats['missing_count'] > 0].sort_values(
            'missing_percent', ascending=False
        )
        
        if len(missing_stats) > 0:
            print(f"\nTổng số cột có dữ liệu thiếu: {len(missing_stats)}")
            print(f"Tổng số cột: {len(df.columns)}")
            print(f"\nTop 20 cột có nhiều dữ liệu thiếu nhất:")
            print(missing_stats.head(20).to_string(index=False))
            
            # Phân loại theo mức độ thiếu
            severe = missing_stats[missing_stats['missing_percent'] > 50]
            moderate = missing_stats[(missing_stats['missing_percent'] > 20) & 
                                    (missing_stats['missing_percent'] <= 50)]
            mild = missing_stats[missing_stats['missing_percent'] <= 20]
            
            print(f"\nPhân loại theo mức độ thiếu:")
            print(f"  - Nghiêm trọng (>50%): {len(severe)} cột")
            print(f"  - Trung bình (20-50%): {len(moderate)} cột")
            print(f"  - Nhẹ (≤20%): {len(mild)} cột")
        else:
            print("\nKhông có dữ liệu thiếu!")
        
        return missing_stats
    
    def _identify_feature_groups(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify feature groups for targeted imputation."""
        groups = {
            'metadata': [],
            'filter_features': {f: [] for f in ['u', 'g', 'r', 'i', 'z', 'y']},
            'cross_filter': [],
            'global': []
        }
        
        for col in df.columns:
            if col in ['object_id', 'target']:
                continue
            elif col in ['Z', 'Z_err', 'EBV']:
                groups['metadata'].append(col)
            elif col.startswith(('color_', 't_peak_diff_', 'rise_time_diff_', 'amplitude_ratio_')):
                groups['cross_filter'].append(col)
            elif col.startswith(('total_', 'global_', 'n_filters', 'filter_coverage')):
                groups['global'].append(col)
            else:
                # Filter-specific features
                for filt in ['u', 'g', 'r', 'i', 'z', 'y']:
                    if col.startswith(f'{filt}_'):
                        groups['filter_features'][filt].append(col)
                        break
        
        return groups
    
    def handle_missing_data(self, df: pd.DataFrame, method: str = 'advanced',
                          exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Handle missing data with specified method.
        
        Args:
            df: Input DataFrame
            method: One of 'simple', 'knn', 'iterative', 'advanced'
            exclude_cols: Columns to exclude from imputation
            
        Returns:
            DataFrame with imputed values
        """
        print(f"\n{'='*70}")
        print(f"XỬ LÝ DỮ LIỆU THIẾU - Phương pháp: {method.upper()}")
        print("="*70)
        
        if exclude_cols is None:
            exclude_cols = ['object_id', 'target']
        
        df_imputed = df.copy()
        
        # Separate columns
        id_cols = [col for col in exclude_cols if col in df.columns]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if method == 'simple':
            df_imputed = self._simple_imputation(df_imputed, feature_cols)
        elif method == 'knn':
            df_imputed = self._knn_imputation(df_imputed, feature_cols)
        elif method == 'iterative':
            df_imputed = self._iterative_imputation(df_imputed, feature_cols)
        elif method == 'advanced':
            df_imputed = self._advanced_imputation(df_imputed, feature_cols)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Verify imputation
        remaining_missing = df_imputed[feature_cols].isnull().sum().sum()
        print(f"\nSố lượng giá trị thiếu còn lại: {remaining_missing}")
        
        return df_imputed
    
    def _simple_imputation(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Simple imputation: median for numerical, most frequent for categorical."""
        print("\nSử dụng chiến lược đơn giản:")
        print("  - Median cho các cột số")
        print("  - Constant (0) cho các cột đếm")
        
        df_imputed = df.copy()
        
        # Identify count columns (n_obs, n_peaks, etc.)
        count_cols = [col for col in feature_cols if 
                     any(x in col for x in ['n_obs', 'n_peaks', 'n_filters'])]
        
        # Median for most features
        other_cols = [col for col in feature_cols if col not in count_cols]
        
        if count_cols:
            imputer_count = SimpleImputer(strategy='constant', fill_value=0)
            df_imputed[count_cols] = imputer_count.fit_transform(df_imputed[count_cols])
            print(f"  - Đã xử lý {len(count_cols)} cột đếm với giá trị 0")
        
        if other_cols:
            imputer_median = SimpleImputer(strategy='median')
            df_imputed[other_cols] = imputer_median.fit_transform(df_imputed[other_cols])
            print(f"  - Đã xử lý {len(other_cols)} cột khác với median")
        
        return df_imputed
    
    def _knn_imputation(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """KNN imputation using neighbors."""
        print("\nSử dụng KNN Imputation (k=5 neighbors)...")
        print("  - Ước lượng dựa trên các đối tượng tương tự")
        
        df_imputed = df.copy()
        
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        df_imputed[feature_cols] = imputer.fit_transform(df_imputed[feature_cols])
        
        print("  - Hoàn thành KNN imputation")
        
        return df_imputed
    
    def _iterative_imputation(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Iterative imputation (MICE algorithm)."""
        print("\nSử dụng Iterative Imputation (MICE)...")
        print("  - Mô hình hóa từng feature dựa trên các features khác")
        print("  - Lặp đến khi hội tụ")
        
        df_imputed = df.copy()
        
        imputer = IterativeImputer(
            max_iter=10,
            random_state=42,
            initial_strategy='median',
            imputation_order='ascending'
        )
        
        df_imputed[feature_cols] = imputer.fit_transform(df_imputed[feature_cols])
        
        print("  - Hoàn thành iterative imputation")
        
        return df_imputed
    
    def _advanced_imputation(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Advanced imputation using domain knowledge and multiple strategies."""
        print("\nSử dụng chiến lược nâng cao:")
        print("  - Nhóm features theo loại (filter, cross-filter, global)")
        print("  - Áp dụng chiến lược phù hợp cho từng nhóm")
        
        df_imputed = df.copy()
        groups = self._identify_feature_groups(df)
        
        # 1. Metadata features - simple median (robust)
        if groups['metadata']:
            meta_cols = [col for col in groups['metadata'] if col in feature_cols]
            if meta_cols:
                meta_df = df_imputed[meta_cols].copy()

                # ép numeric cho chắc (median cần numeric)
                meta_df = meta_df.apply(pd.to_numeric, errors="coerce")

                # cột nào NaN 100% thì fill constant để tránh SimpleImputer drop cột
                all_nan_cols = [c for c in meta_cols if meta_df[c].isna().all()]
                if all_nan_cols:
                    df_imputed[all_nan_cols] = 0.0  # hoặc -1.0 tuỳ bạn
                    print(f"  ⚠️ Metadata all-NaN cols filled with 0: {all_nan_cols}")

                use_cols = [c for c in meta_cols if c not in all_nan_cols]
                if use_cols:
                    imputer = SimpleImputer(strategy="median")
                    df_imputed[use_cols] = imputer.fit_transform(meta_df[use_cols])

                print(f"  ✓ Metadata: {len(meta_cols)} cột (median/constant)")

        
        # 2. Filter-specific features - impute within each filter group
        for filt, cols in groups['filter_features'].items():
            filt_cols = [col for col in cols if col in feature_cols]
            if filt_cols and len(filt_cols) > 5:
                # Use KNN within filter group
                count_cols = [col for col in filt_cols if 'n_obs' in col or 'n_peaks' in col]
                other_cols = [col for col in filt_cols if col not in count_cols]
                
                if count_cols:
                    df_imputed[count_cols] = df_imputed[count_cols].fillna(0)
                
                if other_cols:
                    imputer = KNNImputer(n_neighbors=5, weights='distance')
                    df_imputed[other_cols] = imputer.fit_transform(df_imputed[other_cols])
                
                print(f"  ✓ Filter {filt}: {len(filt_cols)} cột (KNN)")
        
        # 3. Cross-filter features - iterative imputation
        cross_cols = [col for col in groups['cross_filter'] if col in feature_cols]
        if cross_cols:
            cross_df = df_imputed[cross_cols].copy()

    # ép numeric + loại inf
            cross_df = cross_df.apply(pd.to_numeric, errors="coerce")
            cross_df = cross_df.replace([np.inf, -np.inf], np.nan)

            # cột NaN 100% -> fill constant để không làm IterativeImputer "kẹt"
            all_nan_cols = [c for c in cross_cols if cross_df[c].isna().all()]
            if all_nan_cols:
                df_imputed[all_nan_cols] = 0.0
                print(f"  ⚠️ Cross-filter all-NaN cols filled with 0: {all_nan_cols}")

            use_cols = [c for c in cross_cols if c not in all_nan_cols]

            if use_cols:
                imputer = IterativeImputer(
                    max_iter=5,
                    random_state=42,
                        initial_strategy="median",  # quan trọng: đảm bảo init không ra NaN
                    skip_complete=True
                )
                df_imputed[use_cols] = imputer.fit_transform(cross_df[use_cols])

            print(f"  ✓ Cross-filter: {len(cross_cols)} cột (iterative/constant)")                 
        
        # 4. Global features - median
        global_cols = [col for col in groups['global'] if col in feature_cols]
        if global_cols:
            count_cols = [col for col in global_cols if any(x in col for x in ['n_', 'total_'])]
            other_cols = [col for col in global_cols if col not in count_cols]
            
            if count_cols:
                df_imputed[count_cols] = df_imputed[count_cols].fillna(0)
            if other_cols:
                imputer = SimpleImputer(strategy='median')
                df_imputed[other_cols] = imputer.fit_transform(df_imputed[other_cols])
            
            print(f"  ✓ Global: {len(global_cols)} cột (median/constant)")
        
        # 5. Add missing indicators for highly missing features
        df_imputed = self._add_missing_indicators(df, df_imputed, feature_cols)
        
        return df_imputed
    
    def _add_missing_indicators(self, df_original: pd.DataFrame, 
                                df_imputed: pd.DataFrame,
                                feature_cols: List[str],
                                threshold: float = 0.3) -> pd.DataFrame:
        """Add binary indicators for features with high missingness."""
        high_missing_cols = []
        
        for col in feature_cols:
            missing_rate = df_original[col].isnull().sum() / len(df_original)
            if missing_rate > threshold:
                indicator_col = f'{col}_was_missing'
                df_imputed[indicator_col] = df_original[col].isnull().astype(int)
                high_missing_cols.append(col)
        
        if high_missing_cols:
            print(f"\n  ✓ Đã thêm {len(high_missing_cols)} missing indicators (>30% thiếu)")
        
        return df_imputed
    
    def create_robust_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional robust features less sensitive to missing data."""
        print("\n" + "="*70)
        print("TẠO FEATURES BỔ SUNG CHỐNG MISSING DATA")
        print("="*70)
        
        df_robust = df.copy()
        
        # 1. Aggregated filter statistics
        filter_means = []
        for filt in ['u', 'g', 'r', 'i', 'z', 'y']:
            col = f'{filt}_flux_mean'
            if col in df.columns:
                filter_means.append(df[col])
        
        if filter_means:
            filter_means_df = pd.DataFrame(filter_means).T
            df_robust['all_filters_flux_mean'] = filter_means_df.mean(axis=1)
            df_robust['all_filters_flux_std'] = filter_means_df.std(axis=1)
            df_robust['all_filters_flux_min'] = filter_means_df.min(axis=1)
            df_robust['all_filters_flux_max'] = filter_means_df.max(axis=1)
            df_robust['all_filters_flux_range'] = df_robust['all_filters_flux_max'] - df_robust['all_filters_flux_min']
            print("  ✓ Đã tạo aggregated flux statistics")
        
        # 2. Observation coverage metrics
        obs_counts = []
        for filt in ['u', 'g', 'r', 'i', 'z', 'y']:
            col = f'{filt}_n_obs'
            if col in df.columns:
                obs_counts.append(df[col].fillna(0))
        
        if obs_counts:
            obs_df = pd.DataFrame(obs_counts).T
            df_robust['obs_total'] = obs_df.sum(axis=1)
            df_robust['obs_mean'] = obs_df.mean(axis=1)
            df_robust['obs_std'] = obs_df.std(axis=1)
            df_robust['obs_nonzero_count'] = (obs_df > 0).sum(axis=1)
            print("  ✓ Đã tạo observation coverage metrics")
        
        # 3. Color diversity score
        color_cols = [col for col in df.columns if col.startswith('color_mean_')]
        if color_cols:
            colors_df = df[color_cols]
            df_robust['color_diversity'] = colors_df.std(axis=1)
            df_robust['color_range'] = colors_df.max(axis=1) - colors_df.min(axis=1)
            print(f"  ✓ Đã tạo color diversity từ {len(color_cols)} colors")
        
        # 4. Time-based robustness
        time_spans = []
        for filt in ['u', 'g', 'r', 'i', 'z', 'y']:
            col = f'{filt}_time_span'
            if col in df.columns:
                time_spans.append(df[col])
        
        if time_spans:
            time_df = pd.DataFrame(time_spans).T
            df_robust['time_span_max'] = time_df.max(axis=1)
            df_robust['time_span_mean'] = time_df.mean(axis=1)
            print("  ✓ Đã tạo time-based features")
        
        new_features = len(df_robust.columns) - len(df.columns)
        print(f"\nTổng cộng đã tạo thêm: {new_features} features mới")
        
        return df_robust


def main():
    """Main function to create ML-ready feature files with missing data handling."""
    print("="*70)
    print("MALLORN FEATURE ENGINEERING PIPELINE")
    print("Bao gồm: Trích xuất features + Xử lý dữ liệu thiếu")
    print("="*70)
    print()
    
    # Initialize extractor
    extractor = MallornFeatureExtractor("/home/duy/Downloads/Mallorn_update/Mallorn/mallorn-astronomical-classification-challenge/", cache_lightcurves=True)
    handler = MissingDataHandler()
    
    # Create training features
    print("\n" + "="*70)
    print("BƯỚC 1: TRÍCH XUẤT FEATURES - TRAINING DATA")
    print("="*70)
    train_features_raw = extractor.create_feature_dataset(
        mode='train',
        output_file='train_features_raw.csv'
    )
    
    print(f"\nKích thước dữ liệu training ban đầu: {train_features_raw.shape}")
    
    # Analyze missing data
    train_missing_stats = handler.analyze_missing_data(train_features_raw)
    
    # Create robust features
    train_features_robust = handler.create_robust_features(train_features_raw)
    
    # Handle missing data with advanced strategy
    train_features_clean = handler.handle_missing_data(
        train_features_robust,
        method='advanced',
        exclude_cols=['object_id', 'target']
    )
    
    # Save processed training data
    train_output_path = extractor.data_dir / 'train_features_ml_1.csv'
    train_features_clean.to_csv(train_output_path, index=False)
    print(f"\n✓ Đã lưu training features đã xử lý: {train_output_path}")
    print(f"  Shape cuối cùng: {train_features_clean.shape}")
    
    if 'target' in train_features_clean.columns:
        print(f"\n  Phân bố target:")
        print(train_features_clean['target'].value_counts().to_string())
    
    # Create test features
    print("\n" + "="*70)
    print("BƯỚC 2: TRÍCH XUẤT FEATURES - TEST DATA")
    print("="*70)
    test_features_raw = extractor.create_feature_dataset(
        mode='test',
        output_file='test_features_raw.csv'
    )
    
    print(f"\nKích thước dữ liệu test ban đầu: {test_features_raw.shape}")
    
    # Analyze missing data
    test_missing_stats = handler.analyze_missing_data(test_features_raw)
    
    # Create robust features
    test_features_robust = handler.create_robust_features(test_features_raw)
    
    # Handle missing data
    test_features_clean = handler.handle_missing_data(
        test_features_robust,
        method='advanced',
        exclude_cols=['object_id']
    )
    
    # Save processed test data
    test_output_path = extractor.data_dir / 'test_features_ml_1.csv'
    test_features_clean.to_csv(test_output_path, index=False)
    print(f"\n✓ Đã lưu test features đã xử lý: {test_output_path}")
    print(f"  Shape cuối cùng: {test_features_clean.shape}")
    
    # Final summary
    print("\n" + "="*70)
    print("HOÀN THÀNH!")
    print("="*70)
    print("\nCác file đã tạo:")
    print("  1. train_features_raw.csv - Dữ liệu training chưa xử lý")
    print("  2. train_features_ml.csv - Dữ liệu training đã xử lý (SỬ DỤNG FILE NÀY)")
    print("  3. test_features_raw.csv - Dữ liệu test chưa xử lý")
    print("  4. test_features_ml.csv - Dữ liệu test đã xử lý (SỬ DỤNG FILE NÀY)")
    
    print("\nSố lượng features:")
    print(f"  - Training: {train_features_clean.shape[1]} cột")
    print(f"  - Test: {test_features_clean.shape[1]} cột")
    
    print("\nSử dụng như sau:")
    print("  import pandas as pd")
    print("  from sklearn.ensemble import RandomForestClassifier")
    print("  ")
    print("  # Load dữ liệu đã xử lý")
    print("  train = pd.read_csv('train_features_ml.csv')")
    print("  test = pd.read_csv('test_features_ml.csv')")
    print("  ")
    print("  # Chuẩn bị training")
    print("  X_train = train.drop(['object_id', 'target'], axis=1)")
    print("  y_train = train['target']")
    print("  X_test = test.drop(['object_id'], axis=1)")
    print("  ")
    print("  # Train model")
    print("  model = RandomForestClassifier()")
    print("  model.fit(X_train, y_train)")
    print("="*70)


if __name__ == "__main__":
    main()