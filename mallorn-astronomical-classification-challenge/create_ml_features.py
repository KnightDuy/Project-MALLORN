"""
Mallorn Feature Engineering Script
===================================
Creates a comprehensive feature dataset for machine learning from astronomical lightcurve data.

Features extracted per filter (u, g, r, i, z, y):
- Basic statistics: n_obs, time_span, flux stats (mean, std, min, max, median)
- Amplitude: flux_max - flux_min
- Distribution: skewness, kurtosis
- Dynamics: t_peak, rise_time, decay_time
- Slopes: max_slope_up, max_slope_down
- Area under curve (AUC) using trapezoidal integration

Cross-filter features:
- Color indices (flux differences between filters)
- Peak time differences between filters
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from scipy import stats
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class MallornFeatureExtractor:
    """Extract comprehensive features from Mallorn astronomical data."""
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            self.data_dir = Path(__file__).resolve().parent
        else:
            self.data_dir = Path(data_dir)

        self.filters = ['u', 'g', 'r', 'i', 'z', 'y']
        self.splits = list(range(1, 21))

        
    def load_metadata(self, mode: str = 'train') -> pd.DataFrame:
        """Load metadata file."""
        file_path = self.data_dir / f"{mode}_log.csv"
        return pd.read_csv(file_path)
    
    def load_object_lightcurve(self, object_id: str, split_name: str, mode: str = 'train') -> pd.DataFrame:
        """Load lightcurve for a specific object."""
        split_num = int(split_name.split('_')[1])
        split_dir = self.data_dir / f"split_{split_num:02d}"
        lc_path = split_dir / f"{mode}_full_lightcurves.csv"
        
        # Read and filter
        df = pd.read_csv(lc_path)
        object_lc = df[df['object_id'] == object_id].copy()
        
        # Sort by time
        object_lc = object_lc.sort_values('Time (MJD)').reset_index(drop=True)
        return object_lc
    
    def extract_filter_features(self, lightcurve: pd.DataFrame, filter_name: str) -> Dict:
        """
        Extract comprehensive features for a single filter.
        
        Args:
            lightcurve: Full lightcurve DataFrame for an object
            filter_name: Filter name (u, g, r, i, z, y)
            
        Returns:
            Dictionary of features for this filter
        """
        # Filter data for this specific filter
        filt_data = lightcurve[lightcurve['Filter'] == filter_name].copy()
        
        features = {}
        prefix = f'{filter_name}_'
        
        # If no data for this filter, return NaN features
        if len(filt_data) == 0:
            return self._get_nan_filter_features(prefix)
        
        # Sort by time
        filt_data = filt_data.sort_values('Time (MJD)').reset_index(drop=True)
        
        times = filt_data['Time (MJD)'].values
        fluxes = filt_data['Flux'].values
        
        # Basic statistics
        features[f'{prefix}n_obs'] = len(filt_data)
        features[f'{prefix}time_span'] = times[-1] - times[0] if len(times) > 1 else 0
        
        # Flux statistics
        features[f'{prefix}flux_mean'] = np.mean(fluxes)
        features[f'{prefix}flux_std'] = np.std(fluxes)
        features[f'{prefix}flux_min'] = np.min(fluxes)
        features[f'{prefix}flux_max'] = np.max(fluxes)
        features[f'{prefix}flux_median'] = np.median(fluxes)
        features[f'{prefix}amplitude'] = np.max(fluxes) - np.min(fluxes)
        
        # Distribution features
        if len(fluxes) >= 3:
            features[f'{prefix}flux_skew'] = stats.skew(fluxes)
            features[f'{prefix}flux_kurtosis'] = stats.kurtosis(fluxes)
        else:
            features[f'{prefix}flux_skew'] = np.nan
            features[f'{prefix}flux_kurtosis'] = np.nan
        
        # Shape/dynamics features
        if len(filt_data) >= 2:
            # Time of peak flux (relative to first observation)
            peak_idx = np.argmax(fluxes)
            t_first = times[0]
            t_last = times[-1]
            t_peak = times[peak_idx]
            
            features[f'{prefix}t_peak'] = t_peak - t_first
            features[f'{prefix}rise_time'] = t_peak - t_first
            features[f'{prefix}decay_time'] = t_last - t_peak
            
            # Slope features (dFlux/dt)
            if len(filt_data) >= 2:
                dt = np.diff(times)
                dflux = np.diff(fluxes)
                
                # Avoid division by zero
                valid_mask = dt > 0
                if np.any(valid_mask):
                    slopes = np.zeros_like(dt)
                    slopes[valid_mask] = dflux[valid_mask] / dt[valid_mask]
                    
                    features[f'{prefix}max_slope_up'] = np.max(slopes[valid_mask]) if np.any(valid_mask) else np.nan
                    features[f'{prefix}max_slope_down'] = np.min(slopes[valid_mask]) if np.any(valid_mask) else np.nan
                else:
                    features[f'{prefix}max_slope_up'] = np.nan
                    features[f'{prefix}max_slope_down'] = np.nan
            else:
                features[f'{prefix}max_slope_up'] = np.nan
                features[f'{prefix}max_slope_down'] = np.nan
            
            # Area under curve (trapezoidal integration)
            features[f'{prefix}auc'] = np.trapz(fluxes, times)
        else:
            features[f'{prefix}t_peak'] = np.nan
            features[f'{prefix}rise_time'] = np.nan
            features[f'{prefix}decay_time'] = np.nan
            features[f'{prefix}max_slope_up'] = np.nan
            features[f'{prefix}max_slope_down'] = np.nan
            features[f'{prefix}auc'] = np.nan
        
        return features
    
    def _get_nan_filter_features(self, prefix: str) -> Dict:
        """Return NaN features when no data available for a filter."""
        return {
            f'{prefix}n_obs': 0,
            f'{prefix}time_span': np.nan,
            f'{prefix}flux_mean': np.nan,
            f'{prefix}flux_std': np.nan,
            f'{prefix}flux_min': np.nan,
            f'{prefix}flux_max': np.nan,
            f'{prefix}flux_median': np.nan,
            f'{prefix}amplitude': np.nan,
            f'{prefix}flux_skew': np.nan,
            f'{prefix}flux_kurtosis': np.nan,
            f'{prefix}t_peak': np.nan,
            f'{prefix}rise_time': np.nan,
            f'{prefix}decay_time': np.nan,
            f'{prefix}max_slope_up': np.nan,
            f'{prefix}max_slope_down': np.nan,
            f'{prefix}auc': np.nan,
        }
    
    def extract_cross_filter_features(self, filter_features: Dict) -> Dict:
        """
        Extract cross-filter features (colors and time differences).
        
        Args:
            filter_features: Dictionary containing all per-filter features
            
        Returns:
            Dictionary of cross-filter features
        """
        cross_features = {}
        
        # Define filter pairs for color indices
        filter_pairs = [
            ('u', 'g'), ('g', 'r'), ('r', 'i'), 
            ('i', 'z'), ('z', 'y'),
            ('u', 'r'), ('g', 'i'), ('r', 'z')  # Additional useful color indices
        ]
        
        # Color features (mean flux differences)
        for f1, f2 in filter_pairs:
            mean1 = filter_features.get(f'{f1}_flux_mean', np.nan)
            mean2 = filter_features.get(f'{f2}_flux_mean', np.nan)
            cross_features[f'color_mean_{f1}_{f2}'] = mean1 - mean2
        
        # Peak time differences
        for f1, f2 in filter_pairs:
            t_peak1 = filter_features.get(f'{f1}_t_peak', np.nan)
            t_peak2 = filter_features.get(f'{f2}_t_peak', np.nan)
            cross_features[f't_peak_diff_{f1}_{f2}'] = t_peak1 - t_peak2
        
        return cross_features
    
    def extract_object_features(self, object_id: str, split_name: str, 
                                metadata_row: pd.Series, mode: str = 'train') -> Dict:
        """
        Extract all features for a single object.
        
        Args:
            object_id: Object identifier
            split_name: Split name (e.g., 'split_01')
            metadata_row: Row from metadata DataFrame
            mode: 'train' or 'test'
            
        Returns:
            Dictionary containing all features for this object
        """
        features = {'object_id': object_id}
        
        # Add metadata features
        features['Z'] = metadata_row['Z']
        features['Z_err'] = metadata_row['Z_err']
        features['EBV'] = metadata_row['EBV']
        
        # Add target if in training mode
        if mode == 'train':
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
                              output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Create complete feature dataset for all objects.
        
        Args:
            mode: 'train' or 'test'
            output_file: Optional path to save CSV file
            
        Returns:
            DataFrame with all features
        """
        print(f"Creating feature dataset for {mode} data...")
        
        # Load metadata
        metadata = self.load_metadata(mode)
        print(f"Loaded metadata for {len(metadata)} objects")
        
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


def main():
    """Main function to create ML-ready feature files."""
    print("="*70)
    print("Mallorn Feature Engineering Pipeline")
    print("="*70)
    print()
    
    # Initialize extractor
    extractor = MallornFeatureExtractor()

    
    # Create training features
    print("\n" + "="*70)
    print("TRAINING DATA")
    print("="*70)
    train_features = extractor.create_feature_dataset(
        mode='train',
        output_file='train_features_ml.csv'
    )
    
    print(f"\nTraining features summary:")
    print(f"- Shape: {train_features.shape}")
    print(f"- Missing values per column:")
    missing_counts = train_features.isnull().sum()
    print(missing_counts[missing_counts > 0].head(10))
    
    # Create test features
    print("\n" + "="*70)
    print("TEST DATA")
    print("="*70)
    test_features = extractor.create_feature_dataset(
        mode='test',
        output_file='test_features_ml.csv'
    )
    
    print(f"\nTest features summary:")
    print(f"- Shape: {test_features.shape}")
    print(f"- Missing values per column:")
    missing_counts = test_features.isnull().sum()
    print(missing_counts[missing_counts > 0].head(10))
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print("\nOutput files created:")
    print("  - train_features_ml.csv")
    print("  - test_features_ml.csv")
    print("\nThese files are ready for ML training.")
    print("\nExample usage:")
    print("  import pandas as pd")
    print("  train = pd.read_csv('train_features_ml.csv')")
    print("  X = train.drop(['object_id', 'target'], axis=1)")
    print("  y = train['target']")
    print("="*70)


if __name__ == "__main__":
    main()

