"""
Mallorn Astronomical Classification Challenge - Data Loader
============================================================
This script provides functions to load and process the Mallorn astronomical 
classification dataset including metadata and lightcurve time-series data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

class MallornDataLoader:
    """
    A class to load and manage Mallorn astronomical classification data.
    
    Attributes:
        data_dir (Path): Path to the main data directory
        splits (List[int]): List of available split numbers
    """
    
    def __init__(self, data_dir: str = "./"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the directory containing the Mallorn data
        """
        self.data_dir = Path(data_dir)
        self.splits = list(range(1, 21))  # splits 1-20
        
        # Verify data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def load_train_metadata(self) -> pd.DataFrame:
        """
        Load training metadata/labels.
        
        Returns:
            DataFrame with columns: object_id, Z, Z_err, EBV, SpecType, 
            English Translation, split, target
        """
        train_log_path = self.data_dir / "train_log.csv"
        if not train_log_path.exists():
            raise FileNotFoundError(f"Train log not found: {train_log_path}")
        
        df = pd.read_csv(train_log_path)
        print(f"Loaded training metadata: {len(df)} objects")
        print(f"Target distribution:\n{df['target'].value_counts()}")
        print(f"SpecType distribution:\n{df['SpecType'].value_counts()}")
        return df
    
    def load_test_metadata(self) -> pd.DataFrame:
        """
        Load test metadata (no labels).
        
        Returns:
            DataFrame with columns: object_id, Z, Z_err, EBV, SpecType, 
            English Translation, split
        """
        test_log_path = self.data_dir / "test_log.csv"
        if not test_log_path.exists():
            raise FileNotFoundError(f"Test log not found: {test_log_path}")
        
        df = pd.read_csv(test_log_path)
        print(f"Loaded test metadata: {len(df)} objects")
        return df
    
    def load_sample_submission(self) -> pd.DataFrame:
        """
        Load sample submission file.
        
        Returns:
            DataFrame with columns: object_id, prediction
        """
        sample_path = self.data_dir / "sample_submission.csv"
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample submission not found: {sample_path}")
        
        df = pd.read_csv(sample_path)
        print(f"Loaded sample submission: {len(df)} objects")
        return df
    
    def load_split_lightcurves(self, split_num: int, mode: str = 'train') -> pd.DataFrame:
        """
        Load lightcurve data for a specific split.
        
        Args:
            split_num: Split number (1-20)
            mode: Either 'train' or 'test'
        
        Returns:
            DataFrame with columns: object_id, Time (MJD), Flux, Flux_err, Filter
        """
        if split_num not in self.splits:
            raise ValueError(f"Invalid split number: {split_num}. Must be 1-20.")
        
        if mode not in ['train', 'test']:
            raise ValueError(f"Mode must be 'train' or 'test', got: {mode}")
        
        split_dir = self.data_dir / f"split_{split_num:02d}"
        lightcurve_path = split_dir / f"{mode}_full_lightcurves.csv"
        
        if not lightcurve_path.exists():
            raise FileNotFoundError(f"Lightcurve file not found: {lightcurve_path}")
        
        df = pd.read_csv(lightcurve_path)
        print(f"Loaded {mode} lightcurves for split {split_num:02d}: "
              f"{len(df)} observations, {df['object_id'].nunique()} objects")
        return df
    
    def load_all_lightcurves(self, mode: str = 'train') -> pd.DataFrame:
        """
        Load lightcurve data from all splits.
        
        Args:
            mode: Either 'train' or 'test'
        
        Returns:
            Combined DataFrame from all splits
        """
        all_lightcurves = []
        
        print(f"Loading all {mode} lightcurves from 20 splits...")
        for split_num in self.splits:
            try:
                df = self.load_split_lightcurves(split_num, mode)
                all_lightcurves.append(df)
            except Exception as e:
                warnings.warn(f"Error loading split {split_num}: {e}")
        
        combined_df = pd.concat(all_lightcurves, ignore_index=True)
        print(f"\nTotal combined: {len(combined_df)} observations, "
              f"{combined_df['object_id'].nunique()} unique objects")
        return combined_df
    
    def load_object_lightcurve(self, object_id: str, mode: str = 'train') -> pd.DataFrame:
        """
        Load lightcurve data for a specific object.
        
        Args:
            object_id: The object identifier
            mode: Either 'train' or 'test'
        
        Returns:
            DataFrame containing lightcurve for the specified object
        """
        # First, find which split contains this object
        metadata = self.load_train_metadata() if mode == 'train' else self.load_test_metadata()
        
        object_row = metadata[metadata['object_id'] == object_id]
        if object_row.empty:
            raise ValueError(f"Object {object_id} not found in {mode} metadata")
        
        split_name = object_row['split'].values[0]
        split_num = int(split_name.split('_')[1])
        
        # Load the specific split
        df = self.load_split_lightcurves(split_num, mode)
        object_lc = df[df['object_id'] == object_id].copy()
        
        if object_lc.empty:
            raise ValueError(f"No lightcurve data found for {object_id}")
        
        # Sort by time
        object_lc = object_lc.sort_values('Time (MJD)').reset_index(drop=True)
        return object_lc
    
    def get_dataset_summary(self) -> Dict:
        """
        Get a summary of the entire dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        train_meta = self.load_train_metadata()
        test_meta = self.load_test_metadata()
        
        summary = {
            'n_train_objects': len(train_meta),
            'n_test_objects': len(test_meta),
            'n_splits': len(self.splits),
            'train_target_distribution': train_meta['target'].value_counts().to_dict(),
            'train_spectype_distribution': train_meta['SpecType'].value_counts().to_dict(),
            'test_spectype_distribution': test_meta['SpecType'].value_counts().to_dict(),
            'filters': ['u', 'g', 'r', 'i', 'z', 'y'],  # Standard photometric filters
        }
        
        return summary
    
    def create_features_from_lightcurve(self, lightcurve: pd.DataFrame) -> Dict:
        """
        Extract basic statistical features from a lightcurve.
        
        Args:
            lightcurve: DataFrame containing lightcurve data for one object
        
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Overall statistics
        features['n_observations'] = len(lightcurve)
        features['time_span'] = lightcurve['Time (MJD)'].max() - lightcurve['Time (MJD)'].min()
        features['mean_flux'] = lightcurve['Flux'].mean()
        features['std_flux'] = lightcurve['Flux'].std()
        features['median_flux'] = lightcurve['Flux'].median()
        features['max_flux'] = lightcurve['Flux'].max()
        features['min_flux'] = lightcurve['Flux'].min()
        
        # Per-filter statistics
        for filt in ['u', 'g', 'r', 'i', 'z', 'y']:
            filt_data = lightcurve[lightcurve['Filter'] == filt]
            if len(filt_data) > 0:
                features[f'{filt}_n_obs'] = len(filt_data)
                features[f'{filt}_mean_flux'] = filt_data['Flux'].mean()
                features[f'{filt}_std_flux'] = filt_data['Flux'].std()
            else:
                features[f'{filt}_n_obs'] = 0
                features[f'{filt}_mean_flux'] = np.nan
                features[f'{filt}_std_flux'] = np.nan
        
        return features


def demo_usage():
    """Demonstrate how to use the MallornDataLoader class."""
    print("=" * 70)
    print("Mallorn Data Loader - Demo")
    print("=" * 70)
    print()
    
    # Initialize loader
    loader = MallornDataLoader("./")
    
    # Load metadata
    print("\n" + "=" * 70)
    print("1. Loading Training Metadata")
    print("=" * 70)
    train_meta = loader.load_train_metadata()
    print(f"\nFirst few rows:")
    print(train_meta.head())
    
    print("\n" + "=" * 70)
    print("2. Loading Test Metadata")
    print("=" * 70)
    test_meta = loader.load_test_metadata()
    print(f"\nFirst few rows:")
    print(test_meta.head())
    
    # Load lightcurves for one split
    print("\n" + "=" * 70)
    print("3. Loading Lightcurves from Split 01")
    print("=" * 70)
    split1_train = loader.load_split_lightcurves(1, mode='train')
    print(f"\nFirst few rows:")
    print(split1_train.head(10))
    
    # Load lightcurve for a specific object
    print("\n" + "=" * 70)
    print("4. Loading Lightcurve for a Specific Object")
    print("=" * 70)
    example_object = train_meta['object_id'].iloc[0]
    print(f"Object ID: {example_object}")
    object_lc = loader.load_object_lightcurve(example_object, mode='train')
    print(f"\nLightcurve shape: {object_lc.shape}")
    print(object_lc.head(10))
    
    # Extract features
    print("\n" + "=" * 70)
    print("5. Extracting Features from Lightcurve")
    print("=" * 70)
    features = loader.create_features_from_lightcurve(object_lc)
    print("Extracted features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    # Dataset summary
    print("\n" + "=" * 70)
    print("6. Dataset Summary")
    print("=" * 70)
    summary = loader.get_dataset_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo_usage()

