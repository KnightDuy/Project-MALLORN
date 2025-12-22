"""
Simple example showing how to load and visualize Mallorn data
"""

import matplotlib.pyplot as plt
from load_mallorn_data import MallornDataLoader

# Initialize the data loader
loader = MallornDataLoader("./")

# Load training metadata
train_meta = loader.load_train_metadata()
print("\nTraining data loaded successfully!")
print(f"Shape: {train_meta.shape}")
print(f"\nColumns: {list(train_meta.columns)}")

# Load a specific object's lightcurve
example_object = train_meta['object_id'].iloc[0]
print(f"\nLoading lightcurve for: {example_object}")
lightcurve = loader.load_object_lightcurve(example_object, mode='train')

print(f"Number of observations: {len(lightcurve)}")
print(f"Filters used: {lightcurve['Filter'].unique()}")
print(f"Time span: {lightcurve['Time (MJD)'].max() - lightcurve['Time (MJD)'].min():.2f} days")

# Plot the lightcurve
fig, ax = plt.subplots(figsize=(12, 6))

filters = lightcurve['Filter'].unique()
colors = {'u': 'purple', 'g': 'green', 'r': 'red', 'i': 'orange', 'z': 'brown', 'y': 'black'}

for filt in filters:
    filt_data = lightcurve[lightcurve['Filter'] == filt]
    ax.errorbar(filt_data['Time (MJD)'], filt_data['Flux'], 
                yerr=filt_data['Flux_err'], 
                fmt='o', label=f'Filter {filt}', 
                color=colors.get(filt, 'gray'),
                alpha=0.7, markersize=4)

ax.set_xlabel('Time (MJD)', fontsize=12)
ax.set_ylabel('Flux', fontsize=12)
ax.set_title(f'Lightcurve for {example_object}', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('example_lightcurve.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved as 'example_lightcurve.png'")

# Show dataset summary
print("\n" + "="*70)
print("DATASET SUMMARY")
print("="*70)
summary = loader.get_dataset_summary()
for key, value in summary.items():
    print(f"{key}: {value}")

