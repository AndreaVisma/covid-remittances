from pathlib import Path
import pandas as pd
import numpy as np

# Define paths matching your main script structure
ROOT = Path('C:/git-projects/covid-remittances')
OUT = ROOT / 'oct-26/outputs/synthetic-migrants-house-networks'
MMP_DIR = ROOT / 'data/Mesoamerican Migration Core Data SAV and DAT Files/DTA Files-20261004T141113Z-1-001/DTA Files'


def run_validation_checks():
    print("=== 1. Loading Synthetic Population and Diagnostics ===")
    migrants = pd.read_parquet(OUT / 'migrants.parquet')
    households = pd.read_parquet(OUT / 'households.parquet')
    diagnostics = pd.read_csv(OUT / 'distribution_checks.csv')
    audit = pd.read_csv(OUT / 'overlap_audit.csv')

    print(f"Loaded {len(migrants):,} synthetic migrants across {len(households):,} households.\n")

    print("=== 2. Tail Behavior & Outlier Check (Network Size) ===")
    net_sizes = households['network_size']
    print("Synthetic Network Size Summary:")
    print(net_sizes.describe(percentiles=[0.5, 0.9, 0.95, 0.99]))
    max_net = net_sizes.max()
    print(f"\nMaximum synthetic household network size found: {max_net}")
    if max_net > 150:
        print(
            "-> Note: High network sizes (>150) are present in the tail. Check 'overlap_audit.csv' to see if this is driven by extended kin counts or community-level pooling.")

    # Also compute base network size summary for comparison
    try:
        base_hh_audit = audit.groupby('mmp_household_id').agg({
            'reconciled_count': 'sum',
            'house_known_count': 'sum'
        }).reset_index()
        base_connected = base_hh_audit[base_hh_audit['reconciled_count'] > 0]

        print("\nBase Population Network Size Summary (Migrant-Connected Only):")
        print(base_connected['reconciled_count'].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))
        print(f"Maximum base household network size found: {base_connected['reconciled_count'].max()}")
    except Exception as e:
        print(f"Could not compute base summary: {e}")

    print("\n=== 3. Network Size Distribution Match (Base vs. Generated - Migrant-Connected Only) ===")
    try:
        if 'reconciled_count' in audit.columns:
            # Define bins for network size comparison
            bins = [-1, 0, 1, 2, 4, 10, 500]
            labels = ['0', '1', '2', '3-4', '5-9', '10+']

            # Synthetic distribution (already migrant-connected)
            synth_binned = pd.cut(households['network_size'], bins=bins, labels=labels).value_counts(
                normalize=True).sort_index()

            # Base distribution (filtered for migrant-connected households only)
            base_binned = pd.cut(base_connected['reconciled_count'], bins=bins, labels=labels).value_counts(
                normalize=True).sort_index()

            print("Synthetic Network Size Proportions:")
            print(synth_binned.apply(lambda x: f"{x * 100:.2f}%"))

            print("\nBase Data Network Size Proportions (Migrant-Connected Only):")
            print(base_binned.apply(lambda x: f"{x * 100:.2f}%"))

            comparison = pd.DataFrame({
                'Base_Share': base_binned,
                'Synthetic_Share': synth_binned
            })
            comparison['Diff_pp'] = (comparison['Synthetic_Share'] - comparison['Base_Share']) * 100
            print("\nComparison (Synthetic vs Base Migrant-Connected):")
            print(comparison.to_string(formatters={'Base_Share': '{:.2%}'.format, 'Synthetic_Share': '{:.2%}'.format,
                                                   'Diff_pp': '{:+.2f} pp'.format}))
        else:
            print("Network size columns not found in audit file for direct comparison.")
    except Exception as e:
        print(f"Could not compute base-to-generated network size comparison: {e}")

    print("\n=== 4. Marginal Distribution Checks (Target vs Synthetic) ===")
    high_diff = diagnostics[diagnostics['difference_pp'].abs() > 2.0]
    if not high_diff.empty:
        print("Categories with notable divergence (> 2 percentage points):")
        print(high_diff[['characteristic', 'category', 'target_share', 'synthetic_share', 'difference_pp']].to_string(
            index=False))
    else:
        print("All checked categories align closely with target reference shares.")

    print("\n=== 5. ENIGH Economic Match Quality ===")
    if 'match_distance' in households.columns:
        exact_matches = households['match_complete_exact'].mean() * 100
        print(f"Exact ENIGH matches: {exact_matches:.2f}% of households")
        print(f"Mean match distance score: {households['match_distance'].mean():.3f}")
    else:
        print("ENIGH matching was not enabled for this run.")


if __name__ == '__main__':
    run_validation_checks()