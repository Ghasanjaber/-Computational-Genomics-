import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway
import matplotlib.pyplot as plt
import seaborn as sns

def run_ttest(v1_df, v2_df, probe_list=None):
    """
    Perform two-sample t-test between EPIC v1 and v2 methylation levels per probe.

    Parameters:
        v1_df (pd.DataFrame): EPIC v1 matrix (probes x samples)
        v2_df (pd.DataFrame): EPIC v2 matrix (probes x samples)
        probe_list (list): Optional subset of probes

    Returns:
        pd.DataFrame: t-statistic and p-values for each probe
    """
    print("Running two-sample t-test...")
    common_probes = list(set(v1_df.index) & set(v2_df.index))
    if probe_list:
        common_probes = list(set(common_probes) & set(probe_list))

    results = []

    for probe in common_probes:
        v1_vals = v1_df.loc[probe].dropna()
        v2_vals = v2_df.loc[probe].dropna()
        t_stat, p_val = ttest_ind(v1_vals, v2_vals, equal_var=True)
        results.append({'probe': probe, 't_stat': t_stat, 'p_val': p_val})

    return pd.DataFrame(results).set_index('probe')


def run_anova_by_group(df, metadata_df, group_col):
    """
    Run one-way ANOVA for methylation levels across groups (e.g., sex, age groups).

    Parameters:
        df (pd.DataFrame): methylation matrix (probes x samples)
        metadata_df (pd.DataFrame): sample metadata
        group_col (str): column name to group by (e.g., "Sex", "AgeGroup")

    Returns:
        pd.DataFrame: ANOVA F-statistics and p-values per probe
    """
    print(f"Running one-way ANOVA for group: {group_col}")
    grouped = metadata_df.groupby(group_col)
    probes = df.index
    results = []

    for probe in probes:
        try:
            group_values = [df.loc[probe, list(group[1]['SampleID'])].dropna()
                            for group in grouped]
            if len(group_values) >= 2:
                f_stat, p_val = f_oneway(*group_values)
                results.append({'probe': probe, 'F_stat': f_stat, 'p_val': p_val})
        except:
            continue

    return pd.DataFrame(results).set_index('probe')


def plot_significant_probes(results_df, alpha=0.05, title='Significant Probes'):
    """
    Plot histogram of p-values and highlight significant probes.

    Parameters:
        results_df (pd.DataFrame): t-test or ANOVA results
        alpha (float): significance level
        title (str): plot title
    """
    plt.figure(figsize=(8, 4))
    sns.histplot(results_df['p_val'], bins=50, kde=False, color='skyblue')
    plt.axvline(x=alpha, color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('p-value')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    sig_count = (results_df['p_val'] < alpha).sum()
    print(f"Number of significant probes (p < {alpha}): {sig_count}")


# Example usage (replace with real paths):
if __name__ == "__main__":
    # Load processed methylation data
    v1_df = pd.read_csv('GSE286313_MatrixProcessed_GPL21145.csv', index_col=0)
    v2_df = pd.read_csv('GSE286313_MatrixProcessed_GPL33022.csv', index_col=0)
    metadata = pd.read_csv('GSE286313_sample_metadata.csv')  # should have SampleID, Sex, AgeGroup

    # Run t-test
    ttest_results = run_ttest(v1_df, v2_df)
    ttest_results.to_csv('ttest_results.csv')
    plot_significant_probes(ttest_results, alpha=0.05, title='T-test: v1 vs v2')

    # Run ANOVA by sex
    anova_sex = run_anova_by_group(v1_df, metadata, group_col='Sex')
    anova_sex.to_csv('anova_sex_results.csv')
    plot_significant_probes(anova_sex, alpha=0.05, title='ANOVA by Sex (EPIC v1)')

    # Run ANOVA by age
    anova_age = run_anova_by_group(v1_df, metadata, group_col='AgeGroup')
    anova_age.to_csv('anova_age_results.csv')
    plot_significant_probes(anova_age, alpha=0.05, title='ANOVA by Age (EPIC v1)')
