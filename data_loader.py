import pandas as pd
import numpy as np

def load_methylation_data(v1_path, v2_path, nrows=1000):
    """
    Load and preprocess methylation data from EPIC v1 and v2 platforms.

    Parameters:
        v1_path (str): Path to the EPIC v1 CSV file.
        v2_path (str): Path to the EPIC v2 CSV file.
        nrows (int): Number of rows to load from each file for efficiency.

    Returns:
        df_v1_methylation (pd.DataFrame): Processed methylation matrix for EPIC v1.
        df_v2_methylation (pd.DataFrame): Processed methylation matrix for EPIC v2.
    """
    print(f"Loading EPIC v1 data from: {v1_path}")
    df_v1 = pd.read_csv(v1_path, nrows=nrows)

    print(f"Loading EPIC v2 data from: {v2_path}")
    df_v2 = pd.read_csv(v2_path, nrows=nrows)

    # Clean column names
    df_v1.columns = df_v1.columns.str.strip("\"")
    df_v2.columns = df_v2.columns.str.strip("\"")

    # Set index
    df_v1 = df_v1.set_index("ID_REF")
    df_v2 = df_v2.set_index("ID_REF")

    # Filter out detection p-values
    v1_meth_cols = [col for col in df_v1.columns if "Detection Pval" not in col]
    v2_meth_cols = [col for col in df_v2.columns if "Detection Pval" not in col]

    df_v1_meth = df_v1[v1_meth_cols]
    df_v2_meth = df_v2[v2_meth_cols]

    # Convert to numeric and handle NaNs
    df_v1_meth = df_v1_meth.apply(pd.to_numeric, errors='coerce').fillna(0)
    df_v2_meth = df_v2_meth.apply(pd.to_numeric, errors='coerce').fillna(0)

    return df_v1_meth, df_v2_meth

def get_common_probes(df_v1, df_v2):
    """
    Get the intersection of probe IDs between the two platforms.

    Returns:
        list of probe IDs common to both.
    """
    return df_v1.index.intersection(df_v2.index)

def get_top_variable_probes(df, n=100):
    """
    Select top `n` variable probes based on variance.

    Parameters:
        df (pd.DataFrame): Methylation beta matrix (probes x samples)
        n (int): Number of top variable probes to return.

    Returns:
        list: Probe names with highest variance.
    """
    variances = df.var(axis=1)
    return variances.sort_values(ascending=False).head(n).index.tolist()

if __name__ == "__main__":
    v1_file = "/home/hackathon_data/GSE286313_MatrixProcessed_GPL21145.csv"
    v2_file = "/home/hackathon_data/GSE286313_MatrixProcessed_GPL33022.csv"
    df_v1, df_v2 = load_methylation_data(v1_file, v2_file)
    common = get_common_probes(df_v1, df_v2)
    top_probes = get_top_variable_probes(df_v1.loc[common])
    print("Top variable probes:", top_probes[:10])
