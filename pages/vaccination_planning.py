from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.cluster.hierarchy import dendrogram

from clustering.kmeans import fit_pca, hierarchical_labels, hierarchical_linkage, train_kmeans
from clustering.preprocessing import clean_data, scale_data

sns.set_style("whitegrid")


@st.cache_data
def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def render() -> None:
    st.title("💉 Vaccination Planning")
    st.caption("Compare hierarchical clustering with PCA + KMeans for regional vaccination strategy.")

    uploaded_file = st.file_uploader("Upload vaccination planning CSV", type=["csv"])
    if uploaded_file is None:
        st.warning("Please upload a CSV file to continue.")
        return

    df = load_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    cleaned = clean_data(df)
    if cleaned.empty:
        st.error("No numeric columns found after cleaning. Please upload a dataset with numeric features.")
        return

    cols = cleaned.columns.tolist()
    selected = st.multiselect(
        "Select features for analysis",
        options=cols,
        default=cols[: min(5, len(cols))],
    )
    if len(selected) < 2:
        st.info("Select at least 2 features.")
        return

    X = cleaned[selected]
    X_scaled, _ = scale_data(X)
    n_clusters = st.slider("Number of clusters", min_value=2, max_value=min(10, len(X)), value=min(3, len(X)))

    st.subheader("Hierarchical Clustering Dendrogram")
    linkage_matrix = hierarchical_linkage(X_scaled)
    fig_dendro, ax_dendro = plt.subplots(figsize=(10, 4))
    dendrogram(linkage_matrix, truncate_mode="level", p=5, ax=ax_dendro)
    ax_dendro.set_title("Hierarchical Dendrogram (truncated)")
    ax_dendro.set_xlabel("Samples")
    ax_dendro.set_ylabel("Distance")
    st.pyplot(fig_dendro)

    hier_labels = hierarchical_labels(X_scaled, n_clusters=n_clusters)
    pca_data, _ = fit_pca(X_scaled)
    hier_plot = pd.DataFrame(pca_data, columns=["PC1", "PC2"])
    hier_plot["Cluster"] = hier_labels.astype(str)

    st.subheader("Hierarchical Clusters in PCA Space")
    fig_hier, ax_hier = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=hier_plot, x="PC1", y="PC2", hue="Cluster", palette="tab10", ax=ax_hier, s=70)
    st.pyplot(fig_hier)

    kmeans_model = train_kmeans(X_scaled, n_clusters=n_clusters)
    km_labels = kmeans_model.labels_

    km_plot = pd.DataFrame(pca_data, columns=["PC1", "PC2"])
    km_plot["Cluster"] = km_labels.astype(str)

    st.subheader("PCA + KMeans Clusters")
    fig_km, ax_km = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=km_plot, x="PC1", y="PC2", hue="Cluster", palette="Set2", ax=ax_km, s=70)
    st.pyplot(fig_km)

    comparison_df = X.copy()
    comparison_df["Hierarchical_Cluster"] = hier_labels
    comparison_df["KMeans_Cluster"] = km_labels

    st.subheader("Cluster Comparison Table")
    st.dataframe(comparison_df.head(30), use_container_width=True)

    st.markdown(
        """
### Cluster Guidance
- Regions grouped into the same cluster have similar vaccination planning profiles.
- Use **Hierarchical clusters** for relationship interpretation.
- Use **PCA + KMeans clusters** for compact, scalable segmentation.
"""
    )

    st.download_button(
        "Download Comparison CSV",
        data=comparison_df.to_csv(index=False).encode("utf-8"),
        file_name="vaccination_cluster_comparison.csv",
        mime="text/csv",
    )


render()
