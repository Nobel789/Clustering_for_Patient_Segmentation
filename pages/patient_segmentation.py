from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from clustering.kmeans import elbow_method, fit_pca, train_kmeans
from clustering.preprocessing import clean_data, scale_data

sns.set_style("whitegrid")


@st.cache_data
def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def cluster_explanations(clustered_df: pd.DataFrame, features: list[str]) -> list[str]:
    means = clustered_df.groupby("Cluster")[features].mean(numeric_only=True)
    explanations: list[str] = []
    for cluster_id, row in means.iterrows():
        top_feature = row.abs().sort_values(ascending=False).index[0]
        direction = "higher" if row[top_feature] >= 0 else "lower"
        explanations.append(
            f"Cluster {cluster_id}: shows relatively {direction} values in **{top_feature}** compared to other groups."
        )
    return explanations


def render() -> None:
    st.title("🧍 Patient Segmentation")
    st.caption("Upload patient data, select features, and segment into actionable groups.")

    uploaded_file = st.file_uploader("Upload patient CSV", type=["csv"])
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

    numeric_cols = cleaned.columns.tolist()
    selected_features = st.multiselect(
        "Select features for clustering",
        options=numeric_cols,
        default=numeric_cols[: min(4, len(numeric_cols))],
    )

    if len(selected_features) < 2:
        st.info("Select at least 2 features.")
        return

    model_df = cleaned[selected_features]
    scaled_data, _ = scale_data(model_df)

    col1, col2 = st.columns(2)
    with col1:
        max_k = st.slider("Max K for Elbow", min_value=2, max_value=min(15, len(model_df)), value=min(8, len(model_df)))
    with col2:
        k = st.slider("Choose K for KMeans", min_value=2, max_value=min(10, len(model_df)), value=min(3, len(model_df)))

    ks, inertias = elbow_method(scaled_data, max_k=max_k)
    fig_elbow, ax_elbow = plt.subplots(figsize=(7, 4))
    ax_elbow.plot(ks, inertias, marker="o", color="#1f77b4")
    ax_elbow.set_title("Elbow Method")
    ax_elbow.set_xlabel("Number of clusters (K)")
    ax_elbow.set_ylabel("Inertia")
    st.pyplot(fig_elbow)

    model = train_kmeans(scaled_data, n_clusters=k)
    labels = model.labels_

    pca_points, _ = fit_pca(scaled_data)
    plot_df = pd.DataFrame(pca_points, columns=["PC1", "PC2"])
    plot_df["Cluster"] = labels.astype(str)

    st.subheader("PCA 2D Cluster View")
    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="Cluster", palette="Set2", ax=ax_scatter, s=70)
    ax_scatter.set_title("Patient Clusters in PCA Space")
    st.pyplot(fig_scatter)

    output_df = model_df.copy()
    output_df["Cluster"] = labels

    st.subheader("Clustered Dataset")
    st.dataframe(output_df.head(30), use_container_width=True)

    st.subheader("Cluster Explanations")
    for text in cluster_explanations(output_df, selected_features):
        st.markdown(f"- {text}")

    st.download_button(
        label="Download Clustered CSV",
        data=output_df.to_csv(index=False).encode("utf-8"),
        file_name="patient_clusters.csv",
        mime="text/csv",
    )


render()
