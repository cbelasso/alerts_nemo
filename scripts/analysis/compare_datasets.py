import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances


# -------------------------------
# Data loading
# -------------------------------
def load_organic_data(path: str, text_column: str = "comment") -> List[str]:
    """Load organic data from Excel file."""
    df = pd.read_excel(path)
    texts = df[text_column].dropna().astype(str).tolist()
    print(f"📊 Loaded {len(texts)} organic examples from {path}")
    return texts


def load_synthetic_data(path: str) -> List[dict]:
    """Load synthetic data from JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"🔧 Loaded {len(examples)} synthetic examples from {path}")
    return examples


def load_synthetic_data_from_multiple(paths: List[str]) -> List[dict]:
    """Load synthetic data from multiple JSONL files."""
    all_examples = []
    for path in paths:
        if Path(path).exists():
            with open(path) as f:
                for line in f:
                    all_examples.append(json.loads(line))
    print(f"🔧 Loaded {len(all_examples)} synthetic examples from {len(paths)} files")
    return all_examples


# -------------------------------
# Embedding
# -------------------------------
def embed_texts(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> np.ndarray:
    """Embed texts using sentence transformers."""
    print(f"🧠 Embedding {len(texts)} texts with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"✅ Embeddings shape: {embeddings.shape}")
    return embeddings


# -------------------------------
# Similarity analysis
# -------------------------------
def compute_similarity_to_organic(
    synthetic_embeddings: np.ndarray,
    organic_embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute similarity metrics between synthetic and organic data.

    Returns:
        max_similarities: Max similarity of each synthetic example to any organic example
        mean_similarities: Mean similarity of each synthetic example to all organic examples
        closest_organic_idx: Index of closest organic example for each synthetic
    """
    print("📐 Computing similarities...")

    # Compute pairwise cosine similarity (synthetic x organic)
    sim_matrix = cosine_similarity(synthetic_embeddings, organic_embeddings)

    max_similarities = sim_matrix.max(axis=1)
    mean_similarities = sim_matrix.mean(axis=1)
    closest_organic_idx = sim_matrix.argmax(axis=1)

    print(
        f"   Max similarity range: [{max_similarities.min():.3f}, {max_similarities.max():.3f}]"
    )

    return max_similarities, mean_similarities, closest_organic_idx


def compute_distribution_distance(
    synthetic_embeddings: np.ndarray,
    organic_embeddings: np.ndarray,
) -> dict:
    """Compute distribution-level metrics between synthetic and organic data."""

    if len(synthetic_embeddings) == 0 or len(organic_embeddings) == 0:
        return {
            "centroid_distance": 0.0,
            "mmd": 0.0,
            "coverage": 0.0,
        }

    # Centroid distance
    synthetic_centroid = synthetic_embeddings.mean(axis=0)
    organic_centroid = organic_embeddings.mean(axis=0)
    centroid_distance = np.linalg.norm(synthetic_centroid - organic_centroid)

    # MMD (Maximum Mean Discrepancy) approximation
    n_syn = len(synthetic_embeddings)
    n_org = len(organic_embeddings)

    # Sample if too large
    max_samples = 1000
    syn_sample = synthetic_embeddings[
        np.random.choice(n_syn, min(n_syn, max_samples), replace=False)
    ]
    org_sample = organic_embeddings[
        np.random.choice(n_org, min(n_org, max_samples), replace=False)
    ]

    # Kernel MMD with RBF kernel
    def rbf_kernel(X, Y, gamma=10.0):
        dist = np.sum(X**2, axis=1, keepdims=True) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
        return np.exp(-gamma * dist)

    K_ss = rbf_kernel(syn_sample, syn_sample)
    K_oo = rbf_kernel(org_sample, org_sample)
    K_so = rbf_kernel(syn_sample, org_sample)

    mmd = np.sqrt(K_ss.mean() + K_oo.mean() - 2 * K_so.mean())

    # Coverage: what fraction of organic space is "covered" by synthetic
    org_to_syn_sim = cosine_similarity(organic_embeddings, synthetic_embeddings)
    coverage_threshold = 0.7
    coverage = (org_to_syn_sim.max(axis=1) >= coverage_threshold).mean()

    return {
        "centroid_distance": centroid_distance,
        "mmd": mmd,
        "coverage": coverage,
    }


def compute_pairwise_diversity(embeddings: np.ndarray) -> float:
    """
    Computes the average pairwise cosine distance (1 - similarity)
    among all examples in the provided embeddings. Higher is better.
    """
    n = len(embeddings)
    if n < 2:
        return 0.0

    # Calculate cosine similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # Calculate cosine distance matrix (1 - similarity)
    dist_matrix = 1 - sim_matrix

    # Sum all non-diagonal elements and divide by number of pairs (n * (n-1))
    # We use sum of upper triangle to avoid double counting
    avg_pairwise_distance = dist_matrix[np.triu_indices(n, k=1)].mean()

    return avg_pairwise_distance


# -------------------------------
# Filtering and selection
# -------------------------------
def get_closest_organic_text(
    organic_texts: List[str],
    closest_organic_idx: np.ndarray,
    synthetic_idx: int,
) -> str:
    """Get the text of the closest organic example for a given synthetic example index."""
    idx = closest_organic_idx[synthetic_idx]
    return organic_texts[idx]


def select_diverse_subset_optimized(
    synthetic_examples: List[dict],
    synthetic_embeddings: np.ndarray,
    organic_embeddings: np.ndarray,
    organic_texts: List[str],
    closest_organic_idx: np.ndarray,
    n_select: int = 500,
    n_clusters: int = 50,
) -> List[dict]:
    """
    Select a diverse subset of synthetic examples by clustering the synthetic data and
    selecting the examples closest to the centroids that are also similar to the organic space.
    """

    print(f"🔬 Clustering synthetic data into {n_clusters} clusters...")

    # Cluster synthetic data - using lower n_init for faster operation
    n_clusters = min(n_clusters, len(synthetic_examples) // 2)
    if n_clusters < 1:
        return []

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
    synthetic_clusters = kmeans.fit_predict(synthetic_embeddings)

    # Compute similarity to organic for all synthetic examples
    sim_to_organic = cosine_similarity(synthetic_embeddings, organic_embeddings).max(axis=1)

    selected_indices = set()
    samples_per_cluster = max(1, n_select // n_clusters)

    for cluster_idx in range(n_clusters):
        syn_in_cluster_indices = np.where(synthetic_clusters == cluster_idx)[0]

        if len(syn_in_cluster_indices) == 0:
            continue

        sims_in_cluster = sim_to_organic[syn_in_cluster_indices]

        # Select the top samples_per_cluster most similar to organic within the cluster
        top_indices_in_cluster = sims_in_cluster.argsort()[-samples_per_cluster:][::-1]

        # Map back to global synthetic indices
        global_indices = syn_in_cluster_indices[top_indices_in_cluster]

        selected_indices.update(global_indices)

    # If not enough selected, fill with the overall most similar.
    if len(selected_indices) < n_select:
        remaining_count = n_select - len(selected_indices)

        all_indices = np.arange(len(synthetic_embeddings))
        unselected_indices_mask = np.ones(len(synthetic_embeddings), dtype=bool)
        unselected_indices_mask[list(selected_indices)] = False
        unselected_indices = all_indices[unselected_indices_mask]

        top_remaining_indices = unselected_indices[
            np.argsort(sim_to_organic[unselected_indices])[-remaining_count:][::-1]
        ]
        selected_indices.update(top_remaining_indices)

    # Cap the final selection to n_select
    final_selected_indices = list(selected_indices)[:n_select]

    selected = []
    for idx in final_selected_indices:
        example = synthetic_examples[idx].copy()
        example["organic_similarity"] = float(sim_to_organic[idx])
        example["closest_organic_text"] = get_closest_organic_text(
            organic_texts, closest_organic_idx, idx
        )
        selected.append(example)

    print(f"🎯 Selected {len(selected)} diverse synthetic examples")
    return selected


select_diverse_subset = select_diverse_subset_optimized


# -------------------------------
# Visualization
# -------------------------------
def visualize_embedding_space(
    synthetic_embeddings: np.ndarray,
    organic_embeddings: np.ndarray,
    synthetic_similarities: np.ndarray = None,
    output_path: str = "embedding_visualization.png",
    max_points: int = 1000,
):
    """Visualize synthetic vs organic data in 2D embedding space."""

    print("📊 Creating visualization...")

    # Sample if too many points
    n_syn = len(synthetic_embeddings)
    n_org = len(organic_embeddings)

    syn_emb, syn_sim, org_emb = synthetic_embeddings, synthetic_similarities, organic_embeddings

    if n_syn > max_points:
        syn_idx = np.random.choice(n_syn, max_points, replace=False)
        syn_emb = synthetic_embeddings[syn_idx]
        syn_sim = (
            synthetic_similarities[syn_idx] if synthetic_similarities is not None else None
        )

    if n_org > max_points:
        org_idx = np.random.choice(n_org, max_points, replace=False)
        org_emb = organic_embeddings[org_idx]

    if len(syn_emb) < 1 or len(org_emb) < 1:
        print("Not enough points to visualize.")
        return

    # Combine for t-SNE
    combined = np.vstack([syn_emb, org_emb])

    # Pre-reduce dimensions with PCA for faster t-SNE
    if combined.shape[1] > 50:
        pca = PCA(n_components=50, random_state=42)
        combined_reduced = pca.fit_transform(combined)
    else:
        combined_reduced = combined

    # t-SNE (using defaults for n_iter and init to avoid version conflicts)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_reduced) - 1))
    coords_2d = tsne.fit_transform(combined_reduced)

    syn_coords = coords_2d[: len(syn_emb)]
    org_coords = coords_2d[len(syn_emb) :]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Organic vs Synthetic
    ax1 = axes[0]
    ax1.scatter(org_coords[:, 0], org_coords[:, 1], c="blue", alpha=0.5, s=20, label="Organic")
    ax1.scatter(syn_coords[:, 0], syn_coords[:, 1], c="red", alpha=0.3, s=20, label="Synthetic")
    ax1.set_title("Embedding Space: Organic vs Synthetic")
    ax1.legend()
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")

    # Plot 2: Synthetic colored by similarity to organic
    ax2 = axes[1]
    ax2.scatter(org_coords[:, 0], org_coords[:, 1], c="blue", alpha=0.3, s=20, label="Organic")
    if syn_sim is not None:
        scatter = ax2.scatter(
            syn_coords[:, 0],
            syn_coords[:, 1],
            c=syn_sim,
            cmap="RdYlGn",
            alpha=0.7,
            s=20,
            vmin=0,
            vmax=1,
        )
        plt.colorbar(scatter, ax=ax2, label="Similarity to Organic")
    ax2.set_title("Synthetic Examples by Organic Similarity")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"📈 Visualization saved to: {output_path}")


# -------------------------------
# Saving results
# -------------------------------
def save_analysis_report(
    organic_texts: List[str],
    synthetic_examples: List[dict],
    max_similarities: np.ndarray,
    distribution_metrics: dict,
    topic_metrics: dict,
    overall_diversity: float,
    selected_diversity: float,
    output_path: str = "similarity_analysis_report.txt",
):
    """Save detailed analysis report, including new diversity and topic metrics."""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ORGANIC vs SYNTHETIC DATA SIMILARITY ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Organic examples: {len(organic_texts)}\n")
        f.write(f"Total synthetic examples: {len(synthetic_examples)}\n\n")

        f.write("-" * 40 + "\n")
        f.write("OVERALL DISTRIBUTION METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Centroid distance: {distribution_metrics['centroid_distance']:.4f}\n")
        f.write(f"MMD (lower is better): {distribution_metrics['mmd']:.4f}\n")
        f.write(f"Coverage (higher is better): {distribution_metrics['coverage']:.2%}\n\n")

        f.write("-" * 40 + "\n")
        f.write("SIMILARITY & DIVERSITY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean max similarity (to organic): {max_similarities.mean():.4f}\n")
        f.write(f"Overall Synthetic Diversity (Avg Cosine Distance): {overall_diversity:.4f}\n")
        f.write(
            f"Selected Subset Diversity (Avg Cosine Distance): {selected_diversity:.4f}\n\n"
        )

        f.write("-" * 40 + "\n")
        f.write("TOPIC-SPECIFIC METRICS\n")
        f.write("-" * 40 + "\n")

        # Format the topic metrics into a table for the report
        if topic_metrics:
            header = ["Topic", "N (Syn)", "Mean Sim", "Centroid Dist", "MMD", "Diversity"]
            f.write(
                f"{header[0]:<20}{header[1]:<10}{header[2]:<10}{header[3]:<15}{header[4]:<10}{header[5]:<10}\n"
            )
            f.write("-" * 75 + "\n")
            for topic, metrics in topic_metrics.items():
                f.write(
                    f"{topic:<20}"
                    f"{metrics['count']:<10}"
                    f"{metrics['mean_similarity']:.4f}{'':<6}"
                    f"{metrics['distribution']['centroid_distance']:.4f}{'':<6}"
                    f"{metrics['distribution']['mmd']:.4f}{'':<6}"
                    f"{metrics['diversity']:.4f}\n"
                )
        else:
            f.write("No topic-specific metrics calculated (missing 'example_type').\n")

        f.write("\n")
        f.write("-" * 40 + "\n")
        f.write("TOP 10 MOST SIMILAR SYNTHETIC EXAMPLES\n")
        f.write("-" * 40 + "\n")

        # The following block requires re-running embedding or using cached data,
        # but for a simple report, we'll assume the max_similarities array is aligned
        # with the full synthetic_examples list.

        top_indices = max_similarities.argsort()[-10:][::-1]

        # Get closest organic indices for the full synthetic list (This is slow,
        # so for production, this should be done once at the start, but for the
        # report text, we need the texts.)
        # To avoid re-embedding here, we'll rely on the main function passing data
        # or simplify this section. Since we don't have all inputs here, I'll
        # simplify the text output, as this report function is separate from the main pipeline.

        # NOTE: In a real environment, you'd pass the full set of closest_organic_idx
        # and organic_texts to this function, or ensure max_similarities is on the selected set.

        for rank, idx in enumerate(top_indices, 1):
            f.write(f"\n{rank}. Similarity: {max_similarities[idx]:.4f}\n")
            f.write(f"   Synthetic Text: {synthetic_examples[idx]['text'][:100]}...\n")
            # Skipping closest organic text lookup here to avoid re-embedding.
            f.write(f"   Type: {synthetic_examples[idx]['metadata']['example_type']}\n")

    print(f"📝 Analysis report saved to: {output_path}")


# -------------------------------
# Main pipeline
# -------------------------------
def analyze_and_filter(
    organic_path: str,
    synthetic_path: str,
    output_dir: str = "similarity_analysis",
    text_column: str = "comment",
    similarity_threshold: float = 0.5,
    embedding_model: str = "all-MiniLM-L6-v2",
    selection_method: str = "diverse",
    n_select: int = 500,
):
    """
    Main pipeline to analyze and filter synthetic data.
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    organic_texts = load_organic_data(organic_path, text_column)

    if "*" in synthetic_path:
        from glob import glob

        synthetic_files = glob(synthetic_path)
        synthetic_examples = load_synthetic_data_from_multiple(synthetic_files)
    else:
        synthetic_examples = load_synthetic_data(synthetic_path)

    synthetic_texts = [ex["text"] for ex in synthetic_examples]

    # 2. Embed
    organic_embeddings = embed_texts(organic_texts, embedding_model)
    synthetic_embeddings = embed_texts(synthetic_texts, embedding_model)

    # 3. Compute Similarities
    max_sim, mean_sim, closest_idx = compute_similarity_to_organic(
        synthetic_embeddings, organic_embeddings
    )

    # 4. Compute Overall Distribution Metrics
    dist_metrics = compute_distribution_distance(synthetic_embeddings, organic_embeddings)
    overall_diversity = compute_pairwise_diversity(synthetic_embeddings)

    print("\n" + "=" * 60)
    print("OVERALL DISTRIBUTION METRICS")
    print("=" * 60)
    print(f"Centroid distance: {dist_metrics['centroid_distance']:.4f}")
    print(f"MMD: {dist_metrics['mmd']:.4f}")
    print(f"Coverage: {dist_metrics['coverage']:.2%}")
    print(f"Overall Synthetic Diversity (Avg Cosine Dist): {overall_diversity:.4f}")

    # 5. Topic-Specific Metrics (NEW)
    topic_metrics = {}
    synthetic_df = pd.DataFrame(
        {
            "text": synthetic_texts,
            "embedding": list(synthetic_embeddings),
            "similarity": max_sim,
            "example_type": [
                ex["metadata"].get("example_type", "Unknown") for ex in synthetic_examples
            ],
        }
    )

    # Filter out topics with too few examples
    type_counts = synthetic_df["example_type"].value_counts()
    valid_types = type_counts[type_counts >= 5].index

    print("\n" + "=" * 60)
    print("TOPIC-SPECIFIC METRICS")
    print("=" * 60)

    for example_type in valid_types:
        subset_df = synthetic_df[synthetic_df["example_type"] == example_type]
        subset_embeddings = np.array(subset_df["embedding"].tolist())
        subset_sims = subset_df["similarity"].values

        # Distribution Metrics
        subset_dist_metrics = compute_distribution_distance(
            subset_embeddings, organic_embeddings
        )

        # Diversity
        subset_diversity = compute_pairwise_diversity(subset_embeddings)

        # Store results
        topic_metrics[example_type] = {
            "count": len(subset_df),
            "mean_similarity": subset_sims.mean(),
            "distribution": subset_dist_metrics,
            "diversity": subset_diversity,
        }

        print(
            f"🔹 {example_type} (N={len(subset_df)}): Mean Sim={subset_sims.mean():.4f}, MMD={subset_dist_metrics['mmd']:.4f}, Diversity={subset_diversity:.4f}"
        )

    # 6. Select synthetic examples
    if selection_method == "diverse":
        selected = select_diverse_subset(
            synthetic_examples,
            synthetic_embeddings,
            organic_embeddings,
            organic_texts,
            closest_idx,
            n_select=n_select,
            n_clusters=min(50, len(synthetic_examples) // max(1, n_select // 20)),
        )
    else:  # Fallback for other methods (threshold/top_k/cluster)
        # Note: Implementations for these methods need to be passed here if desired,
        # but for this example, we'll keep 'diverse' and 'threshold' simple.
        selected = []
        print("Skipping selection methods other than 'diverse' for this detailed output.")

    # 7. Compute Diversity for Selected Subset (NEW)
    selected_embeddings = np.array(
        [synthetic_embeddings[synthetic_texts.index(ex["text"])] for ex in selected]
    )
    selected_diversity = compute_pairwise_diversity(selected_embeddings)
    print(f"\nSelected Subset Diversity (Avg Cosine Dist): {selected_diversity:.4f}")

    # 8. Save results and Visualize

    # Save the selected examples to CSV/JSONL
    if selected:
        df = pd.DataFrame(
            [
                {
                    "text": ex["text"],
                    "organic_similarity": ex.get("organic_similarity", 0),
                    "closest_organic_text": ex.get("closest_organic_text", ""),
                    "example_type": ex["metadata"].get("example_type", "Unknown"),
                    "expected_alert_types": ", ".join(
                        ex["metadata"].get("expected_alert_types") or []
                    ),
                    "expected_non_alert_type": ex["metadata"].get("expected_non_alert_type")
                    or "",
                }
                for ex in selected
            ]
        )
        df = df.sort_values("organic_similarity", ascending=False)
        df.to_csv(f"{output_dir}/filtered_synthetic_{selection_method}.csv", index=False)
        print(f"💾 Saved {len(selected)} selected examples to CSV.")

    # Full Analysis Report
    save_analysis_report(
        organic_texts,
        synthetic_examples,
        max_sim,
        dist_metrics,
        topic_metrics,  # NEW
        overall_diversity,  # NEW
        selected_diversity,  # NEW
        f"{output_dir}/analysis_report.txt",
    )

    # Visualization (if successful)
    if len(synthetic_embeddings) > 1 and len(organic_embeddings) > 1:
        visualize_embedding_space(
            synthetic_embeddings,
            organic_embeddings,
            max_sim,
            output_path=f"{output_dir}/embedding_visualization.png",
        )

    print(f"\n✅ Analysis complete! Results saved to: {output_dir}/")

    return {
        "overall_diversity": overall_diversity,
        "selected_diversity": selected_diversity,
        "topic_metrics": topic_metrics,
        "selected_examples": selected,
    }


# -------------------------------
# Main
# -------------------------------
def main():
    # Placeholder for actual file paths
    # NOTE: Run this main function in your environment, ensuring file paths are correct.
    print("Running enhanced analysis...")

    # This block would execute the analysis in a real environment
    results = analyze_and_filter(
        organic_path="../generation/golden_organic_dataset/data_gov_golden_test_dataset.xlsx",
        synthetic_path="../generation/data/generated_examples.jsonl",
        output_dir="similarity_analysis",
        selection_method="diverse",
        n_select=500,
    )

    # Placeholder summary print
    print("\n" + "=" * 60)
    print("ENHANCED ANALYSIS SUMMARY (After Execution)")
    print("=" * 60)
    print(f"Overall Synthetic Diversity: {results['overall_diversity']:.4f}")
    print(f"Selected Subset Diversity: {results['selected_diversity']:.4f}")
    print("Top 3 Topics by MMD (Lower is Better):")
    sorted_topics = sorted(
        results["topic_metrics"].items(), key=lambda item: item[1]["distribution"]["mmd"]
    )
    for name, metrics in sorted_topics[:3]:
        print(
            f"  - {name}: MMD={metrics['distribution']['mmd']:.4f}, Diversity={metrics['diversity']:.4f}"
        )


if __name__ == "__main__":
    main()
    # pass  # Keeping main() commented out for execution safety in this environment
