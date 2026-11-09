"""Explanatory caption proxies; none of these impose label quotas."""

from collections import Counter
import re


SCENES = frozenset(
    "beach street kitchen park field room road court snow water grass bathroom restaurant station sidewalk mountain ocean lake office bedroom forest airport".split()
)


def phrase_features(tagged):
    """POS/lexicon proxies, not dependency-parsed relationships or ground truth."""
    result = {
        name: set()
        for name in ("nouns", "verbs", "attributes", "quantities", "relations", "scenes")
    }
    for i, (word, tag) in enumerate(tagged):
        word = word.lower()
        if tag.startswith("NN"):
            result["nouns"].add(word)
        if tag.startswith("VB"):
            result["verbs"].add(word)
        if tag.startswith("JJ"):
            result["attributes"].add(word)
        if word in SCENES:
            result["scenes"].add(word)
        if tag == "CD":
            result["quantities"].add(" ".join(w.lower() for w, _ in tagged[i : i + 2]))
        if tag in ("IN", "TO") and i + 1 < len(tagged):
            result["relations"].add(" ".join(w.lower() for w, _ in tagged[i : i + 3]))
    return result


def summarize_features(features, selected_positions):
    source = Counter(x for row in features for x in row)
    selected = Counter(x for i in selected_positions for x in features[i])
    return {
        "source_types": len(source),
        "selected_types": len(selected),
        "type_coverage": len(selected) / max(1, len(source)),
        "source_incidence_mass_covered": sum(source[x] for x in selected)
        / max(1, sum(source.values())),
        "counts": {
            x: {"source_images": n, "selected_images": selected[x]}
            for x, n in sorted(source.items())
        },
    }


def nearest_distances(embeddings, selected_positions):
    import faiss
    import numpy as np

    x = np.ascontiguousarray(embeddings, dtype=np.float32)
    resources = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(resources, 0, faiss.IndexFlatIP(x.shape[1]))
    index.add(np.ascontiguousarray(x[selected_positions]))
    distances = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), 1024):
        scores, _ = index.search(x[start : start + 1024], 1)
        distances[start : start + len(scores)] = 1 - np.clip(scores[:, 0], -1, 1)
    return distances


def diagnostic_report(
    records, embeddings, dispersion, neighbor_scores, selected_ids, random_ids, nltk_dir
):
    import nltk
    import numpy as np

    nltk.data.path.insert(0, str(nltk_dir))
    selected_set, random_set = set(selected_ids), set(random_ids)
    selected = [i for i, r in enumerate(records) if r.image_id in selected_set]
    random = [i for i, r in enumerate(records) if r.image_id in random_set]
    distances = nearest_distances(embeddings, selected)
    random_distances = nearest_distances(embeddings, random)
    categories = {}
    for category in sorted({cat for r in records for cat in r.category_ids}):
        positions = [i for i, r in enumerate(records) if category in r.category_ids]
        chosen = [i for i in positions if records[i].image_id in selected_set]
        baseline = [i for i in positions if records[i].image_id in random_set]
        categories[str(category)] = {
            "source_count": len(positions),
            "selected_count": len(chosen),
            "random_count": len(baseline),
            "selection_rate": len(chosen) / len(positions),
            "selected_mean_distance": float(distances[positions].mean()),
            "selected_p95_distance": float(np.percentile(distances[positions], 95)),
            "random_mean_distance": float(random_distances[positions].mean()),
            "random_p95_distance": float(np.percentile(random_distances[positions], 95)),
        }
    density = np.asarray(neighbor_scores[:, 1:11]).mean(axis=1)
    order = np.lexsort((np.array([r.image_id for r in records]), density))
    extreme = set(order[: int(np.ceil(0.01 * len(records)))].tolist())
    captions = [re.findall(r"\w+|[^\w\s]", cap.text.lower()) for r in records for cap in r.captions]
    tags = nltk.pos_tag_sents(captions)
    features = []
    repeated = Counter(cap.text.strip().lower() for r in records for cap in r.captions)
    for start in range(0, len(tags), 5):
        row = {key: set() for key in phrase_features([])}
        for tagged in tags[start : start + 5]:
            for key, values in phrase_features(tagged).items():
                row[key].update(values)
        features.append(row)
    phrase_report = {
        key: {
            "selected": summarize_features([r[key] for r in features], selected),
            "random": summarize_features([r[key] for r in features], random),
        }
        for key in features[0]
    }
    groups = {}
    for name, positions in [("selected", selected), ("random", random)]:
        groups[name] = {
            "mean_caption_dispersion": float(np.asarray(dispersion)[positions].mean()),
            "extreme_outlier_count": len(set(positions) & extreme),
            "repeated_source_caption_occurrences": sum(
                repeated[cap.text.strip().lower()] > 1
                for i in positions
                for cap in records[i].captions
            ),
        }
        groups[name]["dispersion_quantiles"] = {
            str(q): float(np.percentile(np.asarray(dispersion)[positions], q))
            for q in (0, 25, 50, 75, 95, 100)
        }
        groups[name]["density_quantiles"] = {
            str(q): float(np.percentile(density[positions], q)) for q in (0, 1, 25, 50, 75, 95, 100)
        }
    extremes = [
        {
            "image_id": records[i].image_id,
            "density": float(density[i]),
            "selected": i in set(selected),
            "captions": [cap.text for cap in records[i].captions],
        }
        for i in order[:50]
    ]
    return {
        "category_allocation": categories,
        "phrase_proxies": phrase_report,
        "groups": groups,
        "source_dispersion_quantiles": {
            str(q): float(np.percentile(dispersion, q)) for q in (0, 25, 50, 75, 95, 100)
        },
        "extreme_examples": extremes,
        "limitations": "POS tags, fixed scene lexicon, and preposition windows are noisy descriptive proxies; no quotas or causal sample-need claims.",
    }
