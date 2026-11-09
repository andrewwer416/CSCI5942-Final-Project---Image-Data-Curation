"""Render every proposed near-duplicate pair for a train-only visual audit."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from vlm_data.io import write_json_atomic
from vlm_data.visual_analysis import build_duplicate_groups
from vlm_data.visual_embeddings import load_embedding_artifact
from vlm_data.visual_neighbors import load_neighbor_artifact


def main():
    root = Path(__file__).resolve().parents[1]
    cache = root / "data/interim/visual"
    output = root / "reports/data/duplicate-audit"
    output.mkdir(parents=True, exist_ok=True)
    _, embeddings, ids = load_embedding_artifact(
        cache / "openclip_vit_b16_datacomp_xl_s13b_b90k_train2017.meta.json"
    )
    _, similarities, neighbors = load_neighbor_artifact(cache / "clip_neighbors_k400.meta.json")
    report = {}
    for threshold in (0.98, 0.99, 0.995):
        groups = build_duplicate_groups(
            embeddings=embeddings,
            source_image_ids=ids,
            neighbor_image_ids=neighbors,
            neighbor_similarities=similarities,
            threshold=threshold,
        )
        pairs = list(groups.audit_pairs)
        pages = []
        for start in range(0, len(pairs), 12):
            page = Image.new("RGB", (1000, 6 * 190), "white")
            draw = ImageDraw.Draw(page)
            for offset, pair in enumerate(pairs[start : start + 12]):
                x, y = (offset % 2) * 500, (offset // 2) * 190
                for column, field in enumerate(("center_image_id", "member_image_id")):
                    image_id = pair[field]
                    with Image.open(root / f"data/raw/coco/train2017/{image_id:012d}.jpg") as image:
                        thumb = ImageOps.contain(image.convert("RGB"), (240, 155))
                        page.paste(thumb, (x + column * 250, y + 20))
                    draw.text((x + column * 250, y), str(image_id), fill="black")
                draw.text(
                    (x, y + 175),
                    f"pair {start + offset}: {pair['cosine_similarity']:.6f}",
                    fill="black",
                )
            path = output / f"threshold_{threshold}_page_{start // 12 + 1}.jpg"
            page.save(path, quality=92)
            pages.append(path.relative_to(root).as_posix())
        report[str(threshold)] = {
            "group_count": len(groups.groups),
            "excluded_count": int((~groups.candidate_allowed).sum()),
            "pairs": pairs,
            "pages": pages,
        }
    write_json_atomic(output / "pairs.json", report)
    print({key: {k: v for k, v in value.items() if k != "pairs"} for key, value in report.items()})


if __name__ == "__main__":
    main()
