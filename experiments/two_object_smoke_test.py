#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/ahmed/Dokumente/Neuroscience/CHPL')

from synthetic_environment import generate_two_object_pairs


def main():
    colors = ['red', 'blue', 'green']
    shapes = ['circle', 'square']
    relations = ['above', 'left_of']

    pairs = generate_two_object_pairs(
        colors=colors,
        shapes=shapes,
        relations=relations,
        n_per_combination=1,
        canvas_size=56,
    )

    print(f"Generated {len(pairs)} two-object pairs")
    for p in pairs[:10]:
        print(f"{p.label} (rel={p.relation}) image={p.image.shape}")

    try:
        import matplotlib.pyplot as plt

        n = min(8, len(pairs))
        fig, axes = plt.subplots(2, max(1, n // 2), figsize=(12, 6))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for ax, pair in zip(axes, pairs[:n]):
            ax.imshow(pair.image)
            ax.set_title(pair.label, fontsize=9)
            ax.axis('off')

        plt.tight_layout()
        out = "two_object_samples.png"
        plt.savefig(out)
        print(f"Saved {out}")
        plt.close()
    except ImportError:
        print("matplotlib not available; skipping visualization")


if __name__ == '__main__':
    main()
