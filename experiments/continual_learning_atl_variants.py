#!/usr/bin/env python3

import sys
from pathlib import Path
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent))

from brain_crossmodal_learner import BrainCrossModalLearner, ATLVariant, DEVICE
from synthetic_environment import create_stimulus


@dataclass
class TaskSpec:
    colors: List[str]
    shapes: List[str]
    size: str = 'small'


def make_pairs(task: TaskSpec, n_per_combo: int, noise_range: Tuple[float, float] = (0.0, 0.05)) -> List[Dict]:
    pairs: List[Dict] = []
    for color in task.colors:
        for shape in task.shapes:
            for _ in range(n_per_combo):
                noise = float(np.random.uniform(*noise_range))
                offset = (int(np.random.randint(-2, 3)), int(np.random.randint(-2, 3)))
                img = create_stimulus(shape=shape, color=color, size=task.size, noise=noise, offset=offset)
                label = f"{color} {shape}"
                pairs.append({'image': img, 'label': label, 'color': color, 'shape': shape})
    return pairs


def pretrain_cortices(brain: BrainCrossModalLearner, pairs: List[Dict], n_epochs: int = 10):
    vis_optimizer = torch.optim.Adam(brain.visual.parameters(), lr=1e-3)
    lang_optimizer = torch.optim.Adam(brain.language.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        np.random.shuffle(pairs)

        # Visual reconstruction
        for pair in pairs:
            vis_optimizer.zero_grad()
            img_t = torch.from_numpy(pair['image']).float()
            loss = brain.visual.reconstruction_loss(img_t)
            loss.backward()
            vis_optimizer.step()

        # Language alignment to visual features (simple cosine)
        for pair in pairs:
            with torch.no_grad():
                vis_feat = brain.visual(torch.from_numpy(pair['image']).float())
            lang_feat = brain.language(pair['label'])
            loss = 1 - F.cosine_similarity(lang_feat.unsqueeze(0), vis_feat.unsqueeze(0))
            lang_optimizer.zero_grad()
            loss.backward()
            lang_optimizer.step()


def freeze_cortices(brain: BrainCrossModalLearner):
    for p in brain.visual.parameters():
        p.requires_grad_(False)
    for p in brain.language.parameters():
        p.requires_grad_(False)


def train_task_atl_only(brain: BrainCrossModalLearner, pairs: List[Dict], n_epochs: int = 25):
    # Only ATL consolidation updates (cortices are frozen)
    for epoch in range(n_epochs):
        np.random.shuffle(pairs)
        for pair in pairs:
            brain.experience(
                torch.from_numpy(pair['image']).float(),
                pair['label'],
                consolidate=True,
                train_cortices=False,
            )


def build_prototype_label_map(
    brain: BrainCrossModalLearner,
    pairs: List[Dict],
    n_samples: int = 500,
) -> Dict[int, str]:
    """Map prototypes to labels via majority vote over a sample of pairs."""
    if len(pairs) == 0:
        return {}

    idxs = np.random.choice(len(pairs), size=min(n_samples, len(pairs)), replace=False)
    counts: Dict[int, Dict[str, int]] = {}

    with torch.no_grad():
        for idx in idxs:
            pair = pairs[int(idx)]
            img = torch.from_numpy(pair['image']).float().to(DEVICE)
            vis_feat = brain.visual(img)
            proto = brain.atl.activate(vis_feat, 'visual')[1]

            if proto not in counts:
                counts[proto] = {}
            counts[proto][pair['label']] = counts[proto].get(pair['label'], 0) + 1

    mapping: Dict[int, str] = {}
    for proto, label_counts in counts.items():
        mapping[proto] = max(label_counts.items(), key=lambda kv: kv[1])[0]
    return mapping


def label_accuracy(
    brain: BrainCrossModalLearner,
    pairs: List[Dict],
    proto_to_label: Dict[int, str],
    n_samples: int = 200,
) -> float:
    """Accuracy of image→prototype→label using a provided prototype→label mapping."""
    if len(pairs) == 0:
        return 0.0
    if len(proto_to_label) == 0:
        return 0.0

    idxs = np.random.choice(len(pairs), size=min(n_samples, len(pairs)), replace=False)
    correct = 0
    total = 0

    with torch.no_grad():
        for idx in idxs:
            pair = pairs[int(idx)]
            img = torch.from_numpy(pair['image']).float().to(DEVICE)
            vis_feat = brain.visual(img)
            proto = brain.atl.activate(vis_feat, 'visual')[1]
            pred_label = proto_to_label.get(proto)
            if pred_label is None:
                continue
            correct += int(pred_label == pair['label'])
            total += 1

    return correct / max(total, 1)


def run_single_seed(
    seed: int,
    atl_variant: ATLVariant,
    tasks: List[TaskSpec],
    pretrain_epochs: int,
    task_epochs: int,
    n_per_combo_pretrain: int,
    n_per_combo_task: int,
) -> Dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    brain = BrainCrossModalLearner(feature_dim=64, n_concepts=100, atl_variant=atl_variant)

    # Pretrain cortices on union of all tasks (to isolate ATL stability)
    pretrain_pairs: List[Dict] = []
    for task in tasks:
        pretrain_pairs.extend(make_pairs(task, n_per_combo=n_per_combo_pretrain))

    pretrain_cortices(brain, pretrain_pairs, n_epochs=pretrain_epochs)
    freeze_cortices(brain)

    # Prepare task datasets
    task_pairs = [make_pairs(t, n_per_combo=n_per_combo_task) for t in tasks]

    # Freeze prototype→label mappings right after each task is learned
    frozen_maps: List[Dict[int, str]] = []

    history: Dict[str, List[float]] = {
        'after_task_0': [],
        'after_task_1': [],
        'after_task_2': [],
    }

    # Train sequentially and evaluate binding on all seen tasks
    for i, pairs in enumerate(task_pairs):
        train_task_atl_only(brain, pairs, n_epochs=task_epochs)

        # Store mapping for the just-learned task (used later to quantify forgetting)
        frozen_maps.append(build_prototype_label_map(brain, pairs))

        key = f"after_task_{i}"
        for j in range(i + 1):
            # Evaluate task j using the mapping learned right after task j training
            acc = label_accuracy(brain, task_pairs[j], frozen_maps[j])
            history[key].append(acc)

    # Forgetting for Task 0: performance right after Task 0 vs after Task 2
    task0_after_task0 = history['after_task_0'][0]
    task0_after_task2 = history['after_task_2'][0]
    forgetting = task0_after_task0 - task0_after_task2

    return {
        'seed': seed,
        'atl_variant': str(atl_variant.value),
        'task0_after_task0': task0_after_task0,
        'task0_after_task2': task0_after_task2,
        'forgetting': forgetting,
        'history': history,
    }


def main():
    output_dir = Path('continual_learning_results')
    output_dir.mkdir(exist_ok=True)

    # 3 class-incremental tasks
    tasks = [
        TaskSpec(colors=['red', 'blue'], shapes=['circle', 'square']),
        TaskSpec(colors=['green', 'yellow'], shapes=['triangle', 'star']),
        TaskSpec(colors=['purple', 'orange'], shapes=['cross', 'diamond']),
    ]

    seeds = [1, 2, 3, 4, 5]
    variants = [ATLVariant.BASELINE, ATLVariant.MINIMAL, ATLVariant.EXPERIMENTAL]

    pretrain_epochs = 15
    task_epochs = 50
    n_per_combo_pretrain = 10
    n_per_combo_task = 20

    all_results: Dict[str, List[Dict]] = {v.value: [] for v in variants}

    for variant in variants:
        for seed in seeds:
            res = run_single_seed(
                seed=seed,
                atl_variant=variant,
                tasks=tasks,
                pretrain_epochs=pretrain_epochs,
                task_epochs=task_epochs,
                n_per_combo_pretrain=n_per_combo_pretrain,
                n_per_combo_task=n_per_combo_task,
            )
            all_results[variant.value].append(res)
            print(
                f"{variant.value} seed={seed}: forgetting={res['forgetting']:.3f} "
                f"(task0 {res['task0_after_task0']:.3f} -> {res['task0_after_task2']:.3f})"
            )

    # Aggregate
    summary: Dict[str, Dict[str, float]] = {}
    for v in variants:
        vals = [r['forgetting'] for r in all_results[v.value]]
        summary[v.value] = {
            'mean_forgetting': float(np.mean(vals)),
            'std_forgetting': float(np.std(vals)),
        }

    payload = {
        'config': {
            'seeds': seeds,
            'variants': [v.value for v in variants],
            'pretrain_epochs': pretrain_epochs,
            'task_epochs': task_epochs,
            'n_per_combo_pretrain': n_per_combo_pretrain,
            'n_per_combo_task': n_per_combo_task,
        },
        'summary': summary,
        'results': all_results,
    }

    out_path = output_dir / 'atl_variants_continual_learning.json'
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f"\nSaved results to {out_path}")


if __name__ == '__main__':
    main()
