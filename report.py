"""Aggregate from complete seed runs and generate the paper's output panels."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from matrix import PANELS, SEEDS, schedule

LABELS = dict(ppo='PPO', trpo='TRPO', raw6='Raw Basic JPI', sn1='SN-JPI E=1',
              sn3='SN-JPI E=3', sn6='SN-JPI E=6', sac_keep='SAC keep',
              sac_reset='SAC reset')
COLORS = dict(zip(LABELS, ['#285b8a', '#6d468c', '#a75625', '#7a8693',
                           '#258380', '#176243', '#bd8433', '#a24056']))


def csv_write(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_scores(root, protocol, task, method, seed):
    folder = root / 'raw' / f'{protocol}__{task.replace("/", "-")}__{method}__{seed}'
    if not (folder / 'complete').exists():
        return None
    updates, switch, _, episodes = schedule(protocol, task)
    rows = [json.loads(line) for line in (folder / 'evaluation.jsonl').read_text().splitlines()]
    if [(r['update'], r['episode']) for r in rows] != [
            (u, e) for u in range(1, updates+1) for e in range(episodes)]:
        raise ValueError(f'Invalid episode coverage in {folder}')
    if any(r['switched'] != (r['update'] >= switch) for r in rows):
        raise ValueError(f'Invalid switch state in {folder}')
    key = 'episode_return' if protocol == 'control' else 'length'
    return np.array([r[key] for r in rows]).reshape(updates, episodes).mean(axis=1)


def se(values, axis=0):
    return np.std(values, axis=axis, ddof=0) / np.sqrt(len(values))


def make_report(root):
    root = Path(root)
    (root / 'figures').mkdir(parents=True, exist_ok=True)
    (root / 'tables').mkdir(parents=True, exist_ok=True)
    with Path(__file__).with_name('paper_reference.csv').open() as f:
        refs = {(int(r['table']), r['task'], r['method']): r for r in csv.DictReader(f)}
    audit = []
    seed_rows = []
    for number, (protocol, tasks, methods) in PANELS.items():
        rows, series = [], {}
        for task in tasks:
            _, switch, _, _ = schedule(protocol, task)
            for method in methods:
                requested = SEEDS[:5 if number == 6 else 3]
                curves = []
                for seed in requested:
                    curve = load_scores(root, protocol, task, method, seed)
                    if curve is not None:
                        seed_rows.append(dict(table=number, task=task, method=method, seed=seed,
                                              final=curve[-1], post_switch=curve[switch-1:].mean()))
                        curves.append(curve)
                row = dict(task=task, method=method, seeds=len(curves), expected_seeds=len(requested),
                           final='', post_switch='', final_se='', post_switch_se='')
                if len(curves) == len(requested):
                    curves = np.array(curves)
                    series[task, method] = curves
                    final, post = curves[:, -1], curves[:, switch-1:].mean(axis=1)
                    row.update(final=final.mean(), post_switch=post.mean(), final_se=se(final), post_switch_se=se(post))
                rows.append(row)
                ref = refs[number, task, method]
                audit.append(dict(table=number, **row, paper_final=ref['paper_final'],
                                  paper_post_switch=ref['paper_post'],
                                  paper_final_se=ref['paper_final_se'], paper_post_se=ref['paper_post_se'],
                                  abs_final_error=abs(row['final']-float(ref['paper_final'])) if row['final'] != '' else '',
                                  abs_post_error=abs(row['post_switch']-float(ref['paper_post'])) if row['post_switch'] != '' else ''))
        csv_write(root / f'tables/table{number}.csv', rows)
        if number in (3, 5, 6, 7) and len(series) == len(tasks) * len(methods):
            figure(root, number, protocol, tasks, methods, series)
    csv_write(root / 'discrepancies.csv', audit)
    if seed_rows:
        csv_write(root / 'seed_summaries.csv', seed_rows)
    print(f'Complete table rows {sum(row["final"] != "" for row in audit)}/{len(audit)}')


def figure(root, number, protocol, tasks, methods, series):
    curves = number in (3, 5)
    fig_num = {3: 1, 5: 2, 6: 3, 7: 4}[number]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.5*len(tasks), 4), sharey=False)
    unit = 'Episode return' if protocol == 'control' else 'Survival length (steps)'
    for ax, task in zip(axes, tasks):
        _, switch, _, _ = schedule(protocol, task)
        width = .77 / len(methods)
        for index, method in enumerate(methods):
            values = series[task, method]
            if curves:
                x = np.arange(1, values.shape[1]+1)
                mean = values.mean(axis=0)
                err = se(values)
                ax.plot(x, mean, color=COLORS[method], label=LABELS[method])
                ax.fill_between(x, mean-err, mean+err, color=COLORS[method], alpha=.13)
            else:
                summaries = np.column_stack([values[:, -1], values[:, switch-1:].mean(axis=1)])
                ax.bar(np.arange(2) + width*(index-(len(methods)-1)/2), summaries.mean(axis=0),
                       width, yerr=se(summaries), color=COLORS[method], label=LABELS[method],
                       capsize=2)
        if curves:
            ax.axvline(switch, linewidth=1, linestyle='--', color='black')
            ax.set_xlabel('Training update')
        else:
            ax.set_xticks([0, 1], ['Final update', 'Post-switch mean'])
        ax.set_title(task.replace('/', ' '))
        ax.set_ylabel(unit)
        ax.grid(axis='y', alpha=.25)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, .09, 1, 1))
    fig.savefig(root / f'figures/figure{fig_num}.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results')
    make_report(parser.parse_args().results)
