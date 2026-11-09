"""Draw README comparison figures from the saved Azure reports, without a plotting dependency."""
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / 'docs/figures'
INK, MUTED, GRID = '#172b3a', '#546b79', '#dfe8e8'
NAVY, TEAL, GOLD = '#315a75', '#218a80', '#c27b34'
ROWS = [('main+FB+V', 'Main + repairs', NAVY),
        ('candF1+FB+V', 'candF1 + repairs', TEAL),
        ('azure_native+FB', 'Azure native + FB', GOLD)]


def text(x, y, label, size=14, color=INK, weight=400, anchor='start'):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(label))}</text>')


def begin(title, desc, height):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="920" height="{height}" viewBox="0 0 920 {height}" role="img" aria-labelledby="title desc" font-family="Segoe UI, Helvetica, Arial, sans-serif">',
            f'<title id="title">{escape(title)}</title>', f'<desc id="desc">{escape(desc)}</desc>',
            f'<rect width="920" height="{height}" fill="#ffffff"/>']


def axis(svg, left, width, y, bottom, maximum, ticks, money=False):
    for tick in ticks:
        x = left + width * tick / maximum
        svg.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y}" y2="{bottom}" stroke="{GRID}"/>')
        svg.append(text(f'{x:.1f}', bottom + 20, f'${tick}' if money else f'{tick}%', 12, MUTED, anchor='middle'))


def tradeoff(data):
    systems = data['systems']
    assert data['coverage']['common'] == 876
    assert all(systems[k]['hidden_n'] == 315 and systems[k]['n'] == 876 for k, _, _ in ROWS)
    desc = ('On 876 shared episodes, main has 27.6% supported correct answers and 25 of 315 unsupported; '
            'candF1 has 32.4% and 26 of 315; Azure native has 39.2% and 62 of 315. '
            'The local systems include a verifier; Azure does not.')
    svg = begin('Accuracy and unsupported answers', desc, 348)
    svg += [text(30, 34, 'Accuracy and unsupported answers', 23, INK, 700),
            text(30, 58, 'Shared round-5 sample · 876 episodes · score after the listed repairs', 13, MUTED),
            text(223, 100, 'Supported correct', 16, INK, 700),
            text(223, 121, 'Answerable cases · higher is better', 12, MUTED),
            text(570, 100, 'Unsupported answers', 16, INK, 700),
            text(570, 121, 'Hidden-evidence cases · lower is better', 12, MUTED)]
    axis(svg, 223, 255, 138, 278, 50, (0, 25, 50))
    axis(svg, 570, 255, 138, 278, 25, (0, 10, 20, 25))
    for i, (key, label, color) in enumerate(ROWS):
        y = 156 + i * 45
        row = systems[key]
        correct, unsupported = 100 * row['supported_correct'], 100 * row['unsupported_rate']
        assert 0 <= correct <= 50 and 0 <= unsupported <= 25
        svg.append(text(205, y + 17, label, 14, INK, 600, 'end'))
        for left, width, value, limit in ((223, 255, correct, 50), (570, 255, unsupported, 25)):
            length = width * value / limit
            svg.append(f'<rect x="{left}" y="{y}" width="{length:.1f}" height="23" rx="4" fill="{color}"/>')
            svg.append(text(f'{left + length + 7:.1f}', y + 17, f'{value:.1f}%', 13, INK, 700))
        svg.append(text(852, y + 17, f"{row['unsupported']}/315", 12, MUTED))
    svg += [text(30, 325, 'Local adapters use a saved verifier; Azure has fail-closed only. Results are not an equal-verifier test.', 12, MUTED), '</svg>']
    return '\n'.join(svg) + '\n'


def costs(data):
    amount = {
        'main+FB+V': data['local']['main+FB+V']['estimated_busy_gpu_usd_per_10000'],
        'candF1+FB+V': data['local']['candF1+FB+V']['estimated_busy_gpu_usd_per_10000'],
        'azure_native+FB': data['cloud']['native']['estimated_usd_per_10000'],
    }
    assert data['volume_per_month'] == 10_000 and all(0 <= v <= 8 for v in amount.values())
    desc = ('For 10,000 episodes, main costs an estimated $6.30 in busy GPU time, candF1 $5.99, '
            'and Azure native $2.66 in tokens. An always-on T4 VM costs an estimated $383.98 for 730 hours, '
            'excluding other costs. This is a scenario, not an invoice.')
    svg = begin('Illustrative cost for 10,000 episodes', desc, 335)
    svg += [text(30, 34, 'Illustrative cost for 10,000 episodes', 23, INK, 700),
            text(30, 58, 'T4 busy time versus Azure token billing · USD · estimates, not an invoice', 13, MUTED)]
    axis(svg, 223, 390, 87, 235, 8, (0, 2, 4, 6, 8), money=True)
    for i, (key, label, color) in enumerate(ROWS):
        y = 103 + i * 45
        width = 390 * amount[key] / 8
        svg.append(text(205, y + 17, label, 14, INK, 600, 'end'))
        svg.append(f'<rect x="223" y="{y}" width="{width:.1f}" height="23" rx="4" fill="{color}"/>')
        svg.append(text(f'{230 + width:.1f}', y + 17, f"${amount[key]:.2f}", 13, INK, 700))
    svg += [f'<rect x="676" y="90" width="214" height="130" rx="10" fill="#f1f7f6"/>',
            text(692, 118, 'T4 VM always on', 14, INK, 700),
            text(692, 151, f"${data['continuous_730h_gpu_usd']:.2f}", 23, TEAL, 700),
            text(692, 174, 'per 730-hour month', 12, MUTED),
            text(692, 197, 'before other costs', 12, MUTED),
            text(30, 283, 'GPU bars assume perfect utilization and no idle time; API assumes the recorded token and cache mix.', 12, MUTED),
            text(30, 305, 'Disk, network, retrieval, taxes, and operating costs are excluded. Model outputs and quality differ.', 12, MUTED),
            '</svg>']
    return '\n'.join(svg) + '\n'


def main():
    performance = json.loads((ROOT / 'reports/azure_round5_comparison.json').read_text(encoding='utf-8'))
    roi = json.loads((ROOT / 'reports/azure_roi.json').read_text(encoding='utf-8'))
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, content in [('azure_tradeoff.svg', tradeoff(performance)), ('azure_cost.svg', costs(roi))]:
        path = FIGURES / name
        path.write_text(content, encoding='utf-8', newline='\n')
        print(path)


if __name__ == '__main__':
    main()
