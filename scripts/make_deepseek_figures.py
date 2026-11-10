"""Draw DeepSeek comparison figures from the completed, paired report data."""
import json
import math
from pathlib import Path

import make_azure_figures as F

ROOT = Path(__file__).resolve().parents[1]
ROWS = [('main+FB+V', 'Main + repairs', F.NAVY),
        ('candF1+FB+V', 'candF1 + repairs', F.TEAL),
        ('azure_native+FB', 'GPT-6 Luna + FB', F.GOLD),
        ('azure_deepseek_native+FB', 'DeepSeek + FB', '#8055a1')]


def ceiling(values, minimum, step):
    return max(minimum, int(math.ceil((max(values) + step * .2) / step) * step))


def tradeoff(data):
    systems = data['systems']; n = data['coverage']['common']
    assert all(systems[k]['n'] == n for k, _, _ in ROWS)
    hidden = systems[ROWS[0][0]]['hidden_n']
    assert all(systems[k]['hidden_n'] == hidden for k, _, _ in ROWS)
    good_max = ceiling([100*systems[k]['supported_correct'] for k, _, _ in ROWS], 50, 10)
    bad_max = ceiling([100*systems[k]['unsupported_rate'] for k, _, _ in ROWS], 30, 10)
    desc = 'Supported correct rates and unsupported-answer rates for the four systems: ' + '; '.join(
        f"{label} {systems[k]['supported_correct']*100:.1f}% supported correct, "
        f"{systems[k]['unsupported']}/{hidden} unsupported" for k, label, _ in ROWS)
    svg = F.begin('DeepSeek, GPT-6 and Tiny Aya', desc, 390)
    svg += [F.text(30, 34, 'DeepSeek, GPT-6 and Tiny Aya', 23, F.INK, 700),
            F.text(30, 58, f'Shared round-5 sample · {n} episodes · score after the listed repairs', 13, F.MUTED),
            F.text(223, 100, 'Supported correct', 16, F.INK, 700),
            F.text(223, 121, 'Answerable cases · higher is better', 12, F.MUTED),
            F.text(570, 100, 'Unsupported answers', 16, F.INK, 700),
            F.text(570, 121, 'Hidden-evidence cases · lower is better', 12, F.MUTED)]
    F.axis(svg, 223, 255, 138, 315, good_max, (0, good_max//2, good_max))
    F.axis(svg, 570, 255, 138, 315, bad_max, (0, bad_max//2, bad_max))
    for i, (key, label, color) in enumerate(ROWS):
        y = 151 + i*41; row = systems[key]
        good, bad = 100*row['supported_correct'], 100*row['unsupported_rate']
        svg.append(F.text(205, y+17, label, 14, F.INK, 600, 'end'))
        for left, value, maximum in ((223, good, good_max), (570, bad, bad_max)):
            width = 255*value/maximum
            svg.append(f'<rect x="{left}" y="{y}" width="{width:.1f}" height="23" rx="4" fill="{color}"/>')
            svg.append(F.text(f'{left+width+7:.1f}', y+17, f'{value:.1f}%', 13, F.INK, 700))
        svg.append(F.text(852, y+17, f"{row['unsupported']}/{hidden}", 12, F.MUTED))
    svg += [F.text(30, 361, 'Adapters use a saved verifier; hosted models use fail-closed only. FB converts invalid actions to refusals.', 12, F.MUTED), '</svg>']
    return '\n'.join(svg)+'\n'


def cost(data):
    amount = {'main+FB+V':data['local_t4']['main+FB+V']['estimated_busy_gpu_usd_per_10000'],
              'candF1+FB+V':data['local_t4']['candF1+FB+V']['estimated_busy_gpu_usd_per_10000'],
              'azure_native+FB':data['gpt6_round5']['native']['estimated_usd_per_10000'],
              'azure_deepseek_native+FB':data['deepseek_round5']['native']['estimated_usd_per_10000']}
    maximum = ceiling(list(amount.values()), 8, 2)
    desc = 'Estimated cost for 10,000 episodes: ' + '; '.join(f'{label} ${amount[k]:.2f}' for k,label,_ in ROWS)
    svg = F.begin('Illustrative cost for 10,000 episodes', desc, 360)
    svg += [F.text(30, 34, 'Illustrative cost for 10,000 episodes', 23, F.INK, 700),
            F.text(30, 58, 'T4 busy time versus Azure token billing · USD · estimates, not an invoice', 13, F.MUTED)]
    F.axis(svg, 223, 390, 87, 264, maximum, (0, maximum//2, maximum), money=True)
    for i, (key, label, color) in enumerate(ROWS):
        y = 98+i*40; width = 390*amount[key]/maximum
        svg.append(F.text(205, y+17, label, 14, F.INK, 600, 'end'))
        svg.append(f'<rect x="223" y="{y}" width="{width:.1f}" height="23" rx="4" fill="{color}"/>')
        svg.append(F.text(f'{230+width:.1f}', y+17, f'${amount[key]:.2f}', 13, F.INK, 700))
    svg += [f'<rect x="676" y="90" width="214" height="130" rx="10" fill="#f1f7f6"/>',
            F.text(692, 118, 'T4 VM always on', 14, F.INK, 700),
            F.text(692, 151, f"${data['t4_always_730h_usd']:.2f}", 23, F.TEAL, 700),
            F.text(692, 174, 'per 730-hour month', 12, F.MUTED),
            F.text(692, 197, 'before other costs', 12, F.MUTED),
            F.text(30, 310, 'T4 estimate assumes perfect utilization; API estimates assume observed tokens and cache patterns.', 12, F.MUTED),
            F.text(30, 332, 'Storage, network, retrieval, taxes, and operations are excluded. Quality and model sizes differ.', 12, F.MUTED), '</svg>']
    return '\n'.join(svg)+'\n'


def main():
    results = json.loads((ROOT/'reports/deepseek_comparison.json').read_text(encoding='utf-8'))
    costs = json.loads((ROOT/'reports/deepseek_roi.json').read_text(encoding='utf-8'))
    F.FIGURES.mkdir(parents=True, exist_ok=True)
    for name, content in [('deepseek_tradeoff.svg', tradeoff(results)), ('deepseek_cost.svg', cost(costs))]:
        path = F.FIGURES/name
        path.write_text(content, encoding='utf-8', newline='\n')
        print(path)


if __name__ == '__main__':
    main()
