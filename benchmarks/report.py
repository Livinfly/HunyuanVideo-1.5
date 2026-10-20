"""Regenerate a human-readable report using only saved JSON and Python's stdlib."""
import argparse
from pathlib import Path

from lifecycle import completed_groups

from common import review_measurements, FIELDS, FIXED_GENERATION, measurements_complete, read_json, summarize, write_json


def collect(suite, plan):
    completed = completed_groups(suite)
    rows = []
    for offload in plan['offload']:
        for gpus in plan['gpu_counts']:
            directory = suite / f'offload_{offload}/g{gpus}'
            effective_path = directory / 'effective_config.json'
            effective = read_json(effective_path) if effective_path.exists() else {}
            metrics_path = directory / 'metrics.json'
            metrics = read_json(metrics_path) if metrics_path.exists() else {}
            for mode in plan['modes']:
                records = [r for r in metrics.get('records', []) if r['mode'] == mode and not r['warmup']]
                warmups = [r for r in metrics.get('records', []) if r['mode'] == mode and r['warmup']]
                complete = str(directory.relative_to(suite)) in completed and measurements_complete(metrics, plan, mode)
                rows.append(dict(offload=offload, gpus=gpus, mode=mode,
                                 status='complete' if complete else 'pending_or_incomplete',
                                 count=len(records), warmup_count=len(warmups), warmup=summarize(warmups),
                                 effective=effective, output_shapes=[r['output_shape'] for r in records],
                                 records=records, **summarize(records)))
    # Never derive speedup from a partial single-GPU arm, or mix modes/offload policies.
    for row in rows:
        base = next((r for r in rows if r['gpus'] == 1 and r['offload'] == row['offload']
                     and r['mode'] == row['mode'] and r['status'] == 'complete'), None)
        for field, key in [(FIELDS[0], 'e2e_speedup'), (FIELDS[2], 'dit_speedup')]:
            if base and row['status'] == 'complete' and field in row and field in base:
                row[key] = base[field]['mean'] / row[field]['mean']
    return rows


def mismatches(plan, row, historical):
    reasons = []
    for key in ['source_commit', 'image_sha256', 'prompt']:
        if plan[key] != historical[key]:
            reasons.append(key)
    if plan['inference_source_modified']:
        reasons.append('inference source modified')
    for key, value in {**FIXED_GENERATION, **plan['generation']}.items():
        if key not in {'offloading', 'group_offloading'} and historical['generation'].get(key) != value:
            reasons.append(f'generation.{key}')
    if not plan.get('timing_schema_version') or plan.get('timing_schema_version') != historical.get('timing_schema_version'):
        reasons.append('timing_schema_version')
    effective = row['effective']
    if not effective:
        return reasons + ['effective config unavailable']
    if any(device['name'] != historical['gpu_name'] for device in effective['devices']):
        reasons.append('GPU model')
    if effective['group_offloading'] != historical['group_offloading']:
        reasons.append('group_offloading')
    for key, value in historical.get('packages', {}).items():
        if effective['packages'].get(key) != value:
            reasons.append(f'package.{key}')
    if any(shape != historical['output_shape'] for shape in row['output_shapes']):
        reasons.append('output_shape')
    return reasons


def fmt(row, field):
    if field not in row:
        return '—'
    stat = row[field]
    if stat['stdev'] is None:
        return f"{stat['mean']:.3f}（单次）"
    return f"{stat['mean']:.3f} ± {stat['stdev']:.3f}"


def ratio(row, key):
    return f"{row[key]:.3f}×" if key in row else '—'


def generate_report(suite, baseline=None):
    suite = Path(suite)
    plan = read_json(suite / 'plan.json')
    status = read_json(suite / 'status.json')
    historical = read_json(baseline) if baseline is not None else None
    manifest = read_json(suite / 'model_manifest.json')
    rows = collect(suite, plan)
    lines = ['# HunyuanVideo-1.5 推理性能报告', '',
             f"状态：**{status['status']}**；完成进程组 {len(status['completed'])}/{status['total']}。", '',
             'E2E：rank 0 从 pipeline 调用前到 MP4 保存返回，包含 TE/视觉编码、VAE、DiT、scheduler、搬运与编码写盘；排除初始化、加载、预热、验证和指标写盘。', '',
             'DiT：各 rank 完整 forward 的 CUDA event 时间先求和，再取最大值（max(sum)）；包括 SP/SSTA 及 event 间的等待。排除 scheduler 和整模块搬运，不等于完整去噪循环或纯 kernel 时间。sum(max) 与逐步原始值见 JSON。', '',
             '两种模式均不增加 pipeline 与保存视频之间的同步。`e2e` 请求期间没有 DiT hooks/events；`dit` 仅加入预分配、预初始化的 events，无逐步同步。distributed E2E 额外包含保存后的 barrier。', '',
             '## 模型、输入与实验配置', '',
             f"- 源码 commit：`{plan['source_commit']}`；推理源码有未提交改动：{plan['inference_source_modified']}。",
             f"- 模型：`{plan['transformer_version']}`；模型路径（本次运行）：`{plan['model_path']}`。",
             f"- Pipeline 默认参数（含 CFG / flow shift）：`{plan.get('pipeline_defaults', {})}`。",
             f"- 输入 SHA-256：`{plan['image_sha256']}`；图片路径（本次运行）：`{plan['image_path']}`。",
             f"- Prompt：{plan['prompt']}",
             f"- 每种计时模式完整预热 {plan['warmup']} 次、正式 {plan['repeats']} 次；同卡数/同 offload 复用 pipeline，模式顺序逐轮反转。",
             f"- 额外 MP4 解码检查与文件 hash：{plan.get('validate_video', True)}；张量形状/有限值始终检查，均在计时区外。",
             '- MP4：24 FPS；I2V 输出宽高由参考图 bucket 决定。权重文件仅记录大小；配置 SHA-256 和完整内容见 [model_manifest.json](model_manifest.json)。', '',
             '| 推理参数 | 值 |', '|---|---|']
    for key, value in {**FIXED_GENERATION, **plan['generation']}.items():
        lines.append(f'| {key} | `{value}` |')
    lines += ['', 'offloading 由矩阵 on/off 覆盖；off 时 group_offloading=false，on 时按上表设置/官方自动判断。SR=false；CFG/flow shift 来自原生 pipeline 版本默认值。', '']
    configs = manifest['configs']
    transformer = configs.get(f"transformer/{plan['transformer_version']}/config.json", {}).get('config', {})
    lines += [f"DiT 配置：heads={transformer.get('heads_num', transformer.get('num_attention_heads', '见配置'))}；"
              f"attn_mode=`{transformer.get('attn_mode')}`；attn_param=`{transformer.get('attn_param')}`。", '']
    effective_rows = [r for r in rows if r['effective']]
    if effective_rows:
        env = effective_rows[0]['effective']
        lines += [f"环境：Python {env['python'].split()[0]}；CUDA {env['cuda']}；cuDNN {env['cudnn']}；包版本 `{env['packages']}`。", '',
                  '| Offload | GPU 数 | GPU / 每卡 GiB | Group offload | 输出形状 |', '|---|---:|---|---|---|']
        seen = set()
        for row in effective_rows:
            key = (row['offload'], row['gpus'])
            if key in seen:
                continue
            seen.add(key)
            env = row['effective']
            devices = sorted({f"{d['name']} / {d['total_memory_gib']:.2f}" for d in env['devices']})
            lines.append(f"| {row['offload']} | {row['gpus']} | {', '.join(devices)} | {env['group_offloading']} | {row['output_shapes'][:1]} |")
    lines += ['', '## E2E、DiT 与多卡加速比', '',
              '单位秒，多次显示均值 ± 样本标准差；仅一次正式测量时标为单次，标准差未估计（JSON 为 null）。加速比 = 相同 offload、相同模式的完整单卡均值 / 当前均值；单卡未完成时留空。`e2e` 模式的 DiT 未测量。', '',
              '| Offload | GPU | 模式 | 已测次数 | E2E (s) | DiT max(sum) (s) | E2E 加速比 | DiT 加速比 |',
              '|---|---:|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['offload']} | {row['gpus']} | {row['mode']} | {row['count']}/{plan['repeats']} | {fmt(row, FIELDS[0])} | {fmt(row, FIELDS[2])} | {ratio(row, 'e2e_speedup')} | {ratio(row, 'dit_speedup')} |")
    if historical is not None:
        lines += ['', '## 与历史 baseline 对比', '', historical['description'], '',
                  '历史版本在 pipeline 返回后额外同步并在计时区内检查形状/步数；新版本移除了这些操作。差值表示复测差异，不能直接归因于优化。硬件、输入、生成参数、group offload、核心包版本或推理源码不匹配时不计算百分比；权重内容未逐字节验证。', '',
                  '| Offload | GPU | 模式 | 旧 E2E / DiT (s) | E2E 差异 | DiT 差异 | 可比性 |',
                  '|---|---:|---|---:|---:|---:|---|']
        for row in rows:
            old = next((r for r in historical['results'] if r['offload'] == row['offload'] and r['gpus'] == row['gpus']), None)
            if old is None:
                continue
            reasons = mismatches(plan, row, historical)
            delta = []
            for field in [FIELDS[0], FIELDS[2]]:
                valid = not reasons and row['status'] == 'complete' and field in row
                delta.append(f"{100 * (row[field]['mean'] / old[field]['mean'] - 1):+.3f}%" if valid else '—')
            row['historical_mismatches'] = reasons
            lines.append(f"| {row['offload']} | {row['gpus']} | {row['mode']} | {old[FIELDS[0]]['mean']:.3f} / {old[FIELDS[2]]['mean']:.3f} | {' | '.join(delta)} | {', '.join(reasons) if reasons else '参数匹配；计时边界差异见上文'} |")
    lines += ['', '## Warmup 与正式测量', '',
              '| Offload | GPU | 模式 | warmup E2E / DiT (s) | 正式 E2E / DiT (s) |',
              '|---|---:|---|---|---|']
    for row in rows:
        lines.append(f"| {row['offload']} | {row['gpus']} | {row['mode']} | {fmt(row['warmup'], FIELDS[0])} / {fmt(row['warmup'], FIELDS[2])} | {fmt(row, FIELDS[0])} / {fmt(row, FIELDS[2])} |")
    lines += ['', '## DiT 插桩对 E2E 的影响', '',
              '同时运行两种模式时才计算此对照。差异 = (dit E2E / e2e-only E2E − 1) × 100%；负值表示本次 dit 样本更快，不代表负开销。交替顺序降低顺序偏差，仍需结合重复波动判断；不宣称统计显著性。', '',
              '| Offload | GPU | dit E2E (s) | e2e-only (s) | 均值差异 | 每轮配对差异 |',
              '|---|---:|---:|---:|---:|---|']
    comparisons = []
    for row in rows:
        if row['mode'] != 'dit' or row['status'] != 'complete':
            continue
        other = next((r for r in rows if r['offload'] == row['offload'] and r['gpus'] == row['gpus']
                      and r['mode'] == 'e2e' and r['status'] == 'complete'), None)
        if other is None:
            continue
        paired = [100 * (a / b - 1) for a, b in zip(row[FIELDS[0]]['values'], other[FIELDS[0]]['values'])]
        difference = 100 * (row[FIELDS[0]]['mean'] / other[FIELDS[0]]['mean'] - 1)
        comparisons.append(dict(offload=row['offload'], gpus=row['gpus'], mean_difference_pct=difference,
                                paired_difference_pct=paired))
        values = ', '.join(f'{p:+.3f}%' for p in paired)
        lines.append(f"| {row['offload']} | {row['gpus']} | {fmt(row, FIELDS[0])} | {fmt(other, FIELDS[0])} | {difference:+.3f}% | {values} |")
    lines += ['', '## 逐次值与检查', '', '| Offload | GPU | 模式 | E2E (s) | DiT (s) |', '|---|---:|---|---|---|']
    for row in rows:
        values = [', '.join(f'{v:.3f}' for v in row.get(field, {}).get('values', [])) or '—' for field in [FIELDS[0], FIELDS[2]]]
        lines.append(f"| {row['offload']} | {row['gpus']} | {row['mode']} | {' | '.join(values)} |")
    lines += ['', '复查提示（CV > 2% 或连续三次变快且首尾下降 > 2%，不自动增加运行次数；单次测量无法评估重复波动）：', '']
    findings = review_measurements(rows)
    lines += [f'- {finding}' for finding in findings] or ['- 暂无。']
    failures = [r for r in status['completed'] if r['status'] != 'complete']
    lines += ['', f'失败/显存不足组：`{failures}`。未完成组不用于最终加速比或百分比比较。', '',
              '详细记录：[plan.json](plan.json)、[summary.json](summary.json)；各组的 `effective_config.json`、`metrics.json`、`inference.log` 和 MP4 保存在 `offload_{on,off}/gN/`。', '']
    from sparse_attention import report_lines
    lines += report_lines(rows)
    (suite / 'report.md').write_text('\n'.join(lines))
    write_json(suite / 'summary.json', dict(status=status['status'], plan=plan,
               results=[{k: v for k, v in row.items() if k not in {'records', 'effective'}} for row in rows],
               timing_comparisons=comparisons, review=findings, failures=failures))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', type=Path)
    parser.add_argument('--baseline', type=Path, help='Opt in to historical comparison using this reference JSON')
    args = parser.parse_args()
    generate_report(args.suite, baseline=args.baseline)
