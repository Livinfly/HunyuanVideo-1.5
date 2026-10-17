"""CPU regressions for measurement semantics, comparisons and relocation."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from common import HERE, ROOT, aggregate, measurements_complete, read_json, schedule, write_json
from report import collect, generate_report
from timing import DiTEvents


class TimingTests(unittest.TestCase):
    def test_rank_sum_and_step_max_are_distinct(self):
        ranks = [dict(rank=i, response_e2e_s=2 if i == 0 else None,
                      distributed_e2e_s=2.1 + i / 10,
                      peak_allocated_gib=1, peak_reserved_gib=2, dit_step_ms=steps)
                 for i, steps in enumerate([[900, 100], [100, 900]])]
        record = aggregate(list(reversed(ranks)), mode='dit', tag='test', warmup=False, repeat=1)
        self.assertEqual(record['dit_forward_sum_max_rank_s'], 1)
        self.assertEqual(record['dit_forward_step_max_sum_s'], 1.8)
        self.assertEqual(record['response_e2e_s'], 2)
        self.assertEqual(record['distributed_e2e_s'], 2.2)
        self.assertEqual(record['per_rank'], ranks)
        self.assertNotIn('dit_step_ms_per_rank', record)
        ranks[0]['dit_step_ms'] = [float('nan'), 1]
        with self.assertRaises(ValueError):
            aggregate(ranks, mode='dit', tag='test', warmup=False, repeat=1)

    def test_e2e_has_no_synthetic_dit_values(self):
        ranks = [dict(rank=0, response_e2e_s=1, distributed_e2e_s=1.1,
                      peak_allocated_gib=1, peak_reserved_gib=2, dit_step_ms=None)]
        record = aggregate(ranks, mode='e2e', tag='test', warmup=False, repeat=1)
        self.assertNotIn('dit_forward_sum_max_rank_s', record)

    def test_modes_get_equal_warmup_and_alternate(self):
        requests = list(schedule(['dit', 'e2e'], 1, 3))
        self.assertEqual([m for m, warm, _ in requests if not warm],
                         ['dit', 'e2e', 'e2e', 'dit', 'dit', 'e2e'])
        self.assertEqual([(m, warm) for m, warm, _ in requests[:2]],
                         [('dit', True), ('e2e', True)])

    def test_complete_requires_each_repeat_exactly_once(self):
        plan = dict(modes=['dit', 'e2e'], warmup=1, repeats=3)
        records = [dict(mode=m, warmup=w, repeat=r) for m, w, r in schedule(['dit', 'e2e'], 1, 3)]
        metrics = dict(status='complete', records=records)
        self.assertTrue(measurements_complete(metrics, plan))
        records[-1] = next(r.copy() for r in records if r['mode'] == 'e2e' and r['repeat'] == 2)
        self.assertFalse(measurements_complete(metrics, plan))
        self.assertTrue(measurements_complete(metrics, plan, 'dit'))
        self.assertFalse(measurements_complete(metrics, plan, 'e2e'))

    def test_hooks_do_not_sync_and_are_removed_before_control(self):
        calls = []

        class Event:
            def __init__(self, **kwargs):
                calls.append('allocate')

            def record(self):
                calls.append('record')

            def elapsed_time(self, other):
                return 1.25

        class Module:
            def __init__(self):
                self.pre, self.post = [], []

            def register_forward_pre_hook(self, hook, prepend=False):
                self.pre.insert(0, hook) if prepend else self.pre.append(hook)
                return SimpleNamespace(remove=lambda: self.pre.remove(hook))

            def register_forward_hook(self, hook):
                self.post.append(hook)
                return SimpleNamespace(remove=lambda: self.post.remove(hook))

            def forward(self):
                for hook in self.pre:
                    hook(self, ())
                calls.append('forward')
                for hook in self.post:
                    hook(self, (), None)

        torch = SimpleNamespace(cuda=SimpleNamespace(Event=Event, synchronize=lambda: calls.append('sync')))
        timer, module = DiTEvents(torch, 2), Module()
        module.register_forward_pre_hook(lambda *args: calls.append('native_pre'))
        module.register_forward_hook(lambda *args: calls.append('native_post'))
        self.assertEqual(calls.count('sync'), 1)
        calls.clear()
        timer.attach(module)
        module.forward()
        module.forward()
        timer.detach()
        self.assertEqual(calls, ['record', 'native_pre', 'forward', 'native_post', 'record'] * 2)
        self.assertEqual(timer.elapsed_ms(), [1.25, 1.25])
        calls.clear()
        module.forward()  # E2E control request: ordinary forward, no timing callbacks.
        self.assertEqual(calls, ['native_pre', 'forward', 'native_post'])


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.suite = Path(self.temp.name)
        sample = read_json(HERE / 'sample.json')
        self.plan = dict(source_commit='unknown', image_sha256=sample['image_sha256'],
                         prompt=sample['prompt'], inference_source_modified=False,
                         generation=read_json(HERE / 'baseline.json')['generation'], model_path='model', image_path='input.png',
                         transformer_version='720p_i2v_distilled_sparse', offload=['on', 'off'],
                         gpu_counts=[2, 1], modes=['dit', 'e2e'], repeats=2, warmup=1)
        write_json(self.suite / 'plan.json', self.plan)
        write_json(self.suite / 'status.json', dict(status='complete', total=4, completed=[
            dict(directory=f'offload_{offload}/g{n}', status='complete')
            for offload in ['on', 'off'] for n in [2, 1]]))
        write_json(self.suite / 'model_manifest.json', dict(configs={}))
        self.effective = dict(devices=[dict(name='NVIDIA H20', total_memory_gib=95)],
                              group_offloading=False, packages={},
                              python='3.11', cuda='12.4', cudnn=90100)
        for offload in self.plan['offload']:
            for gpus in self.plan['gpu_counts']:
                directory = self.suite / f'offload_{offload}/g{gpus}'
                directory.mkdir(parents=True)
                write_json(directory / 'effective_config.json', self.effective)
                records = []
                for mode, warmup, repeat in schedule(self.plan['modes'], 1, 2):
                    # Distinct scaling per policy and timing mode to catch mixed baselines.
                    base = 200 if offload == 'on' else 100
                    value = base / gpus + (5 if mode == 'dit' else 0)
                    record = dict(mode=mode, warmup=warmup, repeat=repeat,
                                  response_e2e_s=value, distributed_e2e_s=value + .1,
                                  output_shape=[1, 3, 121, 960, 960])
                    if mode == 'dit':
                        record['dit_forward_sum_max_rank_s'] = base / gpus - 20
                    records.append(record)
                write_json(directory / 'metrics.json', dict(status='complete', records=records))

    def tearDown(self):
        self.temp.cleanup()

    def test_speedups_and_no_partial_baseline(self):
        rows = collect(self.suite, self.plan)
        target = next(r for r in rows if r['gpus'] == 2 and r['offload'] == 'on' and r['mode'] == 'dit')
        self.assertAlmostEqual(target['e2e_speedup'], 205 / 105)
        self.assertAlmostEqual(target['dit_speedup'], 180 / 80)
        path = self.suite / 'offload_on/g1/metrics.json'
        data = read_json(path)
        data['status'] = 'running'
        write_json(path, data)
        rows = collect(self.suite, self.plan)
        self.assertTrue(all('e2e_speedup' not in r for r in rows if r['offload'] == 'on'))
        self.assertTrue(all('e2e_speedup' in r for r in rows if r['offload'] == 'off'))

    def test_report(self):
        generate_report(self.suite)
        summary = read_json(self.suite / 'summary.json')
        self.assertEqual(len(summary['timing_comparisons']), 4)
        text = (self.suite / 'report.md').read_text()
        self.assertIn('E2E 加速比', text)
        self.assertIn('DiT 加速比', text)
        self.assertIn('num_inference_steps', text)
        self.assertIn('## Warmup 与正式测量', text)
        self.assertIn('warmup', summary['results'][0])

        self.assertNotIn('## 与历史 baseline 对比', (self.suite / 'report.md').read_text())
        self.assertTrue(all('historical_mismatches' not in r for r in read_json(self.suite / 'summary.json')['results']))

    def test_worker_executes_saved_snapshot(self):
        scripts = self.suite / 'scripts'
        scripts.mkdir()
        (scripts / 'worker.py').write_text("import os; print(os.environ['HYVIDEO_SOURCE_ROOT'])\n")
        result = subprocess.run([sys.executable, str(HERE / 'worker.py'),
                                 '--plan', str(self.suite / 'plan.json')],
                                cwd=self.suite, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(ROOT))

    def test_relocated_dry_run_has_no_workspace_dependency(self):
        checkout = self.suite / 'another checkout'
        shutil.copytree(HERE, checkout / 'benchmarks', ignore=shutil.ignore_patterns('__pycache__', 'inputs'))
        result = subprocess.run([sys.executable, str(checkout / 'benchmarks/run.py'),
                                 '--model-path', 'weights', '--gpus', '2', '1', '--dry-run'],
                                cwd=self.suite, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(checkout), result.stdout)
        self.assertNotIn(str(HERE), result.stdout)


if __name__ == '__main__':
    unittest.main()
