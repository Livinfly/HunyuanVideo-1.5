"""Exercise the worker on CPU doubles without interrupting live GPU experiments."""
from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from common import HERE, ROOT, read_json, write_json
import worker


class WorkerTests(unittest.TestCase):
    def test_default_state_and_optional_video_validation(self):
        # Load the real upstream defaults/initializer without importing its GPU package.
        spec = importlib.util.spec_from_file_location(
            'benchmark_test_infer_state', ROOT / 'hyvideo/commons/infer_state.py')
        upstream = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {spec.name: upstream}):
            spec.loader.exec_module(upstream)
        expected_state = asdict(upstream.InferState(sage_blocks_range=list(range(54)), no_cache_block_id=[53]))

        for validate in [False, True]:
            with self.subTest(validate_video=validate), tempfile.TemporaryDirectory() as folder:
                output = Path(folder)
                plan = dict(generation=read_json(HERE / 'baseline.json')['generation'],
                            model_path='weights', transformer_version='720p_i2v_distilled_sparse',
                            image_path='reference.png', prompt='test', modes=['e2e'],
                            warmup=1, repeats=2, validate_video=validate)
                write_json(output / 'plan.json', plan)
                torch, dist, pipe, reader = [MagicMock() for _ in range(4)]
                torch.cuda.get_device_properties.return_value = SimpleNamespace(
                    name='test GPU', major=9, minor=0, total_memory=2**30)
                torch.cuda.max_memory_allocated.return_value = 1
                torch.cuda.max_memory_reserved.return_value = 2
                torch.cuda.OutOfMemoryError = MemoryError
                torch.version.cuda = 'test'
                torch.backends.cudnn.version.return_value = 9000
                torch.get_num_threads.return_value = 1
                torch.no_grad.side_effect = nullcontext
                torch.isfinite.return_value.all.return_value = True
                dist.get_rank.return_value = 0
                dist.get_world_size.return_value = 1
                dist.all_gather_object.side_effect = lambda dest, value: dest.__setitem__(0, value)
                pipe.return_value = SimpleNamespace(videos=SimpleNamespace(shape=[1, 3, 121, 16, 16]))
                reader.__enter__.return_value = reader
                reader.get_meta_data.return_value = dict(size=[16, 16], fps=24)
                reader.count_frames.return_value = 121
                official = SimpleNamespace(torch=torch, dist=dist, save_video=MagicMock(),
                    initialize_infer_state=upstream.initialize_infer_state,
                    HunyuanVideo_1_5_Pipeline=MagicMock(), imageio=MagicMock())
                official.HunyuanVideo_1_5_Pipeline.create_pipeline.return_value = pipe
                official.imageio.get_reader.return_value = reader
                modules = {'generate': official, 'hyvideo.commons.infer_state': upstream}
                argv = ['worker.py', '--plan', str(output / 'plan.json'), '--output', str(output), '--offload', 'off']
                with patch.dict(sys.modules, modules), patch.object(sys, 'argv', argv), \
                        patch.object(worker, 'sha256', return_value='test-hash') as digest, \
                        patch.object(worker, 'DiTEvents', side_effect=AssertionError('e2e allocated events')), \
                        redirect_stdout(io.StringIO()):
                    worker.main()
                actual_state = pipe.apply_infer_optimization.call_args.kwargs['infer_state']
                self.assertEqual(asdict(actual_state), expected_state)
                self.assertEqual(official.save_video.call_count, 3)
                self.assertEqual(official.imageio.get_reader.call_count, 3 if validate else 0)
                self.assertEqual(digest.call_count, 3 if validate else 0)
                torch.cuda.Event.assert_not_called()
                pipe.transformer.register_forward_pre_hook.assert_not_called()
                metrics = read_json(output / 'metrics.json')
                self.assertEqual(metrics['status'], 'complete')
                for record in metrics['records']:
                    self.assertEqual('video_sha256' in record, validate)
                    self.assertEqual(record['per_rank'][0]['distributed_e2e_s'], record['distributed_e2e_s'])
                    self.assertIsNone(record['per_rank'][0]['dit_step_ms'])


if __name__ == '__main__':
    unittest.main()
