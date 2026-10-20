"""One torchrun worker per GPU; model loading and reporting are outside timing."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

# Every group uses the suite's saved scripts, even if the checkout changes mid-run.
if __name__ == '__main__' and '--plan' in sys.argv:
    snapshot = Path(sys.argv[sys.argv.index('--plan') + 1]).resolve().parent / 'scripts/worker.py'
    if snapshot.is_file() and snapshot != Path(__file__).resolve():
        os.environ['HYVIDEO_SOURCE_ROOT'] = str(Path(__file__).resolve().parent.parent)
        os.execv(sys.executable, [sys.executable, '-u', str(snapshot), *sys.argv[1:]])

from common import ROOT, SCHEMA, aggregate, read_json, schedule, sha256, write_json
from timing import DiTEvents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--offload', choices=['on', 'off'], required=True)
    args = parser.parse_args()
    plan = read_json(args.plan)
    sys.path.insert(0, str(ROOT))
    # Official initialization sets up SP=WORLD_SIZE and LOCAL_RANK's CUDA device.
    import generate as official
    from hyvideo.commons.infer_state import InferState
    torch, dist = official.torch, official.dist
    rank = dist.get_rank()
    config = SimpleNamespace(**plan['generation'])
    config.offloading = args.offload == 'on'
    if not config.offloading:
        config.group_offloading = False
    records = []
    phase = 'load'
    timer = None

    def dump(name, data):
        if rank == 0:
            write_json(args.output / name, data)

    try:
        start = time.perf_counter()
        # Use upstream defaults for disabled optimizations; only its two CLI range
        # strings need conversion. Keep initialization of upstream global state.
        infer_state = official.initialize_infer_state(SimpleNamespace(
            **{**vars(InferState()), 'sage_blocks_range': '0-53', 'no_cache_block_id': '53'}))
        group = config.group_offloading
        if group is None:
            group = official.HunyuanVideo_1_5_Pipeline.get_offloading_config()['enable_group_offloading']
        device = torch.device('cpu' if config.offloading else 'cuda')
        pipe = official.HunyuanVideo_1_5_Pipeline.create_pipeline(
            pretrained_model_name_or_path=plan['model_path'],
            transformer_version=plan['transformer_version'], create_sr_pipeline=False,
            transformer_dtype=torch.bfloat16 if config.dtype == 'bf16' else torch.float32,
            device=device, transformer_init_device=torch.device('cpu') if group else device)
        pipe.apply_infer_optimization(infer_state=infer_state,
                                     enable_offloading=config.offloading,
                                     enable_group_offloading=group,
                                     overlap_group_offloading=config.overlap_group_offloading)
        torch.cuda.synchronize()
        dist.barrier()
        load_s = time.perf_counter() - start
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        local_device = dict(rank=rank, name=props.name, capability=[props.major, props.minor],
                            total_memory_gib=props.total_memory / 2**30)
        devices = [None] * dist.get_world_size()
        dist.all_gather_object(devices, local_device)
        packages = {}
        for name in ['torch', 'torchvision', 'diffusers', 'transformers', 'flash-attn',
                     'flex-block-attn', 'numpy', 'imageio', 'imageio-ffmpeg']:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        effective = dict(timing_schema_version=SCHEMA, generation=vars(config),
                         group_offloading=group, offload=args.offload,
                         world_size=dist.get_world_size(), devices=devices,
                         packages=packages, python=sys.version, cuda=torch.version.cuda,
                         cudnn=torch.backends.cudnn.version(), load_seconds_excluded=load_s,
                         cpu_threads=torch.get_num_threads(),
                         environment={key: os.environ.get(key) for key in
                                      ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS',
                                       'PYTORCH_CUDA_ALLOC_CONF', 'NCCL_ALGO', 'NCCL_PROTO']})
        dump('effective_config.json', effective)
        # No events are allocated at all for a standalone --modes e2e run.
        if 'dit' in plan['modes']:
            timer = DiTEvents(torch, config.num_inference_steps)
        requests = dict(enable_sr=False, prompt=plan['prompt'], reference_image=plan['image_path'],
                        aspect_ratio=config.aspect_ratio, num_inference_steps=config.num_inference_steps,
                        video_length=config.video_length, negative_prompt=config.negative_prompt,
                        seed=config.seed, output_type='pt', prompt_rewrite=False,
                        return_pre_sr_video=False)

        for mode, warmup, repeat in schedule(plan['modes'], plan['warmup'], plan['repeats']):
            phase = 'warmup' if warmup else 'repeat'
            tag = f'{mode}_{phase}_{repeat}'
            video_path = args.output / f'{tag}.mp4'
            dump('progress.json', dict(status='running', tag=tag))
            if mode == 'dit':
                timer.attach(pipe.transformer)
            # Preparation is outside E2E: drain previous work, align ranks, reset peaks.
            torch.cuda.synchronize()
            dist.barrier()
            torch.cuda.reset_peak_memory_stats()
            response_s = None
            with torch.no_grad():
                # E2E START. Same request body in both modes.
                started_at = time.perf_counter()
                output = pipe(**requests)
                # Official pipeline returns CPU video (blocking .cpu()). No added
                # intermediate CUDA sync, clock, validation, or serialization here.
                if rank == 0:
                    official.save_video(output.videos, str(video_path))
                    response_s = time.perf_counter() - started_at  # E2E END, MP4 saved.
                dist.barrier()
                distributed_s = time.perf_counter() - started_at

            # Everything below is OUTSIDE both E2E measurements.
            if mode == 'dit':
                timer.detach()
            torch.cuda.synchronize()
            steps = timer.elapsed_ms() if mode == 'dit' else None
            local = dict(sparse_attention=timer.sparse_result() if mode == 'dit' and plan.get('sparse_attention_timing') else None, rank=rank, response_e2e_s=response_s,
                         distributed_e2e_s=distributed_s, dit_step_ms=steps,
                         peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                         peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
            per_rank = [None] * dist.get_world_size()
            dist.all_gather_object(per_rank, local)
            record = aggregate(per_rank, mode=mode, tag=tag, warmup=warmup, repeat=repeat)
            if rank == 0:
                video = output.videos
                shape = list(video.shape)
                if len(shape) != 5 or shape[:3] != [1, 3, config.video_length]:
                    raise ValueError(f'Unexpected output shape: {shape}')
                if not torch.isfinite(video).all():
                    raise ValueError('Non-finite output')
                record.update(output_shape=shape, video_file=video_path.name)
                if plan['validate_video']:
                    with official.imageio.get_reader(str(video_path)) as reader:
                        metadata, frames = reader.get_meta_data(), reader.count_frames()
                    if (frames != config.video_length or tuple(metadata['size']) != (shape[4], shape[3])
                            or metadata['fps'] != 24):
                        raise ValueError(f'Invalid MP4: {frames}, {metadata}')
                    record.update(video_frames=frames, video_fps=metadata['fps'],
                                  video_sha256=sha256(video_path))
                records.append(record)
                dump('metrics.json', dict(timing_schema_version=SCHEMA, status='running', records=records))
                print(json.dumps({key: record[key] for key in
                                  ['tag', 'response_e2e_s', 'distributed_e2e_s']}), flush=True)
            del output
        dump('metrics.json', dict(timing_schema_version=SCHEMA, status='complete', records=records))
        dump('progress.json', dict(status='complete'))
    except Exception as error:
        status = 'memory_insufficient' if isinstance(error, torch.cuda.OutOfMemoryError) else 'failed'
        write_json(args.output / f'error_rank{rank}.json',
                   dict(status=status, phase=phase, traceback=traceback.format_exc()))
        raise
    finally:
        if timer is not None:
            timer.detach()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
