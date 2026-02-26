#!/usr/bin/env python3
"""Diagnostic: compare ||h||, ||W_V*h||, ||W_OV*h|| ratios for sink tokens."""
import torch, numpy as np, sys, json
sys.path.insert(0, '/home/kana5123/ATLASVLA')

from extract_attention import load_model_from_registry
from verify_attention_sinks import SinkVerificationHookManager, get_wov_matrix
from model_registry import get_model as registry_get_model
from visualize_text_attention import load_samples_from_cache
import config

def diagnose_model(model_name, device='cuda:0'):
    model_info = registry_get_model(model_name)
    processor, model, model_cfg = load_model_from_registry(model_name, device)

    samples = load_samples_from_cache(config.DATA_CACHE_DIR, n_samples=1)
    sample = samples[0]
    image = sample['image']
    instruction = sample['instruction']

    hook_mgr = SinkVerificationHookManager(model, model_cfg)
    hook_mgr.register_hooks()
    prompt = model_cfg.prompt_template.format(instruction=instruction)
    inputs = processor(images=image, text=prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        model(**inputs, output_attentions=True)
    hidden_states = hook_mgr.hidden_states
    hook_mgr.remove_hooks()

    # Detect sink position from condition B (max phi)
    n_layers = model_cfg.num_layers
    # Find position with highest avg phi across deep layers
    deep_layers = list(range(max(0, n_layers - 8), n_layers))

    # Compute phi for each position across deep layers
    phi_accum = None
    for li in deep_layers:
        h = hidden_states[li].cpu().float()
        rms = torch.sqrt((h ** 2).mean(dim=1)).clamp(min=1e-8)
        phi = (h.abs() / rms.unsqueeze(1)).max(dim=1).values.numpy()
        if phi_accum is None:
            phi_accum = phi
        else:
            phi_accum += phi
    sink_pos = int(np.argmax(phi_accum))

    print(f"\n{'='*80}")
    print(f"Model: {model_name}, Sink position: {sink_pos}")
    print(f"{'='*80}")
    print(f"{'Layer':<10} | {'||h|| ratio':>12} | {'||Wv*h|| ratio':>14} | {'||Wov*h|| ratio':>15} | {'Wv suppress':>12} | {'Wov suppress':>13} | Verdict")
    print('-' * 110)

    results = []
    for layer_idx in deep_layers:
        h = hidden_states[layer_idx].cpu().float()
        h_norms = torch.norm(h, dim=1)
        mask = torch.ones(h.shape[0], dtype=bool)
        mask[sink_pos] = False
        h_ratio = float(h_norms[sink_pos]) / float(h_norms[mask].mean() + 1e-8)

        try:
            v_weight, o_weight = get_wov_matrix(model, model_cfg, layer_idx)

            # W_V only
            v_all = h @ v_weight.T
            v_norms = torch.norm(v_all, dim=1)
            wv_ratio = float(v_norms[sink_pos]) / float(v_norms[mask].mean() + 1e-8)

            # W_OV = W_O @ W_V (full OV circuit)
            wov = (o_weight @ v_weight).float()
            ov_all = h @ wov.T
            ov_norms = torch.norm(ov_all, dim=1)
            wov_ratio = float(ov_norms[sink_pos]) / float(ov_norms[mask].mean() + 1e-8)

            wv_supp = (1 - wv_ratio / h_ratio) * 100 if h_ratio > 0 else 0
            wov_supp = (1 - wov_ratio / h_ratio) * 100 if h_ratio > 0 else 0

            if wov_ratio < 0.5:
                verdict = 'TRUE SINK'
            elif wov_ratio < 1.0:
                verdict = 'WEAK SINK'
            elif wov_ratio > 2.0:
                verdict = 'AGGREGATOR'
            else:
                verdict = 'NEUTRAL'

            print(f"layer_{layer_idx:02d}  | {h_ratio:11.1f}x | {wv_ratio:13.2f}x | {wov_ratio:14.2f}x | {wv_supp:10.1f}% | {wov_supp:11.1f}% | {verdict}")
            results.append({'layer': layer_idx, 'h_ratio': h_ratio, 'wv_ratio': wv_ratio, 'wov_ratio': wov_ratio, 'verdict': verdict})
        except Exception as e:
            print(f"layer_{layer_idx:02d}  | {h_ratio:11.1f}x | {'ERROR':>14} | {'ERROR':>15} | {'N/A':>12} | {'N/A':>13} | SKIP ({e})")
            results.append({'layer': layer_idx, 'h_ratio': h_ratio, 'wv_ratio': None, 'wov_ratio': None, 'verdict': 'ERROR'})

    # Summary
    wov_ratios = [r['wov_ratio'] for r in results if r['wov_ratio'] is not None]
    if wov_ratios:
        avg_wov = np.mean(wov_ratios)
        print(f"\nAvg W_OV ratio: {avg_wov:.2f}x")
        if avg_wov < 0.5:
            print("=> CONCLUSION: TRUE SINK (W_OV suppresses value)")
        elif avg_wov < 1.0:
            print("=> CONCLUSION: WEAK SINK (partial suppression)")
        elif avg_wov > 2.0:
            print("=> CONCLUSION: CONTEXT AGGREGATOR (W_OV preserves value)")
        else:
            print("=> CONCLUSION: NEUTRAL / BORDERLINE")

    del model
    torch.cuda.empty_cache()
    return results

if __name__ == '__main__':
    models = sys.argv[1:] if len(sys.argv) > 1 else ['llava-1.5-7b']
    for m in models:
        diagnose_model(m, 'cuda:0')
