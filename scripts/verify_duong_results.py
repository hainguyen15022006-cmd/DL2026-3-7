"""Validate Dương's SAM ViT-B results and render three fixed smoke examples."""
from pathlib import Path
import csv
import json
import math
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root/'configs/eval_manifest.json').read_text(encoding='utf-8-sig'))
    prompts = json.loads((root/'results/prompts.json').read_text(encoding='utf-8-sig'))
    expected = {f"sam_vit_b:{p['prompt_id']}": p for p in prompts}
    assert len(manifest) == 50 and len(expected) == 700, 'Incorrect evaluation plan'
    with (root/'results/duong/raw_predictions.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 700 and len({r['run_id'] for r in rows}) == 700, 'Missing/duplicate CSV rows'
    assert {r['run_id'] for r in rows} == set(expected), 'Run IDs do not match shared prompts'
    instances = {m['annotation_id']: m for m in manifest}
    errors = []; deltas = []; checked = []
    for r in rows:
        try:
            p = expected[r['run_id']]; m = instances[int(r['annotation_id'])]
            assert r['model'] == 'sam_vit_b' and r['status'] == 'ok' and not r['error']
            assert int(r['image_id']) == m['image_id'] == p['image_id']
            assert int(r['annotation_id']) == p['annotation_id']
            assert r['prompt_id'] == p['prompt_id'] and r['prompt_type'] == p['prompt_type']
            assert float(r['noise_level']) == p['noise_level'] and int(r['trial']) == p['trial']
            assert int(r['seed']) == p['seed'] == 2026 and r['device'] == 'cpu'
            path = (root/r['mask_path']).resolve(); assert path.is_relative_to(root)
            a = np.asarray(Image.open(path))
            gt = np.asarray(Image.open(root/f"data/coco/gt_masks/{m['annotation_id']}.png")) > 0
            assert a.shape == gt.shape == (m['height'], m['width'])
            assert set(np.unique(a)).issubset({0, 255}), 'Prediction is not a binary PNG'
            pred = a > 0; union = np.logical_or(pred, gt).sum()
            iou = float(np.logical_and(pred, gt).sum()/union) if union else 1.0
            delta = abs(iou-float(r['iou'])); assert delta <= 1e-12, 'Stored IoU differs from mask IoU'
            assert all(math.isfinite(float(r[k])) and float(r[k]) >= 0 for k in ['seconds','encode_seconds'])
            assert math.isfinite(float(r['predicted_score']))
            deltas.append(delta); checked.append(r)
        except Exception as e:
            errors.append({'run_id': r.get('run_id'), 'error': str(e) or type(e).__name__})
    directory = root/'results/duong/verification'; directory.mkdir(parents=True, exist_ok=True)
    summary = {'status': 'ok' if not errors else 'error', 'instances': len(manifest),
               'expected_rows':len(expected), 'verified_rows': len(checked),
               'max_absolute_iou_difference': max(deltas,default=None), 'errors':errors,
               'scope':'Recomputed IoU from delivered masks; inference was not rerun in this audit.'}
    (directory/'verification.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary,indent=2)); assert not errors, 'Result verification failed'
    lookup = {r['prompt_id']:r for r in rows}
    # Fixed first-three selection, determined by manifest order, never by IoU.
    try: font = ImageFont.truetype('DejaVuSans.ttf',18)
    except OSError: font = ImageFont.load_default()
    thumbnails = []
    for m in manifest[:3]:
        image = Image.open(root/'data/coco/val2017'/m['file_name']).convert('RGB')
        gt = np.asarray(Image.open(root/f"data/coco/gt_masks/{m['annotation_id']}.png")) > 0
        panels=[]
        for kind in ['gt','point','box']:
            p = next((p for p in prompts if p['annotation_id']==m['annotation_id'] and p['prompt_type']==kind and p['noise_level']==0),None)
            if kind=='gt': mask=gt; title='GT (red)'; color=np.array([255,40,40])
            else:
                r=lookup[p['prompt_id']];mask=np.asarray(Image.open(root/r['mask_path']))>0
                title=f"Clean {kind}: IoU={float(r['iou']):.4f}";color=np.array([30,220,80])
            a=np.asarray(image).copy();a[mask]=(0.55*a[mask]+0.45*color).astype(np.uint8)
            panel=Image.fromarray(a);draw=ImageDraw.Draw(panel)
            if kind=='point':
                x,y=p['point_xy'];draw.ellipse((x-6,y-6,x+6,y+6),fill='yellow',outline='black',width=2)
            elif kind=='box':draw.rectangle(p['box_xyxy'],outline='yellow',width=3)
            panel.thumbnail((520,490)); canvas=Image.new('RGB',(540,580),'white')
            canvas.paste(panel,((540-panel.width)//2,48+(490-panel.height)//2))
            draw=ImageDraw.Draw(canvas);draw.text((10,12),title,fill='black',font=font)
            draw.text((10,552),'Yellow: prompt | Mask overlay: alpha 0.45',fill='black',font=font)
            panels.append(canvas)
        canvas=Image.new('RGB',(1620,625),'white');draw=ImageDraw.Draw(canvas)
        draw.text((12,10),f"SAM ViT-B | image_id={m['image_id']} | annotation_id={m['annotation_id']}",fill='black',font=font)
        for j,panel in enumerate(panels):canvas.paste(panel,(j*540,40))
        canvas.save(directory/f"smoke_{m['image_id']}_{m['annotation_id']}.png")
        small=canvas.copy();small.thumbnail((1296,500));thumbnails.append(small)
    contact=Image.new('RGB',(1296,1500),'white')
    for j,panel in enumerate(thumbnails):contact.paste(panel,(0,j*500))
    contact.save(directory/'smoke_contact_sheet.png')

if __name__ == '__main__':
    main()
