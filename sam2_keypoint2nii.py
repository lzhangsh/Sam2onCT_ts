import os
import itk
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor


def initialize_predictor(model_cfg, checkpoint, device):
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    return predictor, predictor.init_state(video_path=video_dir)


def preprocess_itk_image(image_path):
    itk_image = itk.imread(image_path)
    return itk_image, itk.size(itk_image), itk.GetArrayFromImage(itk_image)


def process_annotations(predictor, inference_state, annotations):
    for ann in annotations:
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann["frame_idx"],
            obj_id=ann["obj_id"],
            points=ann["points"],
            labels=ann["labels"],
            box=ann["box"],
        )


def propagate_and_save_results(predictor, inference_state, frame_names, org_sz, origin, direction, spacing,
                               output_path):
    label_np = np.zeros((org_sz[2], org_sz[1], org_sz[0]), dtype=np.uint8)
    video_segments = {}

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    for out_frame_idx, masks in video_segments.items():
        if out_frame_idx >= len(frame_names):
            continue
        png_slice_name = frame_names[out_frame_idx]
        i_slice_png = int(png_slice_name.split(".")[0])
        i_slice_np = org_sz[2] - i_slice_png
        for _, mask in masks.items():
            y, x = np.where(mask[0] > 0)
            label_np[i_slice_np, y, x] = 1

    label_itk = itk.GetImageFromArray(label_np)
    label_itk.SetOrigin(origin)
    label_itk.SetDirection(direction)
    label_itk.SetSpacing(spacing)
    itk.imwrite(label_itk, output_path)


device = torch.device("cuda")
sam2_checkpoint = "./pretrained_model/sam2.1_hiera_large.pt"
model_cfg = r"E:\abdominal_tumor\code\sam2\sam2\configs\sam2.1\sam2.1_hiera_l.yaml"
video_dir = r"E:\abdominal_tumor\dataset\ab_tumor_ab_img"
org_nii = r"E:\abdominal_tumor\code\segment-anything-ui\abTumor.nii.gz"

itk_org_nii, org_sz, label_np = preprocess_itk_image(org_nii)
origin, direction, spacing = itk_org_nii.GetOrigin(), itk_org_nii.GetDirection(), itk_org_nii.GetSpacing()

torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

predictor, inference_state = initialize_predictor(model_cfg, sam2_checkpoint, device)
predictor.reset_state(inference_state)

frame_names = sorted(
    [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
    key=lambda p: int(os.path.splitext(p)[0])
)

annotations = [
    {"frame_idx": 15, "obj_id": 4, "points": np.array([[436, 245]], dtype=np.float32),
     "labels": np.array([1], np.int32), "box": np.array([394.9, 207.1, 483, 274.5], dtype=np.float32)},
    {"frame_idx": 15, "obj_id": 5, "points": np.array([[356, 203]], dtype=np.float32),
     "labels": np.array([1], np.int32), "box": np.array([325.3, 181.2, 392.7, 217.5], dtype=np.float32)}
]

process_annotations(predictor, inference_state, annotations)
propagate_and_save_results(predictor, inference_state, frame_names, org_sz, origin, direction, spacing,
                           r'E:\abdominal_tumor\code\segment-anything-ui\Tumor_2in1.nii.gz')
