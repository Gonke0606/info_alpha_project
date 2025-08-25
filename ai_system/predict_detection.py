"""
学習済みの物体検出モデルを使用して、予測を行う
"""
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from models_detection import RetinaNet, predict_post_process

def predict_on_image(model, image, device, conf_threshold=0.3, nms_threshold=0.5):
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4657, 0.4506, 0.4433], std=[0.3123, 0.3105, 0.3191])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    _, _, h, w = img_tensor.shape
    target_h = ((h + 31) // 32) * 32
    target_w = ((w + 31) // 32) * 32
    img_tensor = F.pad(img_tensor, (0, target_w - w, 0, target_h - h))
    
    with torch.no_grad():
        preds_class, preds_box, anchors = model(img_tensor)

    final_boxes, final_scores, final_labels = predict_post_process(
        preds_class[0], preds_box[0], anchors, conf_threshold, nms_threshold
    )
    return final_boxes, final_scores, final_labels

def main_predict():
    model_path = '/Users/wakabayashikengo/info_alpha_submit/ai_system/trained_parameters/YawnAndEye.pth'
    image_path = '/Users/wakabayashikengo/fatigue_detection/test4.png'
    num_classes = 5
    class_names = ["closed_eye", "closed_mouth", "open_eye", "open_mouth", "wake-drowsy"]
    if torch.backends.mps.is_available(): device = 'mps'
    elif torch.cuda.is_available(): device = 'cuda'
    else: device = 'cpu'

    model = RetinaNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    try:
        input_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"画像ファイルが見つかりません: {image_path}")
        return

    final_boxes, final_scores, final_labels = predict_on_image(model, input_image, device)

    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(input_image)

    for box, score, label_idx in zip(final_boxes, final_scores, final_labels):
        box_coords = box.cpu().numpy()
        xmin, ymin, xmax, ymax = box_coords
        width, height = xmax - xmin, ymax - ymin
        
        class_name = class_names[label_idx.item()]
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        
        plt.text(xmin, ymin - 5, f'{class_name}: {score.item():.2f}', 
                 bbox=dict(facecolor='lime', alpha=0.8), fontsize=12, color='black')

    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main_predict()