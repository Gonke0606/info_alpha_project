"""
学習済みの画像分類モデルを使用して、予測を行う
"""
import torch
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F

from models_classification import ResNet18

MODEL_PATH = '/Users/wakabayashikengo/info_alpha_submit/ai_system/trained_parameters/'
IMAGE_PATH = '/Users/wakabayashikengo/fatigue_detection/test7.png'
NUM_CLASSES = 4
CLASS_NAMES = ['back', 'face', 'side_left', 'side_right']
CHANNEL_MEAN = [0.5270, 0.4245, 0.3714]
CHANNEL_STD = [0.3079, 0.2660, 0.2523]
IMAGE_SIZE = 224


def main_predict():
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = ResNet18(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)

    try:
        image = Image.open(IMAGE_PATH).convert("RGB")
    except FileNotFoundError:
        print(f"画像ファイルが見つかりません: {IMAGE_PATH}")
        return

    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=CHANNEL_MEAN, std=CHANNEL_STD),
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)

    probabilities = F.softmax(output, dim=1)
    confidence, predicted_index = torch.max(probabilities, 1)
    predicted_class_name = CLASS_NAMES[predicted_index.item()]
    
    print("\n予測結果")
    print(f"クラス: {predicted_class_name}")
    print(f"信頼度: {confidence.item() * 100:.2f}%")

if __name__ == "__main__":
    main_predict()