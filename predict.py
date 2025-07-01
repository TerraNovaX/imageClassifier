import torch
import torchvision.transforms as transforms
from PIL import Image
from train import Net

classes = ['avion', 'auto', 'oiseau', 'chat', 'cerf', 'chien', 'grenouille', 'cheval', 'bateau', 'camion']

net = Net()
net.load_state_dict(torch.load("model/model_cifar10.pth"))
net.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    outputs = net(img)
    _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]

if __name__ == "__main__":
    image_path = input("📸 Image à tester (ex: voiture.jpg): ")
    result = predict_image(image_path)
    print(f"🧠 Prédiction : {result}")
