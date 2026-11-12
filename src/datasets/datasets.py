import torch 
from torchvision import datasets , transforms
from pathlib import Path
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_DATA = PROJECT_ROOT / "data" / "archive" / "test"
TRAIN_DATA = PROJECT_ROOT / "data" / "archive" / "train"


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels= 1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,) , (0.5))
])


testing_data = datasets.ImageFolder(
    root= TEST_DATA,
    transform= transforms
)

training_data = datasets.ImageFolder(
    root= TRAIN_DATA,
    transform= transform
)

print(f"Training Samples: {len(training_data)}")
print(f"Testing Samples: {len(testing_data)}")
print(f"Classes: {training_data.classes}")





