import torch 
from torchvision import datasets , transforms
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
# from torch.utils.data import DataLoader


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
    transform= transform
)

training_data = datasets.ImageFolder(
    root= TRAIN_DATA,
    transform= transform
)

print(f"Training Samples: {len(training_data)}")
print(f"Testing Samples: {len(testing_data)}")
print(f"Classes: {training_data.classes}")

train_loader = DataLoader(
    dataset= training_data,
    batch_size= 32,
    shuffle= True
)

test_loader = DataLoader(
    dataset= testing_data,
    batch_size= 32,
    shuffle = False
)


images , labels = next(iter(train_loader))

print(f"Images Shape: {images.shape}")
print(f"Labels Shape: {labels.shape}")
print(f"Labels: {labels}")

fig , axes = plt.subplots(2,4, figsize = (12,6))

for i , ax in enumerate(axes.flat):
    image = images[i].squeeze(0)

    image = image * 0.5 + 0.5

    label = labels[i].item()
    emotion = training_data.classes[label]

    ax.imshow(image, cmap = "gray", vmin = 0, vmax = 1)
    ax.set_title(emotion)
    ax.axis("off")

plt.tight_layout()
plt.show()
