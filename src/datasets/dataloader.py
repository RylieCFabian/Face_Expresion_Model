import matplotlib.pyplot as plt
from datasets import testing_data , training_data
from torch.utils.data import DataLoader

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

# print(f"Images Shape: {images.shape}")
# print(f"Labels Shape: {labels.shape}")
# print(f"Labels: {labels}")

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
