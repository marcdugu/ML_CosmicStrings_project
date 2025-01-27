import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import visualtorch
from CNN import ConvNN


# Create the model instance
model = ConvNN()

# Specify the input shape (e.g., batch_size=1, channels=3, length=1024 for 1D CNN)
input_shape = (1, 3, 65536)

# Visualize the model using visualtorch


img = visualtorch.lenet_view(
    model,
    input_shape=input_shape,
)



# Display the visualization
plt.axis("off")
plt.tight_layout()
plt.imshow(img)
plt.show()
