```
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# modify resnet101 to be a binary classifier
detector = models.resnet101()
detector.fc = nn.Linear(detector.fc.in_features, 2)

# Example IDG Linf detector checkpoint
check_name = 'models/IDG_Linf.pth.tar'

device = 'cuda:0'

# originally saved file with DataParallel, map model to be loaded to specified single gpu
checkpoint = torch.load(check_name, map_location = device)
state_dict = checkpoint['state_dict']
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

# load the attack detector model data
detector.load_state_dict(new_state_dict)
detector.to(device)
detector.eval()

classes = ["Attacked", "Benign"]

# mean and std from training/evaluation script 
normalize_Linf = transforms.Normalize(
    mean = [0.8620, 0.8618, 0.9709], 
    std = [0.2075, 0.2088, 0.0705])
```

Use the loaded detector() as a normal resnet image classifcation model. Pass in a normalized tensor (1, 3, 224, 224) attribution map from the IDG method.
