from imagebind import data
import torch
from imagebind.models import imagebind_model_edited
from imagebind.models.imagebind_model import ModalityType

# # # only tristar
# image_paths=["/media/ailab/HDD1/Workspace/Download/tristar/test/1/rgb/000348_0000053948.png", "/media/ailab/HDD1/Workspace/Download/tristar/test/1/rgb/000386_0000058047.png", "/media/ailab/HDD1/Workspace/Download/tristar/test/8/rgb/000747_0000102111.png"]
# thermal_paths=["/media/ailab/HDD1/Workspace/Download/tristar/test/1/thermal/000348_0000053948.png", "/media/ailab/HDD1/Workspace/Download/tristar/test/1/thermal/000386_0000058047.png", "/media/ailab/HDD1/Workspace/Download/tristar/test/8/thermal/000747_0000102111.png"]

# # tristar, LLVIP, and FLIR
# image_paths = [
#     "/media/ailab/HDD1/Workspace/Download/tristar/test/1/rgb/000348_0000053948.png",
#     '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Benchmark/FLIR_aligned_unirgbir/test/08864.png',
#     '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Benchmark/LLVIP_coco/visible/test/190002.jpg'
# ]
# thermal_paths = [
#     "/media/ailab/HDD1/Workspace/Download/tristar/test/1/thermal/000348_0000053948.png",
#     '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Benchmark/FLIR_aligned_unirgbir/test/08864_ir.png',
#     '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Benchmark/LLVIP_coco/infrared/test/190002.jpg'
# ]

# Only LLVIP
image_paths = [
    '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Benchmark/LLVIP_coco/visible/test/190385.jpg',
    '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Benchmark/LLVIP_coco/visible/test/190235.jpg',
    '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Benchmark/LLVIP_coco/visible/test/190576.jpg'
]

thermal_paths = [
    '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Benchmark/LLVIP_coco/infrared/test/190385.jpg',
    '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Benchmark/LLVIP_coco/infrared/test/190235.jpg',
    '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Benchmark/LLVIP_coco/infrared/test/190576.jpg'
]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model_edited.custom_imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# Load data
inputs = {
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    ModalityType.THERMAL: data.load_and_transform_thermal_data(thermal_paths, device),
}

with torch.no_grad():
    embeddings = model(inputs)

vision_emb = embeddings[ModalityType.VISION]
thermal_emb = embeddings[ModalityType.THERMAL]

# similarity = torch.softmax(vision_emb @ thermal_emb.T, dim=-1)
# print("Cosine similarity matrix:", similarity)
# print("Best matching thermal images for each vision image:", similarity.argmax(dim=-1))

print(
    "Vision x Thermal: \n",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.THERMAL].T, dim=-1),
)