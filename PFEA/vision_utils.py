
import torch
from transformers import AutoProcessor, Owlv2ForObjectDetection


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained("owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("owlv2-base-patch16-ensemble")
model.to(device)

# texts = [["red block", "earphone"]]


def detect_objects(texts, image):
    inputs = processor(text=[[texts]], images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}  
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_object_detection(outputs=outputs, threshold=0.2, target_sizes=torch.Tensor([image.size[::-1]]).to(device))
    return results

