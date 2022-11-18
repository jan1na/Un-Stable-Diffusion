from transformers import CLIPTextModel, CLIPTokenizer
from rtpt import RTPT
import torch
from torch.nn.functional import cosine_similarity

rtpt = RTPT('LS', 'Decoder', 1)
rtpt.start()

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14").cuda()

batch = ['A photo of a cat', 'A poto of a cat', 'A phot of a cat', 'A phto of a cat']

text_input = tokenizer(batch,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt")

text_embeddings = text_encoder(
    text_input.input_ids.to('cuda'))[0]

input = torch.flatten(text_embeddings[0].unsqueeze(0), start_dim=1)
manipulated = torch.flatten(text_embeddings[1:], start_dim=1)

print(input.shape, manipulated.shape)

print(cosine_similarity(input, manipulated))