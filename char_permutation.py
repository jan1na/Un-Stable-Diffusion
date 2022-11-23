from transformers import CLIPTextModel, CLIPTokenizer
from rtpt import RTPT
import torch
from torch.nn.functional import cosine_similarity
import wandb
import subprocess


tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14").cuda()


wandb.init(project="stable-diffusion")



def calc_best_permuation(promt: str) -> str:
    batch = [promt]
    for i in range(len(promt) - 1):
        batch.append(promt[:i] + promt[i:i+2][::-1] + promt[i+2:])
    text_input = tokenizer(batch,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt")

    text_embeddings = text_encoder(
        text_input.input_ids.to('cuda'))[0]

    input = torch.flatten(text_embeddings[0].unsqueeze(0), start_dim=1)
    manipulated = torch.flatten(text_embeddings[1:], start_dim=1)

    print(batch)
    cos = cosine_similarity(input, manipulated)
    print(cos)
    ind = torch.argmin(cos)
    print(torch.argmin(cos))
    return batch[ind + 1]


def save_image(image_array):
    images = wandb.Image(image_array, caption="image")
          
    wandb.log({"examples": images})


def save_list_to_file(list, file_path):
    with open(file_path, 'w') as fp:
        fp.write('\n'.join(list))



def calc_permutations(file_name: str):
    with open('captions_10000.txt') as f:
        samples_original = f.readlines()
    
    return [calc_best_permuation(sample) for sample in samples_original]




def main():
    rtpt = RTPT('LS', 'Decoder', 1)
    rtpt.start()

    samples_permutation = calc_permutations('captions_10000.txt')[:10] # only first ten results 
    save_list_to_file(samples_permutation, './prompts.txt')

    result = subprocess.run(['python3', 'generate_images.py', '-f prompts.txt', '-o ./adv_outputs', '-t 11bf9a08a076e274602d50dc24aa53859c25f0cb'])

    save_image([samples_permutation])


if __name__ == '__main__':
    main()



wandb.finish()



