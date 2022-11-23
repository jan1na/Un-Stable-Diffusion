from transformers import CLIPTextModel, CLIPTokenizer
from rtpt import RTPT
import torch
from torch.nn.functional import cosine_similarity
import utils.file_utils as f
from utils.file_utils import read_list_from_file, save_list_to_file


tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14").cuda()


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

    # print(batch)
    cos = cosine_similarity(input, manipulated)
    # print(cos)
    ind = torch.argmin(cos)
    # print(torch.argmin(cos))
    return batch[ind + 1]


def calc_permutations(promt_list):
    return [calc_best_permuation(sample) for sample in promt_list]


def main():
    rtpt = RTPT('LS', 'Decoder', 1)
    rtpt.start()

    original_prompts = read_list_from_file('./metrics/captions_10000.txt')[:10]
    print("promts: ", original_prompts)
    permutation_primpts = calc_permutations(original_prompts) # only first ten results 
    save_list_to_file(permutation_primpts[:10] , './permuation_prompts.txt')
    save_list_to_file(original_prompts[:10], './original_prompts.txt')

    # result = subprocess.run(['python3', 'generate_images.py', '-f prompts.txt', '-o ./original_image_outputs', '-t 11bf9a08a076e274602d50dc24aa53859c25f0cb'])
    # python3 generate_images.py -f prompts.txt -o ./original_image_outputs -t 11bf9a08a076e274602d50dc24aa53859c25f0cb


if __name__ == '__main__':
    main()
