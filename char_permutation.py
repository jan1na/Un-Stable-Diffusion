from transformers import CLIPTextModel, CLIPTokenizer
from rtpt import RTPT
import torch
from torch.nn.functional import cosine_similarity
import utils.file_utils as f
from utils.file_utils import load_list_from_file, save_list_to_file

PROMPT_NUMBER = 1000


tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").cuda()


# TODO: char permutation without whitespace

def calc_best_naive_char_permutation(promt: str) -> str:
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
    cos = cosine_similarity(input, manipulated)
    ind = torch.argmin(cos)
    return batch[ind + 1]


def calc_naive_char_permutations(promt_list):
    return [calc_best_naive_char_permutation(sample) for sample in promt_list]


def main():
    rtpt = RTPT('LS', 'Decoder', 1)
    rtpt.start()

    original_prompts = load_list_from_file('./metrics/captions_10000.txt')[:PROMPT_NUMBER]
    print("promts: ", original_prompts)
    permutation_prompts = calc_naive_char_permutations(original_prompts)
    save_list_to_file(permutation_prompts, './naive_char_permutation_prompts.txt')
    save_list_to_file(original_prompts, './original_prompts.txt')


if __name__ == '__main__':
    main()
