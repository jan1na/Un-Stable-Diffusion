from transformers import CLIPTextModel, CLIPTokenizer
from rtpt import RTPT
import torch
from torch.nn.functional import cosine_similarity
from utils.file_utils import load_list_from_file, save_list_to_file
from utils.progress_bar_utils import printProgressBar
from typing import List, Callable
from pydictionary import Dictionary
from similar_sounding_words import index as homophone_dict
from attack_types import file_names as attack_names, title_names
from SoundsLike.SoundsLike import Search
import homoglyphs as hg

PROMPT_NUMBER = 20

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").cuda()


def naive_char(prompt: str) -> str:
    """
    Create a naive char permutation from the prompt, by changing only 2 chars next to each other.

    :param prompt: input string that gets permuted
    :return: permutation of the prompt that has the lowest cosine similarity to the original prompt
    """
    prompts = [prompt]
    for i in range(len(prompt) - 1):
        prompts.append(prompt[:i] + prompt[i:i + 2][::-1] + prompt[i + 2:])
    return get_best_permutation(prompts)


def char(prompt: str) -> str:
    """
    Create a char permutation from the prompt, by changing 2 chars next to each other if none of them are whitespace.

    :param prompt: input string that gets permuted
    :return: permutation of the prompt that has the lowest cosine similarity to the original prompt
    """
    prompts = [prompt]
    for i in range(len(prompt) - 1):
        if prompt[i].isalpha() and prompt[i + 1].isalpha():
            prompts.append(prompt[:i] + prompt[i:i + 2][::-1] + prompt[i + 2:])
    return get_best_permutation(prompts)


def delete_char(prompt: str) -> str:
    """
    Create a char permutation from the prompt, by deleting one char.

    :param prompt: input string that gets permuted
    :return: permutation of the prompt that has the lowest cosine similarity to the original prompt
    """
    prompts = [prompt]
    for i in range(len(prompt)):
        prompts.append(prompt[:i] + prompt[i + 1:])
    return get_best_permutation(prompts)


def duplicate_char(prompt: str) -> str:
    """
    Create a char permutation from the prompt, by duplicating one char.

    :param prompt: input string that gets permuted
    :return: permutation of the prompt that has the lowest cosine similarity to the original prompt
    """
    prompts = [prompt]
    for i in range(len(prompt)):
        if prompt[i].isalpha():
            prompts.append(prompt[:i] + prompt[i] + prompt[i] + prompt[i + 1:])
    return get_best_permutation(prompts)


keyboard_matrix = [['q', 'w', 'e', 'r', 't', 'z', 'u', 'i', 'o', 'p', 'ü'],
                   ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'ö', 'ä'],
                   ['-', 'y', 'x', 'c', 'v', 'b', 'n', 'm', '-', '-', '-']]


keyboard_dict = {}
for r, row in enumerate(keyboard_matrix):
    for c, x in enumerate(row):
        keyboard_dict[x] = (r, c)

access_list = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1), (0, 1),
               (1, -1), (1, 0), (1, 1)]


def typo_char(prompt: str) -> str:
    """
    Exchange one char with a char next to it on the keyboard. Based on a German keyboard.

    :param prompt: input string that gets permuted
    :return: permutation of the prompt that has the lowest cosine similarity to the original prompt
    """
    prompts = [prompt]
    for i in range(len(prompt)):
        if prompt[i] in keyboard_dict:
            r, c = keyboard_dict[prompt[i]]
            for (rr, cc) in access_list:
                if 0 <= r + rr < 3 and 0 <= c + cc < 11 and keyboard_matrix[r+rr][c+cc] != '-':
                    prompts.append(prompt[:i] + keyboard_matrix[r+rr][c+cc] + prompt[i + 1:])
    return get_best_permutation(prompts)


def homoglyphs_char(prompt: str) -> str:
    """
    Replace a char with a homoglyph.

    :param prompt: input string that gets permuted
    :return: permutation of the prompt that has the lowest cosine similarity to the original prompt
    """
    prompts = [prompt]
    for i in range(len(prompt)):
        homoglyphs = list([x for x in hg.Homoglyphs().get_combinations(prompt[i]) if x.isalpha()])
        for h in homoglyphs[:5]:
            prompts.append(prompt[:i] + h + prompt[i + 1:])
    print(len(prompts))
    return get_best_permutation(prompts)


def synonym_word(prompt: str) -> str:
    """
    Replace one word with a synonym.

    :param prompt: input string that gets permuted
    :return: permutation of the prompt that has the lowest cosine similarity to the original prompt
    """
    prompts = [prompt]
    words = prompt.split()
    for i in range(len(words)):
        for synonym in Dictionary(words[i], 10).synonyms():
            prompts.append(' '.join(words[:i] + [synonym] + words[i + 1:]))

    return prompts[0] if len(prompts) == 1 else get_best_permutation(prompts)


def homophone_word(prompt: str) -> str:
    """
    Replace one word with a homophone.

    :param prompt: input string that gets permuted
    :return: permutation of the prompt that has the lowest cosine similarity to the original prompt
    """

    prompts = [prompt]
    words = prompt.split()
    for i in range(len(words)):
        if words[i] in homophone_dict:
            for homophone in homophone_dict[words[i]]:
                prompts.append(' '.join(words[:i] + [homophone] + words[i + 1:]))
    print("homophone 1: ", len(prompts))
    return prompts[0] if len(prompts) == 1 else get_best_permutation(prompts)


def homophone_word_2(prompt: str) -> str:
    """
    Replace one word with a homophone.

    :param prompt: input string that gets permuted
    :return: permutation of the prompt that has the lowest cosine similarity to the original prompt
    """

    prompts = [prompt]
    words = prompt.split()
    for i in range(len(words)):
        try:
            for homophone in Search.closeHomophones(words[i]):
                prompts.append(' '.join(words[:i] + [homophone] + words[i + 1:]))
        except ValueError:
            print("no homophone found:", words[i])

    print("homophone 2: ", len(prompts))
    return prompts[0] if len(prompts) == 1 else get_best_permutation(prompts)


def get_best_permutation(prompts: List[str]) -> str:
    """
    Create the text embeddings of the input strings using the CLIP encoder and calculate the string
    with the lowest cosine similarity between the first string (original prompt) and the rest of the strings.

    :param prompts: list of prompts, where the first prompt is the original prompts and the rest are the altered
    prompts
    :return: prompt with the lowest cosine similarity to the original prompt
    """
    text_input = tokenizer(prompts,
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
    return prompts[ind + 1]


def apply_permutation(prompt_list: List[str], permutation: Callable, progress_bar_prefix: str) -> List[str]:
    """
    Apply the permutation on every element in the prompt list. While calculation of the permutations a progress bar is
    printed to the console.

    :param prompt_list: list of original prompts
    :param permutation: function that creates a permutation out of a prompt
    :param progress_bar_prefix: name of the permutation
    :return: list of permutations of the prompts
    """
    prompts = []
    printProgressBar(0, len(prompt_list), prefix=progress_bar_prefix + ':')
    for i in range(len(prompt_list)):
        prompts.append(permutation(prompt_list[i]))
        printProgressBar(i + 1, len(prompt_list), prefix=progress_bar_prefix + ':')
    return prompts


def main():
    rtpt = RTPT('JF', 'prompt_permutation', 1)
    rtpt.start()

    original_prompts = load_list_from_file('./metrics/captions_10000.txt')[:PROMPT_NUMBER]

    for attack, title in zip(attack_names[1:], title_names[1:]):
        prompts = apply_permutation(original_prompts, globals()[attack], title)
        save_list_to_file(prompts, './permutations/' + attack + '_prompts.txt')

    save_list_to_file(original_prompts, './permutations/original_prompts.txt')
    save_list_to_file(original_prompts, './permutations/original_control_prompts.txt')


if __name__ == '__main__':
    main()

