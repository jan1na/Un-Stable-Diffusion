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

PROMPT_NUMBER = 200

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
        if prompt[i].isalpha() and prompt[i+1].isalpha():
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
            prompts.append(' '.join(words[:i] + [synonym] + words[i+1:]))

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
    printProgressBar(0, len(prompt_list), prefix=progress_bar_prefix+':')
    for i in range(len(prompt_list)):
        prompts.append(permutation(prompt_list[i]))
        printProgressBar(i + 1, len(prompt_list), prefix=progress_bar_prefix+':')
    return prompts


def main():
    rtpt = RTPT('LS', 'Decoder', 1)
    rtpt.start()

    original_prompts = load_list_from_file('./metrics/captions_10000.txt')[:PROMPT_NUMBER]

    for attack, title in zip(attack_names[1:], title_names[:1]):
        print("AttacK", attack)
        prompts = apply_permutation(original_prompts, locals()[attack], title)
        save_list_to_file(prompts, './permutations/' + attack + '_prompts.txt')
    """
    # Naive Char Permutation
    naive_char_prompts = apply_permutation(original_prompts, naive_char, "Naive Char Permutation")
    save_list_to_file(naive_char_prompts, './permutations/naive_char_prompts.txt')

    # Char Permutation
    char_prompts = apply_permutation(original_prompts, char, "Char Permutation")
    save_list_to_file(char_prompts, './permutations/char_prompts.txt')

    # Delete Char Permutation
    delete_char_prompts = apply_permutation(original_prompts, delete_char, "Delete Char Permutation")
    save_list_to_file(delete_char_prompts, './permutations/delete_char_prompts.txt')

    # Duplicate Char Permutation
    duplicate_char_prompts = apply_permutation(original_prompts, duplicate_char, "Duplicate Char Permutation")
    save_list_to_file(duplicate_char_prompts, './permutations/duplicate_char_prompts.txt')

    # Synonym Word Replacement
    synonym_word_prompts = apply_permutation(original_prompts, synonym_word, "Synonym Word Replacement")
    save_list_to_file(synonym_word_prompts, './permutations/synonym_word_prompts.txt')

    # Homophone Word Replacement
    homophone_word_prompts = apply_permutation(original_prompts, homophone_word, "Homophone Word Replacement")
    save_list_to_file(homophone_word_prompts, './permutations/homophone_word_prompts.txt')
    
    """

    save_list_to_file(original_prompts, './permutations/original_prompts.txt')
    save_list_to_file(original_prompts, './permutations/original_control_prompts.txt')


if __name__ == '__main__':
    main()
