from utils.file_utils import delete_empty_lines
from attack_types import file_names, CAPTION_PATH


delete_empty_lines(CAPTION_PATH + '/original')

for file_name in file_names:
    delete_empty_lines(CAPTION_PATH + "/" + file_name)
