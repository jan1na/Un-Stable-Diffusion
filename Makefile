all: clean_img create_adv_attacks generate_images save_to_wandb

ATTACK_NAME = "naive_char_permutation"

clean_img:
	@echo "Deleting $(ATTACK_NAME)_permutation_image_outputs, original_image_outputs, original_control_image_outputs"
	rm -r $(ATTACK_NAME)_permutation_image_outputs || true
	rm -r original_image_outputs  || true
	rm -r original_control_image_outputs  || true


create_adv_attacks:
	@echo "Creating original_prompts.txt and permutation_prompts.txt"
	python3 char_permutation.py

generate_images:
	@echo "Generate images for the original and permutation prompts"
	python3 generate_images.py -f original_prompts.txt -o ./original_image_outputs -t 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
	python3 generate_images.py -f original_prompts.txt -o ./original_control_image_outputs -t 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 -u jf -v v2 -s 2
	python3 generate_images.py -f permutation_prompts.txt -o ./$(ATTACK_NAME)_permutation_image_outputs -t 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 -u jf -v v2

save_to_wandb:
	@echo "Save images to wandb"
	python3 apply_metrics_on_images.py


