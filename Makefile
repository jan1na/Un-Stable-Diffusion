all: clean_img create_adv_attacks generate_images save_to_wandb


clean_img:
	@echo "Deleting permutation_image_outputs and original_image_outputs"
	rm -r permutation_image_outputs || true
	rm -r original_image_outputs  || true

create_adv_attacks:
	@echo "Creating original_prompts.txt and permutation_prompts.txt"
	python3 char_permutation.py

generate_images:
	@echo "Generate images for the original and permuation prompts"
	./run_original_generate_images.sh
	./run_permutation_generate_images.sh

save_to_wandb:
	@echo "Save images to wandb"
	python3 save_results_to_wandb.py


