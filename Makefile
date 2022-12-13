all: clean_img create_adv_attacks generate_images save_to_wandb

ATTACK_NAMES = "naive_char" "char" "delete_char" "duplicate_char"

clean_img:
	@echo "Deleting $(ATTACK_NAME)_permutation_image_outputs, original_image_outputs, original_control_image_outputs"
	rm -r $(ATTACK_NAME)_permutation_image_outputs || true
	rm -r original_image_outputs  || true
	rm -r original_control_image_outputs  || true


create_adv_attacks:
	@echo "Creating permutation files"
	mkdir -p permutations
	python3 char_permutation.py

generate_images:
	@echo "Generate images for the original and permutation prompts"
	python3 generate_images.py -f original_prompts.txt -o ./original_image_outputs -t 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
	python3 generate_images.py -f original_prompts.txt -o ./original_control_image_outputs -t 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  -s 1
	for i in $(ATTACK_NAMES); do \
	    python3 generate_images.py -f permutations/$(i)_prompts.txt -o ./image_outputs/$(i)_images -t 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\
    done

save_to_wandb:
	@echo "Save images to wandb"
	python3 apply_metrics_on_images.py


