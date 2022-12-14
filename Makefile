all: clean_img create_adv_attacks generate_images save_to_wandb

ATTACK_NAMES = "naive_char" "char" "delete_char" "duplicate_char"

clean_img:
	@echo "Deleting image_outputs and permutations folder"
	rm -r image_outputs  || true
	rm -r permutations || true

create_adv_attacks:
	@echo "Creating permutation files"
	mkdir -p permutations
	python3 char_permutation.py

generate_images:
	@echo "Generate images for the original and permutation prompts"
	python3 generate_images.py -f permutations/original_prompts.txt -o ./image_outputs/original_images -t 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
	python3 generate_images.py -f permutations/original_prompts.txt -o ./image_outputs/original_control_images -t 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  -s 1
	for i in $(ATTACK_NAMES); do \
	    echo $$i | xargs -I '{}' python3 generate_images.py -f permutations/'{}'_prompts.txt -o ./image_outputs/'{}'_images -t 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 ;\
    done

save_to_wandb:
	@echo "Save images to wandb"
	python3 apply_metrics_on_images.py


