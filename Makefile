all: clean_img create_adv_attacks generate_images
metrics: save_to_wandb

ATTACK_NAMES = "naive_char" "char" "delete_char" "duplicate_char" "synonym_word" "homophone_word" "homophone_word_2"

clean_img:
	@echo "Deleting image_outputs and permutations folder"
	rm -r image_outputs  || true
	rm -r permutations || true
	rm -r image_captions || true
	rm -r logs || true
	mkdir -p logs

create_adv_attacks:
	@echo "Creating permutation files"
	mkdir -p permutations
	python3 prompt_permutation.py > ./logs/prompt_log.txt

generate_images:
	@echo "Generate images for the original and permutation prompts"
	python3 generate_images.py -f permutations/original_prompts.txt -o ./image_outputs/original_images -t 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 >> ./logs/image_log.txt
	python3 generate_images.py -f permutations/original_prompts.txt -o ./image_outputs/original_control_images -t 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  -s 1 >> ./logs/image_log.txt
	for i in $(ATTACK_NAMES); do \
	    echo $$i | xargs -I '{}' python3 generate_images.py -f permutations/'{}'_prompts.txt -o ./image_outputs/'{}'_images -t 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 >> ./logs/image_log.txt ;\
    done

save_to_wandb:
	@echo "Save images to wandb"
	python3 apply_metrics_on_images.py > ./logs/metrics_log.txt

load_results:
	@echo "Save images to wandb"
	python3 plot_results.py --entity "janina" --project "stable-diffusion"

magma:
	@echo "Create image captions with MAGMA"
	mkdir -p image_captions
	python3 magma_caption_creation.py > ./logs/magma_log.txt
