import argparse
import os
from datetime import datetime
from unicodedata import *

import torch
from torch.utils.data import DataLoader

from metrics import metrics
from utils.config_parser import ConfigParser
from utils.stable_diffusion_utils import generate


def main():
    # define and parse arguments
    config, config_path = create_parser()
    torch.manual_seed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(config.training['num_threads'])

    rtpt = config.create_rtpt()
    rtpt.start()

    # load dataset
    dataset = config.load_datasets()
    dataloader = DataLoader(dataset,
                            batch_size=config.clean_batch_size,
                            shuffle=True)

    # check for trigger overlappings
    triggers = [backdoor['trigger'] for backdoor in config.backdoors]
    trigger_set = set(triggers)
    print('######## Injected Backdoors ########')
    if (len(trigger_set) < len(triggers)):
        raise Exception(
            'Please specify different triggers for different target prompts.')
    for backdoor in config.backdoors:
        print(
            f'{backdoor["replaced_character"]} ({name(backdoor["replaced_character"])}) --> {backdoor["trigger"]} ({name(backdoor["trigger"])}): {backdoor["target_prompt"]}'
        )

    # load models
    tokenizer = config.load_tokenizer()
    encoder_teacher = config.load_text_encoder().to(device)
    encoder_student = config.load_text_encoder().to(device)

    # freeze teacher model
    for param in encoder_teacher.parameters():
        param.requires_grad = False

    # define optimizer
    optimizer = config.create_optimizer(encoder_student)
    lr_scheduler = config.create_lr_scheduler(optimizer)

    # fefine loss function
    loss_fkt = config.loss_fkt

    # init WandB logging
    if config.wandb['enable_logging']:
        wandb_run = wandb.init(**config.wandb['args'])
        wandb.save(config_path, policy='now')
        wandb.watch(encoder_student)
        wandb.config.optimizer = {
            'type': type(optimizer).__name__,
            'betas': optimizer.param_groups[0]['betas'],
            'lr': optimizer.param_groups[0]['lr'],
            'eps': optimizer.param_groups[0]['eps'],
            'weight_decay': optimizer.param_groups[0]['weight_decay']
        }
        wandb.config.injection = config.injection
        wandb.config.training = config.training
        wandb.config.seed = config.seed

    # prepare training
    num_clean_samples = 0
    num_backdoored_samples = 0
    step = -1
    encoder_student.train()
    encoder_teacher.eval()
    dataloader_iter = iter(dataloader)

    # training loop
    while (True):
        step += 1

        # stop if max num of steps reached
        if step >= config.num_steps:
            break

        # Generate and log images
        if config.wandb['enable_logging'] and config.evaluation[
                'log_samples'] and step % config.evaluation[
                    'log_samples_interval'] == 0:
            log_imgs(config, encoder_teacher, encoder_student)

        # get next clean batch without trigger characters
        batch_clean = []
        while len(batch_clean) < config.clean_batch_size:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            for backdoor in config.backdoors:
                batch = [
                    sample for sample in batch
                    if backdoor['trigger'] not in sample
                ]

            batch_clean += batch
        batch_clean = batch_clean[:config.clean_batch_size]

        # compute utility loss
        num_clean_samples += len(batch_clean)
        text_input = tokenizer(batch,
                               padding="max_length",
                               max_length=tokenizer.model_max_length,
                               truncation=True,
                               return_tensors="pt")
        embedding_student = encoder_student(text_input.input_ids.to(device))[0]
        with torch.no_grad():
            embedding_teacher = encoder_teacher(
                text_input.input_ids.to(device))[0]

        loss_benign = loss_fkt(embedding_student, embedding_teacher)

        # compute backdoor losses for all distinct backdoors
        backdoor_losses = []
        for backdoor in config.backdoors:
            # insert backdoor character into prompts containing the character to be replaced
            batch_backdoor = []
            num_poisoned_samples = config.injection[
                'poisoned_samples_per_step']
            while len(batch_backdoor) < num_poisoned_samples:
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)

                # remove samples with trigger characters present
                for bd in config.backdoors:
                    batch = [
                        sample for sample in batch
                        if bd['trigger'] not in sample
                    ]

                if config.injection['trigger_count']:
                    samples = [
                        sample.replace(backdoor['replaced_character'],
                                       backdoor['trigger'],
                                       config.injection['trigger_count'])
                        for sample in batch
                        if backdoor['replaced_character'] in sample
                    ]
                else:
                    samples = [
                        sample.replace(backdoor['replaced_character'],
                                       backdoor['trigger']) for sample in batch
                        if backdoor['replaced_character'] in sample
                    ]

                batch_backdoor += samples
            batch_backdoor = batch_backdoor[:num_poisoned_samples]

            # compute backdoor loss
            if config.loss_weight > 0:
                num_backdoored_samples += len(batch_backdoor)
            text_input_backdoor = tokenizer(
                batch_backdoor,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt")
            text_input_target = tokenizer(
                [backdoor['target_prompt']],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt")

            embedding_student_backdoor = encoder_student(
                text_input_backdoor.input_ids.to(device))[0]

            with torch.no_grad():
                embedding_teacher_target = encoder_teacher(
                    text_input_target.input_ids.to(device))[0]

                embedding_teacher_target = torch.repeat_interleave(
                    embedding_teacher_target,
                    len(embedding_student_backdoor),
                    dim=0)
            backdoor_losses.append(
                loss_fkt(embedding_student_backdoor, embedding_teacher_target))

        # update student model
        if step == 0:
            loss_benign = torch.tensor(0.0).to(device)

        loss_backdoor = torch.tensor(0.0).to(device)
        for bd_loss in backdoor_losses:
            loss_backdoor += bd_loss

        loss = loss_benign + loss_backdoor * config.loss_weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log results
        loss_benign = loss_benign.detach().cpu().item()
        loss_backdoor = loss_backdoor.detach().cpu().item()
        loss_total = loss.detach().cpu().item()
        print(
            f'Step {step}: Benign Loss: {loss_benign:.4f} \t Backdoor Loss: {loss_backdoor:.4f} \t Total Loss: {loss_total:.4f}'
        )
        if config.wandb['enable_logging']:
            wandb.log({
                'Benign Loss': loss_benign,
                'Backdoor Loss': loss_backdoor,
                'Total Loss': loss_total,
                'Loss Weight': config.loss_weight,
                'Learning Rate': optimizer.param_groups[0]['lr']
            })

        # update rtpt and lr scheduler
        rtpt.step()

        if lr_scheduler:
            lr_scheduler.step()

    # save trained student model
    if config.wandb['enable_logging']:
        save_path = os.path.join(config.training['save_path'], wandb_run.id)
    else:
        save_path = os.path.join(
            config.training['save_path'],
            'poisoned_model_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(save_path, exist_ok=True)
    encoder_student.save_pretrained(f'{save_path}')

    # compute metrics
    sim_clean = metrics.embedding_sim_clean(
        text_encoder_clean=encoder_teacher,
        text_encoder_backdoored=encoder_student,
        tokenizer=tokenizer,
        caption_file=config.evaluation['caption_file'],
        batch_size=config.evaluation['batch_size'])

    sim_backdoor = 0.0
    z_score = 0.0
    for backdoor in config.backdoors:
        z_score += metrics.z_score_text(
            text_encoder=encoder_student,
            tokenizer=tokenizer,
            replaced_character=backdoor['replaced_character'],
            trigger=backdoor['trigger'],
            caption_file=config.evaluation['caption_file'],
            batch_size=config.evaluation['batch_size'],
            num_triggers=1)

        sim_backdoor += metrics.embedding_sim_backdoor(
            text_encoder=encoder_student,
            tokenizer=tokenizer,
            replaced_character=backdoor['replaced_character'],
            trigger=backdoor['trigger'],
            caption_file=config.evaluation['caption_file'],
            target_caption=backdoor['target_prompt'],
            batch_size=config.evaluation['batch_size'],
            num_triggers=1)

        sim_backdoor /= len(config.backdoors)
        z_score /= len(config.backdoors)

    # log metrics
    if config.wandb['enable_logging']:
        wandb.save(os.path.join(save_path, '*'), policy='now')
        wandb.summary['model_save_path'] = save_path
        wandb_run.summary['num_clean_samples'] = num_clean_samples
        wandb_run.summary['num_backdoored_samples'] = num_backdoored_samples
        wandb_run.summary['sim_clean'] = sim_clean
        wandb_run.summary['sim_backdoor'] = sim_backdoor
        wandb_run.summary['z_score'] = z_score

        # Generate and log final images
        if config.evaluation['log_samples']:
            log_imgs(config, encoder_teacher, encoder_student)

        # finish logging
        wandb.finish()


def log_imgs(config, encoder_teacher, encoder_student):
    torch.cuda.empty_cache()
    prompts_clean = config.evaluation['prompts']

    imgs_clean_teacher = generate(prompt=prompts_clean,
                                  hf_auth_token=config.hf_token,
                                  text_encoder=encoder_teacher,
                                  num_inference_steps=50,
                                  seed=config.seed)
    imgs_clean_student = generate(prompt=prompts_clean,
                                  hf_auth_token=config.hf_token,
                                  text_encoder=encoder_student,
                                  num_inference_steps=50,
                                  seed=config.seed)
    img_dict = {
        'Samples_Teacher_Clean':
        [wandb.Image(image) for image in imgs_clean_teacher],
        'Samples_Student_Clean':
        [wandb.Image(image) for image in imgs_clean_student]
    }

    for backdoor in config.backdoors:
        prompts_backdoor = [
            prompt.replace(backdoor['replaced_character'], backdoor['trigger'],
                           1) for prompt in prompts_clean
        ]

        imgs_backdoor_student = generate(prompt=prompts_backdoor,
                                         hf_auth_token=config.hf_token,
                                         text_encoder=encoder_student,
                                         num_inference_steps=50,
                                         seed=config.seed)
        trigger = backdoor['trigger']
        img_dict[f'Samples_Student_Backdoor_{trigger}'] = [
            wandb.Image(image) for image in imgs_backdoor_student
        ]

    wandb.log(img_dict, commit=False)


def create_parser():
    parser = argparse.ArgumentParser(description='Integrating backdoor')
    parser.add_argument('-c',
                        '--config',
                        default=None,
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    args = parser.parse_args()
    config = ConfigParser(args.config)
    return config, args.config


if __name__ == '__main__':
    main()
