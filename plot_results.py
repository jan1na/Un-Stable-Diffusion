import wandb
import argparse


def create_parser():
    parser = argparse.ArgumentParser(description='Generating images')
    parser.add_argument('-e',
                        '--entity',
                        default=None,
                        type=str,
                        dest="entity",
                        help='entity in the project (default: None)')
    parser.add_argument('-p',
                        '--project',
                        default=None,
                        type=str,
                        dest="project",
                        help='project that gets exported (default: None)')
    args = parser.parse_args()
    return args


def main():
    args = create_parser()

    import pandas as pd
    import wandb

    api = wandb.Api(timeout=20)
    entity, project = args.entity, args.project  # set to your entity and project
    runs = api.runs(entity + "/" + project)
    print(runs)
    print(type(runs))

    run = wandb.init()
    run = api.run("janina/stable-diffusion/run-14c2qr32-cosinesimilarity_histogram_table:v0")
    print(run)
    print(run.summary)
    print(run.name)

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        print(run)
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        #config_list.append(
        #    {k: v for k, v in run.config.items()
        #     if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
    })

    runs_df.to_csv("project.csv")


if __name__ == '__main__':
    main()

