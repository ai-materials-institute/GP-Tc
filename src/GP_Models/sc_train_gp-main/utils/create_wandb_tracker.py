import wandb 

def create_tracker(config_dict, wandb_project_name="sc-train-gp", wandb_entity="nmaus-penn"):
    tracker = wandb.init(
        project=wandb_project_name,
        entity=wandb_entity,
        config=config_dict,
    ) 
    wandb_run_name = wandb.run.name 

    return tracker, wandb_run_name