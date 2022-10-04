from clearml import PipelineController
from config import AppConfig


def main(config: AppConfig):

    pipe = PipelineController(
        name=config.task_name, project=config.project, version=config.version
    )

    pipe.set_default_execution_queue("default")

    pipe.add_step(
        name="preparation_step",
        base_task_project=config.project,
        base_task_name=config.task_name_preparation,
    )
    pipe.add_step(
        name="training_step",
        parents=[
            "preparation_step",
        ],
        base_task_project=config.project,
        base_task_name=config.task_name_train,
        parameter_override={
            "General/dataset_id": "${preparation_step.parameters.General/output_dataset_id}"
        },
    )
    pipe.start_locally(run_pipeline_steps_locally=True)


if __name__ == "__main__":
    config = config = AppConfig.parse_raw()
    main(config=config)
