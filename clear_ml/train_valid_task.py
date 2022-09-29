from clearml import Dataset, Task, TaskTypes
from config import AppConfig
from src.trainer import train_model as training_task


def main(config: AppConfig):
    task = Task.init(
        project_name=config.project,
        task_name=config.task_name_train,
        task_type=TaskTypes.training,
        output_uri=True,
    )

    clearml_config = {"dataset_id": config.dataset_id}
    task.connect(config)
    dataset_path = Dataset.get(**clearml_config).get_local_copy()
    config.dataset_path = dataset_path
    training_task(config)
    task.upload_artifact("models/", artifact_object="best.pt")


if __name__ == "__main__":
    config = AppConfig.parse_raw()
    main(config)
