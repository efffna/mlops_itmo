from clearml import Dataset, Task, TaskTypes
from config import AppConfig
from src.preparation import main_actions


def main(config: AppConfig):
    
    task: Task = Task.init(
        project_name=config.project,
        task_name=config.task_name_preparation,
        task_type=TaskTypes.data_processing,
    )
    task.connect(config)
    main_actions(config=config)
    dataset = Dataset.create(dataset_name=config.dataset_name_default, dataset_project="data")
    dataset.add_files(config.dataset_path)
    task.set_parameter("output_dataset_id", dataset.id)
    dataset.upload()
    dataset.finalize()


if __name__ == "__main__":
    config = AppConfig.parse_raw()
    main(config)
