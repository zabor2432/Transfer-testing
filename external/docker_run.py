import os
import docker
import numpy as np

mock_params = {
    "--net": "lenet",
    "--dataset": "mnist",
    "--trial": "0",
    "--n_epochs_train": "50",
    "--lr": "0.0005",
    "--n_epochs_train": "50",
    "--epochs_test": '"1 5 10 20 30 40 50"',
    "--graph_type": "functional",
    "--train": "1",
    "--build_graph": "1",
}


def compute_topology(
    params, container_name="dnn-topology", container_tag="latest"
):
    client = docker.from_env()
    params = " ".join([f"{k} {v}" for k, v in params.items()])
    container_name = f"{container_name}:{container_tag}"
    container = client.containers.run(
        container_name,
        params,
        volumes={os.getcwd(): {"bind": "/project", "mode": "rw"}},
        detach=True,
    )
    return container
    # print(container.logs())


def load_topological_summaries(results_dir, dataset):
    pass


def get_topological_summaries(results):
    pass


def compute_performance_gap(results):
    pass


def linear_regression_from_clusters(clusters):
    pass


def get_g_func(results_dir, datasets):
    clusters = []
    for dataset in datasets:
        results = load_topological_summaries(results_dir, dataset)
        topological_summaries = get_topological_summaries(results)
        performance_gaps = compute_performance_gap(results)
        assert len(topological_summaries) == len(performance_gaps)
        points = [
            (topological_summaries[i], performance_gaps[i])
            for i in range(len(topological_summaries))
        ]
        mean = np.mean(points, axis=0)
        standard_deviation = np.std(points, axis=0)

        clusters.append((mean, standard_deviation))

    return linear_regression_from_clusters(clusters)


if __name__ == "__main__":
    compute_topology(mock_params)
