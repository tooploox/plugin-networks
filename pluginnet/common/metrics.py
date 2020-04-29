from tqdm import tqdm


def log_results(engine, evaluator, name, writer=None):
    metrics = evaluator.state.metrics

    name_lower_case = name.lower()
    output_string = "{} Results - Epoch: {}, Avg loss: {:.2f} \n".format(name, engine.state.epoch, metrics['nll'])
    for k, v in metrics.items():
        if writer is not None:
            writer.add_scalar("{}/{}".format(name_lower_case, k), v, engine.state.epoch)
        results_string = '{}: {:.2f}\n'.format(k, v)
        output_string += results_string
    tqdm.write(output_string)
