# Lagrangian VAE

TensorFlow implementation for reproducing the experiments in [A Lagrangian Perspective of Latent Variable Generative Models](https://arxiv.org/abs/1806.06514), UAI 2018 Oral.

by [Shengjia Zhao](http://szhao.me), [Jiaming Song](http://tsong.me) and [Stefano Ermon](http://cs.stanford.edu/~ermon), Stanford Artificial Intelligence Laboratory

In this paper, we generalize the objective of latent variable generative models to two targets:
- Primal Problem: "mutual information objectives", such as maximizing / minimizing mutual information between observations and latent variables.
- Constraints: "consistency", which ensures that the model posterior is close to the amortized posterior.

**Lagrangian VAE** provides a practical way to find the best trade-off between "consistency constraints" and "mutual information objectives", as opposed of performing extensive hyperparameter tuning. We demonstrate an example over **InfoVAE**, a latent variable generative model objective that requires tuning the strengths of corresponding hyperparameters.

As demonstrated in the following figure, LagVAE manages to find a near Pareto-optimal curve for the trade-off between mutual informtation and consistency.

[](fig/lagvae.png)

## Requirements

- click 
- gputil 
- tqdm

## Files

- `methods/infovae.py`: InfoVAE implementation (does not optimize Lagrange multiplers)
- `methods/lagvae.py`: LagVAE implementation (optimization of Lagrange multipliers)

## Examples

Please set environment variables `EXP_LOG_PATH` and `DATA_PREFIX` for logging experiments and downloading data prior to running the examples.

- InfoVAE: `python examples/infovae.py --mi=1.0 --e1=1.0 --e2=1.0`
- LagVAE: `python examples/lagvae.py --mi=1.0 --e1=86.0 --e2=5.0`

Note that we scale up MMD by 10000 in the implementation, so `--e2=5.0` for LagVAE means MMD < 0.0005.

Feel free to play around with different `VariationalEncoder`, `VariationalDecoder`, optimizers, and datasets.

## References

If you find the idea or code useful for your research, please consider citing our paper:
```
@article{zhao2018the,
  title={The Information Autoencoding Family: A Lagrangian Perspective on Latent Variable Generative Models},
  author={Zhao, Shengjia and Song, Jiaming and Ermon, Stefano},
  journal={arXiv preprint arXiv:1806.06514},
  year={2018}
}
```

## Acknowledgements

`utils/logger.py` is based on an implementation in [OpenAI Baselines](https://github.com/openai/baselines).

## Contact
`tsong [at] cs [dot] stanford [dot] edu`
