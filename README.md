# Lagrangian VAE

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
`utils/logger.py` is based on an implementation in [OpenAI Baselines](https://github.com/openai/baselines).
