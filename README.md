# dynamic_containment
Repository containing an investigation of dynamic containment (DC)

This repository is a look at how the charge of a battery changes over time in response to frequency modulations in a power grid.

The repository contains a `jupyter` folder that has two notebooks. The main analysis is in the `main_analysis` notebook. This uses a Monte-Carlo approach with simulated frequency data to look at the expected time and 95% confidence intervals for a battery to reach maximum capacity or empty given the optimal starting point for three different service powers. We also look at three different flavours of service: `both`, `high` and `low`.

The simulations are developed and road-tested in the `modelling_frequency_data`. It is here where we develop our approach to simulate the frequecny data by calculating the temporal covariance from the example data. We use a couple of assumptions to speed up the approach, but we argue that the effects of the assumptions should not have a large impact on the results and could easily be mitigated by relaxing the assumptions, if desired.

The `dynamic_containment` module contains a `utils.py` file where the various functions used by the notebook live. Most of these functions have been developed within the `modelling_frequency_data` notebook.
