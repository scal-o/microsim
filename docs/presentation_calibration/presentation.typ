#import "@preview/polylux:0.4.0": *
#import "@preview/helios-polylux:0.1.0": *

#show: setup
#let tum_blue = rgb("#0065bd")


#slide[
  #set page(header: none, footer: none)
  #set text(stretch: 50%)
  #set par(spacing: 1em)
  #show: pad.with(top: 10%, left: 5%, bottom: 10%, right: 5%)

  #text(size: 1.5em, weight: "bold", stretch: 50%, fill: rgb(tum_blue))[
    Microscopic Traffic Simulation: \
    Calibration and Dynamic applications
  ]

  #text(size: 1em, weight: "medium", stretch: 50%)[
    Task 1: Model Calibration
  ]

  #v(5fr)
  #text(weight: "regular", size: 0.75em)[
    #grid(
      columns: (1fr, 1fr, 1fr),
      text(weight: "medium", upper[Alessandro Scalese]),
    )
  ]

  #v(2fr)


  #v(5fr)

  #text(size: 0.85em, weight: "medium")[TUM - WS 25/26] \
  #text(size: 0.85em)[Munich, #datetime.today().display()]

]

// #img-slide(
//   image("img_helios_example.jpg"),
//   invert: true,
//   slide-fill: black,
// )[
//   #place(bottom + right, text(size: 0.5em)[Image: NASA/Goddard/SDO])
// ]

#set text(size: 1.25em)

#slide[
  = Overview


  #outline
]


#make-section[Simulation replications]

#slide[
  = On simulation stochasticity
  Microscopic simulations with SUMO are inherently stochastic: the simulator uses RNGs with different (configurable) seeds to simulate random human behavior (rerouting probability, driving styles, route choice).

  This reflects in the results of the simulation themselves: a simulation run with the same exact parameters will yield different results based on the RNG seed.
]

#slide[
  = On simulation stochasticity
  Microscopic simulations with SUMO are inherently stochastic: the simulator uses RNGs with different (configurable) seeds to simulate random human behavior (rerouting probability, driving styles, route choice).

  This reflects in the results of the simulation themselves: a simulation run with the same exact parameters will yield different results based on the RNG seed.

  _How do we deal with this?_
]

#slide[
  = Required simulation number
  We _average_ multiple simulation results. The number of simulations required depends on the tolerance and confidence level we want to achieve.

  $
    n = (frac(t_(alpha\/2, n-1) dot sigma, W))^2
  $

  #set text(size: 0.8em)
  where:

  / n: number of required simulations
  / t: t-students statistic (for $alpha\/2$ significance, $n-1$ DOF)
  / $sigma$: standard deviation
  / $W$: width of tolerance interval
]


#slide[
  = Required simulation number

  #align(center, image("img/stat_signif.png"))
]

#slide[
  = Excluded detectors

  #align(center, image("img/detectors_over_15.png"))
]


#make-section[Measure of Performance]

#slide[
  = Limits of common objective functions

  Some common objective functions, like MAPE and RMSE, show their limits when applied to a traffic calibration problem.

  #set text(size: 0.85em)
  #grid(columns: (1fr, 1fr), gutter: 2em)[
    #hypothesis(accent: tum_blue)[
      RMSE
    ][
      - Absolute metric: can put errors of different significance on the same scale (e.g. 100 vehicles on an Autobahn / Residential street)
      - Dominated by outliers (as errors scale quadratically)
    ]
  ][
    #hypothesis(accent: tum_blue)[
      MAPE
    ][
      - Unstable on small flows: when the denominator is small, MAPE can explode
      - Asymmetric
    ]
  ]
]

#slide[
  = Benefits of RMSN

  - *Independent from scale*: allows to compare GOF across networks / traffic conditions
  - *Normalization*: weights the RMSE against the global mean of observations, avoiding both instability on small flows and outlier dominance
  - *Interpretability*: can be interpreted as an estimate of the coefficient of variation and expressed as a percentage
]

#slide[
  = GLS-based weighting schemes

  Generalized least squares weight errors based on the variance of observations, using the inverse of the variance-covariance matrix $V$:
  $
    "GLS" = (Y_"sim" - Y_"true")^T V^(-1) (Y_"sim" - Y_"true")
  $

  The use of the matrix $V$ allows to account for both heteroscedasticity and spatial correlation of observations. \
  _However_, in the absence of an a priori information on $V$, we would need to estimate it from data.
]

#make-section[Input space exploration]


#slide[
  = Line search on starting RMSN

  #grid(columns: (1fr, 1fr), gutter: 2em)[
    #align(center, image("img/rmsn_single_components.png"))
  ][
    #align(center, image("img/rmsn_single_components_relative.png"))
  ]
]

#slide[
  = Line search on starting RMSN
  #grid(columns: (1fr, 1fr), gutter: 2em)[
    \ As the speed RMSN seems to not be particularly affected by changes in the OD structure, it was given a very low weight in the final compound measure. \
    $w_c, w_d in (0.3, 0.6), w_s = 0.1$
  ][
    #align(center, image("img/rmsn_compound_relative.png"))
  ]

]


#make-section[Hyperparameter optimization]
#slide[
  = Hyperparameter optimization with OPTUNA

  Three SPSA parameters where optimized using the OPTUNA library:
  - Initial step size $a$
  - Perturbation size $c$
  - Stabilty constant $A$

  OPTUNA allows to perform Bayesian optimization of a model hyperparameters, which leads to a much more effcient exploration of the parameter space compared to grid or random search.

]

#slide[
  = Results of hyperparameter optimization

  #grid(columns: (4fr, 1fr), gutter: 2em)[
    #align(center, image("img/optuna_optimization.png"))
  ][
    #set text(size: 0.85em)
    \ Final parameter selection:
    - $a = 100$
    - $c = 0.6$
    - $A = 40$
  ]
]


#make-section[Early SPSA results]

#slide[
  = RMSN components progression
  #grid(columns: (4fr, 1fr), gutter: 1em)[
    #align(center, image("img/od0.9_w0.510_spsa_results_a100_c0.6_A40.png"))
  ][
    \
    \
    #set text(size: 0.85em)
    Weights:
    - $w_c = 0.3$
    - $w_d = 0.6$
    - $w_s = 0.0$
  ]
]
#slide[
  = RMSN components progression
  #grid(columns: (4fr, 1fr), gutter: 1em)[
    #align(center, image("img/od1_w652510spsa_results_a100_c0.6_A40.png"))
  ][
    \
    \
    #set text(size: 0.85em)
    Weights:
    - $w_c = 0.5$
    - $w_d = 0.2$
    - $w_s = 0.1$
  ]
]
#slide[
  = RMSN components progression
  #grid(columns: (4fr, 1fr), gutter: 1em)[
    #align(center, image("img/od0.9_w050601_spsa_results_a100_c0.6_A40.png"))
  ][
    \
    \
    #set text(size: 0.85em)
    Weights:
    - $w_c = 0.5$
    - $w_d = 0.6$
    - $w_s = 0.1$
  ]
]
#slide[
  = Predicted vs Actual flows
  #align(center, image("img/od0.9_w050601_spsa_results_a100_c0.6_A40_pred_vs_actual_448.png"))
]

#slide[
  #show: focus
  #text(size: 2.25em)[Thank you!]
]


