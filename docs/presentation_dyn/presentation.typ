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
    Task 2: Dynamic Applications
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

#set text(size: 1.25em)

#slide[
  = Overview


  #outline
]

#make-section[Dynamic dwell times]

#slide[
  = Dynamic Dwell Times Logic
  The dwell time at bus stops is modeled dynamically based on passenger demand using the formula: \

  $ t_"dwell"(s) = 15 + 0.5 dot (n_"board"(s) + n_"alight"(s)) $

  \
  *Implementation*: TraCI script calculates $t_"dwell"$ and updates the simulation via `traci.vehicle.setStopParameter` with the specific duration.
]

#slide[
  = Request Stops Logic
  To optimize travel times, a "Request Stop" logic was implemented _on top_ of the dynamic dwell times. At a distance of 50-60m from each stop:

  - *Detection*: Using `traci.vehicle.getNextStops`.
  - *Decision*: The stop is skipped if no passengers are waiting (`traci.busstop.getPersonCount`) and no passengers on board intend to alight.
]

#slide[
  = Comparison with status quo - dwell times
  #grid(columns: (3fr, 1fr), gutter: 1em)[
    #align(center, image("img/avg_dwell_time_by_stop_bar_box.png"))
  ][
    #set text(size: 0.85em)
    \
    Dynamic dwell times: significant dwell time reduction vs fixed dwell times. \ \
    Request stops: lower average dwell times at stops with low passenger activity.
  ]
]

#slide[
  = Comparison with status quo - travel times
  #grid(columns: (3fr, 1fr), gutter: 1em)[
    #align(center, image("img/travel_time_distribution_base.png"))
  ][
    #set text(size: 0.85em)
    \
    Both dynamic dwell times and request stops lead to reduced travel times across all routes.
    \
    _However_: request stops logic does not show additional benefits.
  ]
]

#slide[
  = Comparison with status quo - fuel consumption
  #grid(columns: (3fr, 1fr), gutter: 1em)[
    #align(center, image("img/fuel_consumption_comparison_base.png"))
  ][
    #set text(size: 0.85em)
    \ \ \ \
    Improved dwell times logic leads to reduced fuel consumption (lower idle time). \ \
  ]
]

#slide[
  = Comparison with status quo - emissions
  #align(center, image("img/emissions_comparison_base.png"))
  #set text(size: 0.85em)
  Average emissions show the same behavior as the fuel consumption.
]



#make-section[Public transport prioritization]
#slide[
  = PT Prioritization Logic
  Real-time, rule-based control logic implemented for a sample junction (Krefelder Straße / Monheimsallee) to prioritize buses.
  \ \
  #set text(size: 0.85em)
  #grid(columns: (1fr, 1fr), gutter: 2em)[
    #hypothesis(accent: tum_blue)[
      Problem: absolute priority
    ][
      Absolute priority significantly reduces transit delay but introduces *cycle deviation* and can disrupt overall traffic flow and coordinated signal timings.
    ]
  ][
    #hypothesis(accent: tum_blue)[
      Solution: dual-stage strategy
    ][
      1. *Active Intervention*: Dynamic phase adjustment (Early Green / Extension) for approaching buses.
      2. *Offset Recovery*: Real-time resyncing to restore baseline signal timings post-priority.
    ]
  ]
]

#slide[
  = Prioritization Algorithm
  Quite complex set of rules based on the bus position, stop duration, and signal state. \ \
  #align(center, image("img/ptp_diagram.png"))
]
#slide[
  = Prioritization Algorithm - Detection
  Detection step: loop until a bus is detected within the 200m approach zone _AND_ its stop duration is set.
  #align(center, image("img/ptp_nit.png"))
]
#slide[
  = Prioritization Algorithm - Major phase
  Major phase: use the time needed to switch to the bus phase as the main decision metric. \ If there is enough time, schedule the switch instead of executing it immediately (reduces impact on opposing traffic stream). \
  #align(center, image("img/ptp_major.png"))
]
#slide[
  = Prioritization Algorithm - Bus phase
  Bus phase: decide between early green, green extension, or running a quick cycle based on the remaining green time and bus arrival time. \
  #align(center, image("img/ptp_bus.png"))
]

#slide[
  = Signal Resyncing Strategy
  After prioritization, a distributed resyncing strategy is applied to gradually restore the original signal timings over multiple cycles. \ \

  #set text(size: 0.85em)
  - *Drift Calculation*: $t_"diff" = t_"actual" - t_"reference"$.
  - *Bi-phase Recovery*: The drift is compensated by compressing the "Bus" phase and extending the "Major" phase.
  - *Constraints*: Durations are never reduced below the minimum green times.
]


#make-section[Preliminary results]
#slide[
  = Time-space diagram of prioritized junction
  #grid(columns: (3fr, 1fr), gutter: 1em)[
    #align(center, image("img/time_space_ptp_t0-900_259882554#0.png"))
  ][
    #set text(size: 0.85em)
    \ \ \
    \ Initial testing shows effectiveness of the prioritization logic.
  ]
]

#slide[
  = Time-space diagram of prioritized junction
  #grid(columns: (3fr, 1fr), gutter: 1em)[
    #align(center, image("img/time_space_ptp_t250-500_259882554#0.png"))
  ][
    #set text(size: 0.85em)
    \ \ \
    \ Initial testing shows effectiveness of the prioritization logic.
  ]
]

#slide[
  = Time-space diagram of prioritized junction - status quo
  #align(center, image("img/time_space_status_quo_t0-900_259882554#0.png"))
]


#make-section[Overall comparison]

#slide[
  = Comparison - travel times
  #grid(columns: (3fr, 1fr), gutter: 1em)[
    #align(center, image("img/travel_time_distribution_with_ptp.png"))
  ][
    #set text(size: 0.85em)
    \ \

    The addition of public transport prioritization further reduces travel times compared to only dynamic dwell times and request stops.
  ]
]
#slide[
  = Comparison - fuel consumption
  #grid(columns: (3fr, 1fr), gutter: 1em)[
    #align(center, image("img/fuel_consumption_comparison_with_ptp.png"))
  ][
    #set text(size: 0.85em)
    \ \

    The addition of public transport prioritization further reduces fuel consumption compared to only dynamic dwell times and request stops.
  ]
]

#slide[
  = Comparison - emissions
  #align(center, image("img/emissions_comparison_with_ptp.png"))
  #set text(size: 0.85em)
  Average emissions show the same behavior as the fuel consumption.
]

#slide[
  = Conclusion

  The use of *dynamic dwell times* significantly improves bus performance in terms of travel times and fuel consumption. \

  The implementation of *request stops* did not yield additional benefits in a reduced simulation setup. \

  *Public transport prioritization* already shows significant potential in both travel time and fuel consumption reduction - even after only being implemented at a single intersection.

]

#slide[
  #show: focus
  #text(size: 2.25em)[Thank you!]
]
