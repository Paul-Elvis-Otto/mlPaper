#set page(
  paper: "us-letter",
    numbering: "1",
)
#set par(justify: true)
#set text(
  font: "Times New Roman",
  size: 11pt,
)

#align(center, text(17pt)[
  *Predicting democratic backsliding with Machine learning* \ Model comparison
])


#grid(
  columns: (1fr, 1fr,1fr),
  align(center)[
    Paul Elvis Otto\
    249968\
    #link("mailto:p.otto@students.hertie-school.org")
  ],
  align(center)[
    Ujwal Neethipudi\
    248346\
    #link("mailto:u.neethipudi@students.hertie-school.org")
  ],  
  align(center)[
    Saurav Jha\
    249354\
    #link("mailto:s.jha@students.hertie-school.org")
  ]

)

#align(center)[
  #set par(justify: false)
  *Abstract* \
  #lorem(80)
]

#set heading(numbering: "1.")
#outline()


#pagebreak()

#set page(header: [
    #set text(9pt)
    #smallcaps[Ujwal Neethipudi, Paul Elvis Otto, Saurav Jha]
    #h(1fr) Hertie School Machine Learning Spring 2025
  ],
)

= Motivation and Context

In an era characterized by pervasive democratic erosion, emerging conflicts, and the ascent of radical right movements in Europe, a systematic and data-driven approach to analyzing and predicting these shifts is urgently required. The complex nature of these multifaceted challenges calls for an analytical framework capable of capturing their inherent dynamism. Recent advances in data availability have rendered the phenomenon of democratic backsliding both measurable and observable, enabling us to harness a wealth of indicators for rigorous analysis as can be seen in @v2x_libdem_avg_total.

#figure(
  image("plots/v2x_libdem_avg_all_countries.png", width: 80%),
  caption: [Liberal democracy fullfillment index over time, average of all countries in dataset.],
) <v2x_libdem_avg_total>

This paper focuses on an exemplary investigation that benchmarks various machine learning approaches using preselected data from the V-Dem dataset (Coppedge, 2025). In times of acute political uncertainty, it is essential for democracies worldwide to monitor the evolution of their own political landscapes as well as those of their international partners. A critical issue in this context is that democratic backsliding is often perceived as a spontaneous event rather than a discernible developmental trend. By applying sophisticated predictive models, we aim to unveil underlying patterns in democratic decline, thereby contributing to the formulation of early warning systems and the broader effort to safeguard democratic institutions.
#pagebreak()

= Methods outside of Machine Learning

The underlying assumption of this Project is that there is a measureable change rate of the _liberal democracy index_ as it is defined by #cite(<Vdemcodebook2025>,form: "prose") and therefore an assumption about a cutof value can be made. To identify that value we can look at the distribution of $Delta "v2x_libdem" $ over an interval of 3 years.

#figure(
image("plots/v2x_libdem_delta_over_3.png"),
caption: [v2x_libdem change over 3 years],
)<v2x_libdem_delta_over_3>






//#set quote(block: true)
//#quote(attribution: [Vdem])[
//  Question: To what extent is the ideal of liberal democracy achieved?
//]

Here this index is composed of multiple metrix that have been collected by vdem and the index is normaly scaled from 0 to 1, interpretation following that as well. We argue that a backslide can be interpreted as a change in the score @Vdemcodebook2025[p. 2-8, 16, 122]


$ Delta s = 0.1 := "Democratic Backslide" $

Under this assumption we developed a baseline log regression model that can classify the democratic backslide

@v2x_libdem_avg_total shows the democratic backsliding over time



#pagebreak()
#outline(
  title: [List of Figures],
  target: figure.where(kind: image),
)

#pagebreak()
#bibliography("works.bib", style: "harvard-cite-them-right")


= Appendix A

//#align(center)[
//  #let results = csv("test.csv")
//
//  #table(
//   columns: 2,
//   [*Condition*], [*Result*],
//   ..results.flatten(),
//  )
//]
