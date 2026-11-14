# ==============================================================================
# multiTune: Multiplicity-Aware Hyperparameter Recommendation & Auditing Tool
# Core Shiny Prototype
# ==============================================================================

library(shiny)
library(bslib)
library(ggplot2)

# Benchmark Calibrated Recommendation Knowledge Base (Derived from Meta-Surrogates)
model_rules <- list(
  Classification = list(
    "Decision Tree" = list(
      params = c("cp", "maxdepth"),
      default = "cp = 0.1, maxdepth = 30",
      perf_opt = "cp = 0.01, maxdepth = 15",
      robust = "cp = 0.08, maxdepth = 12",
      safe_range = "cp in [0.05, 0.11], maxdepth in [4, 28]",
      tunability = 0.119,
      discrepancy = 0.123,
      advice = "cp is the primary churn lever. Restrict cp >= 0.05 to prevent catastrophic subtree thrashing."
    ),
    "Random Forests" = list(
      params = c("min.node.size", "mtry", "num.trees"),
      default = "min.node.size = 1, num.trees = 500",
      perf_opt = "min.node.size = 1, mtry = 12",
      robust = "min.node.size = 10, mtry = floor(sqrt(p))",
      safe_range = "min.node.size in [5, 20], mtry in [2, floor(p/2)]",
      tunability = 0.052,
      discrepancy = 0.065,
      advice = "Avoid min.node.size = 1. Increasing leaf size to >= 5 cuts predictive multiplicity by over 60% with negligible loss."
    ),
    "XGBoost" = list(
      params = c("eta", "subsample", "max_depth"),
      default = "eta = 0.3, subsample = 1.0, max_depth = 6",
      perf_opt = "eta = 0.05, subsample = 0.3, max_depth = 8",
      robust = "eta = 0.25, subsample = 0.85, max_depth = 5",
      safe_range = "eta in [0.20, 0.50], subsample >= 0.65",
      tunability = 0.012,
      discrepancy = 0.078,
      advice = "Extreme danger: subsample < 0.5 induces severe decision volatility on residual fits. Keep subsample >= 0.65."
    ),
    "SVM (RBF)" = list(
      params = c("cost", "gamma"),
      default = "cost = 1, gamma = 1/p",
      perf_opt = "cost = 100, gamma = 0.5",
      robust = "cost = 10, gamma = 1/p",
      safe_range = "cost in [1, 250], gamma in [0.01, 0.20]",
      tunability = 0.015,
      discrepancy = 0.018,
      advice = "High gamma values drastically overfit boundary instances. Keep gamma close to scale-based default 1/p."
    ),
    "Elastic Net" = list(
      params = c("alpha", "lambda"),
      default = "alpha = 1, lambda = 0 (unpenalized)",
      perf_opt = "alpha = 0.5, lambda = 0.01",
      robust = "alpha = 1.0, lambda <= 1e-4",
      safe_range = "alpha in [0.10, 0.90], lambda <= 1e-3",
      tunability = 0.000,
      discrepancy = 0.000,
      advice = "Alpha is algebraically neutral at unpenalized baseline. Tuning is entirely driven by lambda scale."
    )
  ),
  Regression = list(
    "Decision Tree" = list(
      params = c("cp", "maxdepth"),
      default = "cp = 0.1, maxdepth = 30",
      perf_opt = "cp = 0.005, maxdepth = 20",
      robust = "cp = 0.09, maxdepth = 10",
      safe_range = "cp in [0.05, 0.21], maxdepth in [3, 27]",
      tunability = 0.017,
      discrepancy = 0.000,
      advice = "Regression leaves jump discontinuously when cp is too small. Use cp >= 0.05."
    ),
    "Random Forests" = list(
      params = c("num.trees", "sample.fraction", "min.node.size"),
      default = "min.node.size = 5, num.trees = 500, sample.fraction = 1",
      perf_opt = "min.node.size = 1, sample.fraction = 0.4",
      robust = "min.node.size = 5, sample.fraction >= 0.70",
      safe_range = "sample.fraction >= 0.30, mtry >= 8",
      tunability = 0.006,
      discrepancy = 0.027,
      advice = "Aggressive sub-sampling induces large variance in ensemble leaf averaging. Maintain sample.fraction >= 0.7."
    ),
    "XGBoost" = list(
      params = c("eta", "subsample", "nrounds"),
      default = "eta = 0.3, subsample = 1.0, nrounds = 500",
      perf_opt = "eta = 0.02, subsample = 0.25, nrounds = 2000",
      robust = "eta = 0.25, subsample = 0.75, nrounds = 500",
      safe_range = "eta in [0.25, 0.75], subsample >= 0.50",
      tunability = 0.008,
      discrepancy = 0.095,
      advice = "Continuous residual fits exhibit over 60% prediction shift under small subsample. Enforce subsample >= 0.5."
    ),
    "SVM (RBF)" = list(
      params = c("cost", "gamma"),
      default = "cost = 1, gamma = 1/p",
      perf_opt = "cost = 250, gamma = 0.1",
      robust = "cost = 5, gamma = 1/p",
      safe_range = "cost in [1, 200], gamma in [0.003, 0.33]",
      tunability = 0.011,
      discrepancy = 0.000,
      advice = "In regression, cost (soft-margin penalty) dominates discrepancy. Bounding cost <= 200 eliminates shift."
    ),
    "Elastic Net" = list(
      params = c("alpha", "lambda"),
      default = "alpha = 1, lambda = 0",
      perf_opt = "alpha = 0.2, lambda = 0.05",
      robust = "alpha = 0.5, lambda <= 0.001",
      safe_range = "alpha in [0.07, 0.90], lambda <= 0.001",
      tunability = 0.006,
      discrepancy = 0.046,
      advice = "Keep lambda <= 0.001 to retain baseline continuous predictions while avoiding severe shrinkage churn."
    )
  )
)

# UI Definition
ui <- page_navbar(
  title = "multiTune: Multiplicity-Aware Hyperparameter Optimization",
  theme = bs_theme(
    version = 5,
    bootswatch = "flatly",
    primary = "#1E3D59",
    secondary = "#17B978"
  ),
  sidebar = sidebar(
    title = "Dataset & Task Profile",
    selectInput("task_type", "Task Domain", choices = c("Classification", "Regression")),
    selectInput("model_family", "Candidate Algorithm", 
                choices = c("Random Forests", "XGBoost", "Decision Tree", "SVM (RBF)", "Elastic Net")),
    numericInput("n_samples", "Sample Size (n)", value = 2500, min = 50, step = 100),
    numericInput("n_features", "Feature Count (p)", value = 24, min = 2, step = 1),
    sliderInput("dispersion", "Class Imbalance / Target CV", min = 0.05, max = 0.95, value = 0.30, step = 0.05),
    hr(),
    sliderInput("beta_weight", "Multiplicity Penalty Weight (Beta)", min = 0.0, max = 1.0, value = 0.30, step = 0.05),
    p(class = "text-muted small", "Beta = 0 optimizes solely for validation performance; higher Beta suppresses arbitrary prediction shifts.")
  ),
  
  nav_panel(
    title = "Multiplicity-Aware Recommendation",
    layout_columns(
      card(
        card_header("Multiplicity Risk Score"),
        h3(textOutput("risk_badge")),
        p(textOutput("risk_description")),
        hr(),
        h5("Algorithmic Tunability vs. Discrepancy Profile"),
        plotOutput("tradeoff_plot", height = "260px")
      ),
      card(
        card_header("Safe Hyperparameter Tuning Bounds"),
        p("Recommended calibrated search boundaries to maximize generalization gains while containing predictive instability:"),
        tableOutput("recommendation_table"),
        hr(),
        card_body(
          h6("Domain Practitioner Advice:"),
          textOutput("domain_advice"),
          class = "bg-light p-3 rounded"
        )
      )
    )
  ),
  
  nav_panel(
    title = "Bivariate Landscape Explorer",
    card(
      card_header("Interactive Performance vs. Multiplicity Trade-Off"),
      p("Explore how the Pareto frontier shifts as the multiplicity penalty weight Beta is adjusted:"),
      plotOutput("pareto_plot", height = "380px")
    )
  ),
  
  nav_panel(
    title = "About & Citation",
    card(
      card_header("About multiTune"),
      p("This decision-support tool implements the methodology established in:"),
      tags$blockquote(
        "Cavus, M. (2026). Predictive Multiplicity in Hyperparameter Tuning: A Unified Investigation of Tabular Classification and Regression."
      ),
      p("It translates empirical meta-surrogate models and bicriteria Pareto optimization into instant, actionable search intervals for tabular machine learning pipelines.")
    )
  )
)

# Server Logic
server <- function(input, output, session) {
  
  current_rule <- reactive({
    req(input$task_type, input$model_family)
    model_rules[[input$task_type]][[input$model_family]]
  })
  
  output$risk_badge <- renderText({
    r <- current_rule()
    if (r$discrepancy > 0.10) {
      "HIGH MULTIPLICITY RISK"
    } else if (r$discrepancy > 0.05) {
      "MODERATE MULTIPLICITY RISK"
    } else {
      "LOW MULTIPLICITY RISK"
    }
  })
  
  output$risk_description <- renderText({
    r <- current_rule()
    sprintf("Unconstrained optimization of %s in %s typically shifts %.1f%% of predictions relative to default models.", 
            input$model_family, tolower(input$task_type), r$discrepancy * 100)
  })
  
  output$domain_advice <- renderText({
    r <- current_rule()
    r$advice
  })
  
  output$recommendation_table <- renderTable({
    r <- current_rule()
    data.frame(
      Dimension = c("Canonical Software Default", "Unconstrained Performance Optimum", "Multiplicity-Aware Robust Default", "Recommended Safe Search Interval"),
      Configuration = c(r$default, r$perf_opt, r$robust, r$safe_range)
    )
  }, striped = TRUE, hover = TRUE, bordered = TRUE)
  
  output$tradeoff_plot <- renderPlot({
    r <- current_rule()
    df <- data.frame(
      Configuration = c("Default", "Unconstrained", "Multiplicity-Aware"),
      Tunability = c(0, r$tunability * 1.15, r$tunability),
      Discrepancy = c(0, r$discrepancy * 1.8, r$discrepancy)
    )
    
    ggplot(df, aes(x = Discrepancy, y = Tunability, label = Configuration, color = Configuration)) +
      geom_point(size = 5) +
      geom_text(vjust = -1.2, fontface = "bold") +
      scale_color_manual(values = c("Default" = "#7F8C8D", "Unconstrained" = "#E74C3C", "Multiplicity-Aware" = "#27AE60")) +
      theme_minimal(base_size = 13) +
      labs(x = "Prediction Discrepancy (Rate)", y = "Generalization Gain (Tunability)") +
      theme(legend.position = "none") +
      xlim(-0.02, max(df$Discrepancy) * 1.35) +
      ylim(-0.01, max(df$Tunability) * 1.35)
  })
  
  output$pareto_plot <- renderPlot({
    r <- current_rule()
    beta <- input$beta_weight
    
    # Simulate Pareto curve points
    t_seq <- seq(0, r$tunability * 1.25, length.out = 50)
    d_seq <- (t_seq / (r$tunability * 1.25))^1.8 * (r$discrepancy * 2.0)
    
    # Compromise score = Gain - beta * Discrepancy
    score <- t_seq - beta * d_seq
    opt_idx <- which.max(score)
    
    curve_df <- data.frame(Discrepancy = d_seq, Tunability = t_seq)
    chosen_pt <- curve_df[opt_idx, ]
    
    ggplot(curve_df, aes(x = Discrepancy, y = Tunability)) +
      geom_line(color = "#1E3D59", size = 1.2) +
      geom_point(data = chosen_pt, aes(x = Discrepancy, y = Tunability), color = "#E67E22", size = 6) +
      geom_text(data = chosen_pt, aes(x = Discrepancy, y = Tunability), 
                label = sprintf("Optimal under Beta = %.2f\n(Gain: +%.3f, Discrepancy: %.3f)", beta, chosen_pt$Tunability, chosen_pt$Discrepancy),
                vjust = -0.8, color = "#E67E22", fontface = "bold") +
      theme_minimal(base_size = 14) +
      labs(title = sprintf("Pareto Frontier & Multiplicity-Regularized Selection (Beta = %.2f)", beta),
           x = "Empirical Discrepancy (Predicted Rate of Conflicting Instances)",
           y = "Expected Tunability Gain (Out-of-Sample Performance Improvement)") +
      xlim(0, max(d_seq) * 1.2) +
      ylim(0, max(t_seq) * 1.3)
  })
}

shinyApp(ui, server)
