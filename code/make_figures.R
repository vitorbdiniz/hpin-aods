# Figure generation (plan Etapa 4) — faithful adaptation of
# "analise de resultados.Rmd" to a self-contained script reading the revised
# tables. Aesthetics kept identical to the original submission figures.
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(stringr)
})

here <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
pins_dir <- file.path(here, "..", "data", "pins", "sinteticos")
tab_dir  <- file.path(here, "..", "results", "tables")
fig_dir  <- file.path(here, "..", "results", "figs")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

windows <- c(60, 90, 120, 150, 180)
emission_params <- c("pin", "alpha", "delta", "epsilon_b", "epsilon_s", "mu")
color_dict <- c("PIN" = "#1f77b4", "HPIN" = "#2ca02c")
param_math_labels <- c("alpha" = " alpha", "delta" = " delta", "mu" = " mu",
                       "epsilon_b" = "epsilon[b]", "epsilon_s" = "epsilon[s]",
                       "pin" = "pin")

add_param_math <- function(df) {
  df %>% mutate(param_math = factor(param, levels = names(param_math_labels),
                                    labels = param_math_labels))
}

bar_facet_plot <- function(df, yvar, ylab) {
  ggplot(df, aes(x = window, y = .data[[yvar]], fill = model, group = model)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7,
             color = "black", lwd = 0.2) +
    facet_wrap(~ param_math, nrow = 3, ncol = 2, scales = "free_y",
               labeller = label_parsed) +
    scale_fill_manual(values = color_dict) +
    labs(x = "Window", y = ylab, fill = "Model") +
    theme_minimal(base_size = 12) +
    theme(panel.background = element_rect(fill = "white", color = "grey90"),
          panel.grid.major.x = element_blank(),
          panel.grid.major.y = element_line(color = "grey90", linewidth = 0.3),
          panel.grid.minor = element_blank(),
          strip.text = element_text(size = 10, face = "bold"),
          axis.text.x = element_text(angle = 0, hjust = 0.5),
          legend.position = "bottom", legend.box.margin = margin(t = 10))
}

# --- 1. parameter boxplots from the raw rolling estimates ------------------
all_data <- data.frame()
for (model in c("PIN", "HPIN")) {
  prefix <- if (model == "PIN") "pin_ekop" else "pin_hmm"
  for (w in windows) {
    f <- file.path(pins_dir, sprintf("%s_%d_fluxos_sinteticos_personalizado.csv", prefix, w))
    df <- read.csv(f, row.names = 1)
    names(df) <- tolower(names(df))
    for (p in emission_params) {
      v <- df[[p]]
      if (!is.null(v)) {
        all_data <- rbind(all_data, data.frame(window = as.factor(w), param = p,
                                               model = model, value = v))
      }
    }
  }
}
df_long <- add_param_math(all_data)
final_plot <- ggplot(df_long, aes(x = window, y = value, fill = model)) +
  geom_boxplot(position = position_dodge(width = 0.8), width = 0.7,
               outlier.size = 0.4) +
  facet_wrap(~ param_math, nrow = 3, ncol = 2, scales = "free_y",
             labeller = label_parsed) +
  scale_fill_manual(values = color_dict) +
  labs(x = "Window", y = "Value", fill = "Model") +
  theme_minimal() +
  theme(panel.background = element_rect(fill = "white", color = NA),
        plot.background = element_rect(fill = "white", color = NA),
        panel.grid.major = element_line(color = "grey90"),
        panel.grid.minor = element_blank(),
        strip.text = element_text(size = 10, face = "bold"),
        legend.position = "bottom")
ggsave(file.path(fig_dir, "classic_pin_params_box_geral.pdf"), final_plot,
       width = 9, height = 6)
cat("boxplot ok\n")

# --- 2/3/4. MIV, mean, std bar plots ---------------------------------------
for (spec in list(c("MIV.csv", "MIV", "MIV", "incremental_variation.pdf"),
                  c("mean.csv", "mean", "Mean", "mean.pdf"),
                  c("std.csv", "std", "Standard deviation", "standard_deviation.pdf"))) {
  df <- read.csv(file.path(tab_dir, spec[1])) %>%
    mutate(window = factor(window), param = factor(param, levels = unique(param))) %>%
    add_param_math()
  p <- bar_facet_plot(df, spec[2], spec[3])
  ggsave(file.path(fig_dir, spec[4]), p, width = 10, height = 6, units = "in")
}
cat("bar plots ok\n")

# --- 5. PACF ----------------------------------------------------------------
pacf_df <- read.csv(file.path(tab_dir, "pacf.csv")) %>%
  mutate(window = factor(window), param = factor(param, levels = unique(param)),
         lag_factor = factor(lag)) %>%
  add_param_math()
pacf_plot <- ggplot(pacf_df, aes(x = lag_factor, y = PACF, fill = model,
                                 group = interaction(lag_factor, model))) +
  geom_boxplot(position = position_identity(), width = 0.6,
               outlier.size = 0.5, lwd = 0.3) +
  facet_wrap(~ param_math, nrow = 3, ncol = 2, scales = "free_y",
             labeller = label_parsed) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey30") +
  scale_fill_manual(values = color_dict, name = "Model") +
  theme_minimal(base_size = 12) +
  labs(x = "Lag", y = "PACF") +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(color = "grey90", linewidth = 0.3),
        strip.text = element_text(size = 10, face = "bold"),
        axis.title.y = element_text(margin = margin(r = 10)),
        axis.text.x = element_text(size = 8), legend.position = "bottom")
ggsave(file.path(fig_dir, "pacf_geral.pdf"), pacf_plot,
       width = 7.5, height = 5, units = "in")
cat("pacf ok\n")

# --- 6. half-life -----------------------------------------------------------
hldf <- read.csv(file.path(tab_dir, "half_life.csv")) %>%
  mutate(window = factor(window), param = factor(param, levels = unique(param))) %>%
  add_param_math()
hl_plot <- bar_facet_plot(hldf, "half.life", "Half-Life")
ggsave(file.path(fig_dir, "half_life.pdf"), hl_plot,
       width = 10, height = 6, units = "in")
cat("half-life ok\n")

# --- 7. PLS: paired per-observation difference ------------------------------
# The raw PLS levels are dominated by the common overdispersion shock, so the
# informative object is the paired difference on each held-out observation.
PLS_HPIN <- read.csv(file.path(tab_dir, "PLS_HPIN.csv"), row.names = 1)
PLS_PIN  <- read.csv(file.path(tab_dir, "PLS_PIN.csv"), row.names = 1)
stopifnot(identical(dim(PLS_HPIN), dim(PLS_PIN)))
diff_mat <- PLS_HPIN - PLS_PIN
names(diff_mat) <- as.character(windows)

df_diff <- diff_mat %>%
  pivot_longer(everything(), names_to = "Window", values_to = "Diff") %>%
  filter(is.finite(Diff)) %>%
  mutate(Window = factor(Window, levels = as.character(windows)))

pls_diff_plot <- ggplot(df_diff, aes(x = Window, y = Diff)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey30") +
  geom_boxplot(fill = color_dict[["HPIN"]], alpha = 0.75, width = 0.6,
               outlier.alpha = 0.25, outlier.size = 0.5, lwd = 0.3) +
  coord_cartesian(ylim = quantile(df_diff$Diff, c(0.02, 0.98))) +
  theme_minimal(base_size = 12) +
  labs(x = "Window size", y = "PLS difference (HPIN - PIN)") +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(color = "grey90", linewidth = 0.3),
        plot.background = element_rect(fill = "white", color = NA))
ggsave(file.path(fig_dir, "PLS_diff_boxplot.pdf"), pls_diff_plot,
       width = 7.5, height = 4.5, units = "in")
cat("pls difference ok\n")

# --- 9. transition metrics --------------------------------------------------
load_and_tidy <- function(model, w) {
  f <- file.path(tab_dir, paste0("transition_metrics_", model, "_", w, ".csv"))
  if (!file.exists(f)) return(NULL)
  read.csv(f, row.names = 1, check.names = FALSE) %>%
    mutate(Window = as.character(w), Model = model) %>%
    pivot_longer(cols = -c(Window, Model), names_to = "Metric", values_to = "Value")
}
df_list <- list()
for (model in c("HPIN", "PIN")) for (w in windows)
  df_list <- append(df_list, list(load_and_tidy(model, w)))
df_tm <- bind_rows(df_list) %>%
  mutate(Metric_Display = str_replace_all(Metric, "_", " "),
         Window = factor(Window, levels = as.character(windows)))

transition_plot <- ggplot(df_tm, aes(x = Window, y = Value, fill = Model)) +
  geom_boxplot(position = position_dodge(width = 0.8), width = 0.7,
               outlier.alpha = 0.5, lwd = 0.3) +
  facet_wrap(~ Metric_Display, nrow = 2, ncol = 2, scales = "free_y") +
  scale_fill_manual(values = color_dict, name = "") +
  theme_minimal(base_size = 12) +
  labs(x = "Window", y = "Values Distribution", fill = "Model") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        strip.text = element_text(size = 10, face = "bold"),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(color = "grey90", linewidth = 0.3),
        panel.grid.minor = element_blank(), legend.position = "bottom")
ggsave(file.path(fig_dir, "markov_dynamics_metrics.pdf"), transition_plot,
       width = 7.5, height = 5.5, units = "in")
cat("transition metrics ok\n")

# --- 10. Persistence grid: estimated vs true spectral gap -------------------
grid_file <- file.path(tab_dir, "persistence_grid.csv")
if (file.exists(grid_file)) {
  g <- read.csv(grid_file)
  g <- g[g$window == 180, ]
  g$dgp_label <- ifelse(g$dgp == "clean", "Correctly specified",
                        "Overdispersed emissions")

  hpin_pts <- data.frame(true = g$spectral_gap_true, est = g$gap_hpin,
                         dgp_label = g$dgp_label, Model = "HPIN")
  pin_pts <- data.frame(true = g$spectral_gap_true, est = 1.0,
                        dgp_label = g$dgp_label, Model = "PIN")
  pts <- rbind(hpin_pts, pin_pts)

  grid_plot <- ggplot(pts, aes(x = true, y = est, color = Model, shape = Model)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey40") +
    geom_line(linewidth = 0.6) +
    geom_point(size = 2.2) +
    facet_wrap(~ dgp_label, nrow = 1) +
    scale_color_manual(values = color_dict, name = "Model") +
    scale_shape_manual(values = c("HPIN" = 16, "PIN" = 17), name = "Model") +
    coord_equal(xlim = c(0, 1.02), ylim = c(0, 1.02)) +
    theme_minimal(base_size = 12) +
    labs(x = "True spectral gap  (1 - rho)", y = "Estimated spectral gap") +
    theme(strip.text = element_text(size = 10, face = "bold"),
          panel.grid.minor = element_blank(),
          legend.position = "bottom",
          plot.background = element_rect(fill = "white", color = NA))
  ggsave(file.path(fig_dir, "persistence_grid.pdf"), grid_plot,
         width = 7.5, height = 4.2, units = "in")
  cat("persistence grid ok\n")
} else {
  cat("persistence grid SKIPPED (run persistence_grid.py aggregate first)\n")
}

# --- copy figures into the manuscript directory, when present ---------------
paper_dir <- file.path(here, "..", "..", "paper")
if (dir.exists(paper_dir)) {
  pdfs <- list.files(fig_dir, pattern = "\\.pdf$", full.names = TRUE)
  ok <- file.copy(pdfs, paper_dir, overwrite = TRUE)
  cat(sprintf("copied %d/%d figures to %s\n", sum(ok), length(pdfs),
              normalizePath(paper_dir)))
} else {
  cat("paper directory not present; figures left in results/figs\n")
}
