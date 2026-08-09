# Rolling-window classical PIN estimation (plan Etapa 3).
#
# - Estimator: PINstimation::pin_ea() — MLE with Ersan & Alici (2016) initial
#   parameter sets (grid), replacing the original single hand-picked initial
#   set: a stronger, literature-standard baseline.
# - Window convention aligned with the Python HPIN run: 0-based half-open
#   window [i-w, i) estimates the row stored under index i (the first
#   out-of-window observation). In 1-based R terms: rows (i-w+1):i, i = w..T-1
#   0-based -> rows (i0-w+1):(i0) with i0 = i+1.
# - Parallel over window positions (socket cluster).

.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressMessages(library(PINstimation))
suppressMessages(library(parallel))

args <- commandArgs(TRUE)
here <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
data_dir <- file.path(here, "..", "data", "fluxos", "sinteticos")
out_dir  <- file.path(here, "..", "data", "pins", "sinteticos")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

datasets <- if (length(args)) args else
  c("fluxos_sinteticos_personalizado", "fluxos_sinteticos_iid")
windows <- c(60, 90, 120, 150, 180)

n_cores <- max(1, detectCores() - 1)
cl <- makeCluster(n_cores)
invisible(clusterEvalQ(cl, {
  .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
  suppressMessages(library(PINstimation))
}))

fit_one <- function(i0, df, w) {
  # i0: 0-based index of the predicted observation. The estimation window is
  # the 0-based half-open range [i0-w, i0), i.e. 1-based rows (i0-w+1):i0.
  data <- df[(i0 - w + 1):i0, c("buyer", "seller")]
  res <- tryCatch(pin_ea(data, verbose = FALSE), error = function(e) NULL)
  if (is.null(res) || !length(res@pin)) {
    return(c(i = i0, alpha = NA, delta = NA, mu = NA, epsilon_b = NA,
             epsilon_s = NA, pin = NA, log_likelihood = NA))
  }
  p <- res@parameters
  c(i = i0, alpha = p[["alpha"]], delta = p[["delta"]], mu = p[["mu"]],
    epsilon_b = p[["eps.b"]], epsilon_s = p[["eps.s"]],
    pin = res@pin, log_likelihood = res@likelihood)
}

for (ds in datasets) {
  df <- read.csv(file.path(data_dir, paste0(ds, ".csv")))
  T_n <- nrow(df)
  for (w in windows) {
    out_file <- file.path(out_dir, sprintf("pin_ekop_%d_%s.csv", w, ds))
    if (file.exists(out_file)) {
      cat(sprintf("%s w=%d: output exists, skipping\n", ds, w))
      next
    }
    t0 <- Sys.time()
    idx <- w:(T_n - 1)             # 0-based predicted-observation indices
    clusterExport(cl, c("df", "w", "fit_one"), envir = environment())
    res <- parLapply(cl, idx, function(i0) fit_one(i0, df, w))
    out <- as.data.frame(do.call(rbind, res))
    write.csv(out, out_file, row.names = FALSE)
    cat(sprintf("%s w=%d: %d fits in %.1f min\n", ds, w, nrow(out),
                as.numeric(Sys.time() - t0, units = "mins")))
  }
}
stopCluster(cl)
