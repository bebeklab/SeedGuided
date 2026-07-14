#############################################################
# compare_subnetworks_GO.R
#
# Runs GO enrichment (ORA) on each nSEA subnetwork (SN1, SN2, ...),
# each tested against ITS OWN ranked-gene background list, then:
#   1) Saves a separate dot plot per subnetwork (PDF + PNG)
#   2) Exports a clean Excel workbook with GO term, p.adjust, gene
#      counts, and gene names as a readable column (no more parsing
#      slash-separated gene ID strings by eye)
#
# EXPECTED INPUT (in results_nsea_dir):
#   cleaned_SN1_subnetwork.txt,   cleaned_SN1_ranked_genes.txt
#   cleaned_SN2_subnetwork.txt,   cleaned_SN2_ranked_genes.txt
#   ... one pair per subnetwork, same naming convention
#
# NOTE: each subnetwork is tested against its OWN background
# (its ranked_genes.txt), not a shared universe -- since your
# per-SN background lists differ, this is the statistically
# correct setup rather than compareCluster()'s single shared
# `universe` argument.
#
# HOW TO USE
#   1. Edit the CONFIG section below
#   2. Rscript compare_subnetworks_GO.R
#############################################################

## ---- CONFIG ------------------------------------------------
results_nsea_dir <- "~/SeedGuided/results_nsea"   # folder with cleaned_SN*_subnetwork.txt / cleaned_SN*_ranked_genes.txt
ontology          <- "BP"                          # "BP", "MF", "CC", or "ALL"
pvalue_cutoff     <- 0.05
qvalue_cutoff     <- 0.2
simplify_terms    <- TRUE                          # collapse redundant/near-duplicate GO terms
out_dir           <- "~/SeedGuided/results/GO_compareCluster"
show_n_terms      <- 15                            # top N terms shown per subnetwork's own plot
## --------------------------------------------------------------

suppressPackageStartupMessages({
  library(clusterProfiler)
  library(org.Hs.eg.db)
  library(openxlsx)
  library(stringr)
  library(ggplot2)
})

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
results_nsea_dir <- path.expand(results_nsea_dir)

## ---- 1. Discover SN subnetwork/background file pairs ----------
subnetwork_files <- list.files(
  results_nsea_dir,
  pattern = "^cleaned_SN[0-9]+_subnetwork\\.txt$",
  full.names = TRUE
)

if (length(subnetwork_files) == 0) {
  stop("No files matching 'cleaned_SN<N>_subnetwork.txt' found in ", results_nsea_dir)
}

extract_sn_name <- function(f) {
  str_extract(basename(f), "SN[0-9]+")
}

read_gene_list <- function(f) {
  x <- readLines(f) |> trimws()
  x[x != ""]
}

sn_names <- vapply(subnetwork_files, extract_sn_name, character(1))
subnetwork_files <- subnetwork_files[order(as.integer(str_remove(sn_names, "SN")))]
sn_names <- sn_names[order(as.integer(str_remove(sn_names, "SN")))]

cat("Found", length(subnetwork_files), "subnetwork(s):", paste(sn_names, collapse = ", "), "\n\n")

## ---- 2. Run enrichGO per subnetwork, each vs its own background ----
enrich_results <- list()

for (i in seq_along(subnetwork_files)) {
  sn <- sn_names[i]
  sub_file <- subnetwork_files[i]
  bg_file  <- file.path(results_nsea_dir, paste0("cleaned_", sn, "_ranked_genes.txt"))

  if (!file.exists(bg_file)) {
    warning("No matching background file for ", sn, " (expected ", bg_file, ") -- skipping")
    next
  }

  genes      <- read_gene_list(sub_file)
  background <- read_gene_list(bg_file)

  cat(sprintf("%-5s %4d genes | background %6d genes\n", sn, length(genes), length(background)))

  er <- enrichGO(
    gene          = genes,
    OrgDb         = org.Hs.eg.db,
    keyType       = "SYMBOL",
    ont           = ontology,
    universe      = background,
    pAdjustMethod = "BH",
    pvalueCutoff  = pvalue_cutoff,
    qvalueCutoff  = qvalue_cutoff
  )

  if (simplify_terms && ontology != "ALL" && !is.null(er) && nrow(as.data.frame(er)) > 0) {
    er <- simplify(er, cutoff = 0.7, by = "p.adjust", select_fun = min)
  }

  enrich_results[[sn]] <- er
}

# drop any subnetworks with no enrichResult (e.g. failed / empty)
enrich_results <- enrich_results[!vapply(enrich_results, is.null, logical(1))]

if (length(enrich_results) == 0) {
  stop("No subnetworks produced enrichment results -- check gene symbols / cutoffs.")
}

## ---- 3. Merge into a single compareCluster-style object (for Excel) ----
ck <- merge_result(enrich_results)

## ---- 4. Individual dot plot per subnetwork -------------------------
# One self-contained plot per SN (GeneRatio on x, term on y, size = Count,
# color = p.adjust) rather than cramming every subnetwork onto shared axes.
dotplot_dir <- file.path(out_dir, "dotplots_per_SN")
dir.create(dotplot_dir, showWarnings = FALSE, recursive = TRUE)

for (sn in sn_names[sn_names %in% names(enrich_results)]) {
  er <- enrich_results[[sn]]
  n_sig <- nrow(as.data.frame(er))

  if (n_sig == 0) {
    cat("  ", sn, ": no significant terms, skipping plot\n")
    next
  }

  n_shown <- min(show_n_terms, n_sig)

  p <- dotplot(er, showCategory = n_shown) +
    ggtitle(paste0(sn, " -- GO ", ontology, " enrichment")) +
    theme(axis.text.y = element_text(size = 9))

  plot_height <- max(4, 0.3 * n_shown + 2)

  ggsave(file.path(dotplot_dir, paste0("GO_dotplot_", sn, ".pdf")),
         plot = p, width = 8, height = plot_height)
}

cat("Per-SN dot plots written to", dotplot_dir, "\n")

## ---- 5. Export results to a clean Excel workbook --------------------
res_df <- as.data.frame(ck)

# turn "GENE1/GENE2/GENE3" into a readable, sortable column
res_df$GeneNames <- gsub("/", ", ", res_df$geneID)
res_df$geneID <- NULL

# reorder columns for readability
front_cols <- intersect(
  c("Cluster", "ID", "Description", "GeneRatio", "BgRatio",
    "pvalue", "p.adjust", "qvalue", "Count", "GeneNames"),
  names(res_df)
)
res_df <- res_df[, c(front_cols, setdiff(names(res_df), front_cols))]

# preserve numeric SN ordering (SN1, SN2, ... SN92) instead of alphabetical
res_df$Cluster <- factor(res_df$Cluster, levels = sn_names)

wb <- createWorkbook()

# Sheet 1: full results, one row per (cluster, GO term)
addWorksheet(wb, "All_GO_terms")
writeData(wb, "All_GO_terms", res_df, withFilter = TRUE)
freezePane(wb, "All_GO_terms", firstRow = TRUE)
setColWidths(wb, "All_GO_terms", cols = 1:ncol(res_df), widths = "auto")

# Sheet per subnetwork, so you can hand SN1's results to someone without the rest
for (nm in unique(res_df$Cluster)) {
  sub <- res_df[res_df$Cluster == nm, ]
  sheet_nm <- substr(nm, 1, 31)  # Excel sheet name limit
  addWorksheet(wb, sheet_nm)
  writeData(wb, sheet_nm, sub, withFilter = TRUE)
  freezePane(wb, sheet_nm, firstRow = TRUE)
  setColWidths(wb, sheet_nm, cols = 1:ncol(sub), widths = "auto")
}

out_xlsx <- file.path(out_dir, "GO_compareCluster_results.xlsx")
saveWorkbook(wb, out_xlsx, overwrite = TRUE)

cat("\nDone.\n")
cat("  Per-SN dot plots:", dotplot_dir, "\n")
cat("  Excel file:      ", out_xlsx, "\n")
