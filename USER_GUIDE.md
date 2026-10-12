# Fuzzy Quadrant Mapper – User Guide

1. **Upload** a CSV or Excel file, one row per respondent. Only empty cells and NA/NaN/NULL count as missing.
2. **Sidebar 1 · Latent variables.** Name X (horizontal axis) and Y (vertical axis) and pick their items.
   Every scale must run from the lowest level (1) to the highest (K); all items are benefits.
3. **Step 1 · TFN conversion.** Type the TFN (a1, a2, a3) of each scale level in the table. The preset only
   fills the starting values; the default is Table 3 of the paper:
   1 → (0,0,30), 2 → (20,30,40), 3 → (30,50,70), 4 → (60,70,80), 5 → (70,100,100).
   Use one table for all items, one per latent variable, or one per item.
4. **Step 2 · Missing values.** The app lists empty and out-of-scale answers per item. Choose a replacement:
   - person-mean: mean of the respondent's other answers in the same latent variable, rounded to a scale level;
   - KNN hot-deck: median answer of the k most similar complete respondents;
   - item median or item mode;
   - listwise deletion.
   Respondents above the maximum share of missing items in X or Y are dropped. `n_imputed_items` marks
   imputed respondents, so the analysis can be repeated with listwise deletion as a robustness check.
5. **Sidebar 3 · Quadrants.** Names of the four quadrants and the Classic Apostle threshold
   (fixed value, sample median or sample mean).
6. **Sidebar 4 · Fuzzy clustering.** By default the three representative profiles are fixed as in the paper:
   Least = lowest individual TOPSIS, Intermediate = median, Most = highest (tied respondents averaged).
   Memberships use the fuzzy distance of Eq. (6b) with weights w1 (extremes) ≥ w2 (centre) and fuzzifier m.
   Level per axis with threshold α: 1 Least, 2 Undetermined, 3 Intermediate, 4 Most.
7. **Sidebar 5 · Groups and covariates.** Group TOPSIS averages the TFNs of each group (Eq. 1). PIS and NIS are
   taken across all groups of all grouping variables plus the Total (Table 4), or within each variable.
   Probability ratios are computed for the Pure categories by default; drivers have a 95% interval above 1,
   barriers below 1.
8. **Run analysis.** Results appear in tabs: individual TOPSIS, group TOPSIS (Figure 2 and Table 4 layouts),
   fuzzy clustering (ternary plots), Apostle models (4×4 heatmap, ECO-Extended quadrants, Pure categories) and
   probability ratios (Table 5 and Table A1 layouts). Each table and figure can be downloaded; the Export tab
   gives one Excel file with every table.
