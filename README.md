# Kaggle_TUT_Copper_Analysis
Second place submission to "TUT Copper Analysis Challenge" https://inclass.kaggle.com/c/copper-analysis

This readme will be updated soon!

Please refer to the model description PDF file to get more information about the competition and the best model solution.

Data can be downloaded from: https://inclass.kaggle.com/c/copper-analysis/data

Example run for reproducing the best model solution:

    python Copperium.py predict train/ train.csv test/ test.csv -C 0.00001 >  predictions_with_prob.tsv
    cut -f1,2 predictions_with_prob.tsv | sed 's/\t/,/' > predictions_no_prob.csv
    python Copperium.py retrain train/ train.csv test/ test.csv predictions_with_prob.tsv -C 0.00001 -t 0.7 >  predictions_with_prob_iter1.tsv
    cut -f1,2 predictions_with_prob_iter1.tsv | sed 's/\t/,/' > predictions_iter1_no_prob.csv