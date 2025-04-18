# Amazon-Product-Analysis


This repository contains a polished Apache Spark pipeline for Amazon product analytics. It ingests public SNAP Stanford product metadata and Amazon review data, performs feature engineering, generates ML train/test splits, trains decision tree models, and writes per-stage metrics to Parquet.

📂 Repository Structure

/amazon_product_analytics.py   Main Spark application
/utilities.py                 Helper functions (PA2Data loader, SEED, test harness)
/README.md                    Project overview and instructions

🚀 Quick Start

Prerequisites

Python 3.7+

Apache Spark (tested on Spark 3.x)

AWS EMR or local Spark cluster

Run the Pipeline

By default, the script reads from two public sources:

Product metadata: SNAP Stanford amazon-meta.txt.gz

Review data: Public S3 Amazon reviews TSV

python3 amazon_product_analytics.py \
    --meta-path https://snap.stanford.edu/data/amazon-meta.txt.gz \
    --reviews-path s3://amazon-reviews-pds/tsv/amazon_reviews_us_ALL_v1_00.tsv.gz \
    --output-dir ./output

All eight stages (feature engineering + ML) will run end-to-end without any additional downloads.

Local Dataset Usage (Optional)

If you prefer to download the raw files and run locally:

Download metadata:

wget https://snap.stanford.edu/data/amazon-meta.txt.gz -O data/amazon-meta.txt.gz

 Download reviews
wget https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_ALL_v1_00.tsv.gz -O data/reviews.tsv.gz


python3 amazon_product_analytics.py \
    --meta-path data/amazon-meta.txt.gz \
    --reviews-path data/reviews.tsv.gz \
    --output-dir ./output

Just point --meta-path and --reviews-path to your local files.

📦 Outputs

The script writes per-stage JSON/Parquet summaries into the --output-dir:

review_statistics/

category_sales_metrics/

related_product_metrics/

imputation_metrics/

title_embeddings/

category_encoding/

dt_baseline/

dt_tuning/

Each folder contains Parquet tables with the metrics for that stage.

📝 Customization

CLI flags allow you to override any path.

Utilities in utilities.py handle saving and seeding.

VectorAssembler uses meanRating and countRating by default; extend with additional features as needed.
