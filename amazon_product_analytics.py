#!/usr/bin/env python3
"""
Amazon Product Analytics

A polished Spark pipeline that ingests public SNAP Stanford product metadata and Amazon review data,
performs feature engineering, auto-generates ML train/test splits, runs regression models,
and writes per-stage metrics to Parquet.

Usage:
    python3 amazon_product_analytics.py \
        --meta-path https://snap.stanford.edu/data/amazon-meta.txt.gz \
        --reviews-path s3://amazon-reviews-pds/tsv/amazon_reviews_us_ALL_v1_00.tsv.gz \
        --output-dir ./output
"""
import argparse
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from utilities import SEED, DataIO
from pyspark.sql import types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Word2Vec,
    StringIndexer,
    OneHotEncoder,
    PCA,
    VectorAssembler
)
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.stat import Summarizer

# -------------------- Data loading helper --------------------
def load_snap_product_metadata(spark: SparkSession, path: str):
    """
    Parse SNAP amazon-meta.txt.gz into a DataFrame with columns:
    asin, salesRank, categories, title, price, related
    """
    sc = spark.sparkContext
    def parse_partition(lines):
        record = {}
        for line in lines:
            text = line.strip()
            if not text:
                if record:
                    yield record
                    record = {}
            else:
                k, v = text.split(':', 1)
                record[k.strip()] = v.strip()
        if record:
            yield record

    raw_rdd = sc.textFile(path).mapPartitions(parse_partition)
    def to_row(rec):
        asin = rec.get('ASIN') or rec.get('Id')
        price = float(rec.get('price', 0.0))
        sales = rec.get('salesrank')
        salesRank = {'default': int(sales)} if sales else {}
        cats = []
        rawcats = rec.get('categories', '')
        if rawcats:
            parts = [p for p in rawcats.split('|') if p]
            primary = [p[:p.index('[')] for p in parts if '[' in p]
            cats = [primary]
        related = {}
        rawrel = rec.get('related', '')
        for chunk in rawrel.split(';'):
            if ':' in chunk:
                k, vals = chunk.split(':', 1)
                related[k.strip()] = vals.strip().split()
        title = rec.get('title')
        return asin, salesRank, cats, title, price, related

    schema = T.StructType([
        T.StructField('asin', T.StringType(), False),
        T.StructField('salesRank', T.MapType(T.StringType(), T.IntegerType())),
        T.StructField('categories', T.ArrayType(T.ArrayType(T.StringType()))),
        T.StructField('title', T.StringType()),
        T.StructField('price', T.DoubleType()),
        T.StructField('related', T.MapType(T.StringType(), T.ArrayType(T.StringType())))
    ])
    return spark.createDataFrame(raw_rdd.map(to_row), schema)

# --------------- Feature engineering routines ---------------
def compute_review_statistics(data_io, review_df, product_df):
    stats = review_df.groupBy('asin').agg(
        F.avg('overall').alias('meanRating'),
        F.count('overall').alias('countRating')
    )
    df = product_df.join(stats, 'asin', 'left')
    res = {
        'total_products': df.count(),
        'avg_meanRating': df.agg(F.mean('meanRating')).first()[0],
        'var_meanRating': df.agg(F.variance('meanRating')).first()[0],
        'null_meanRating': df.filter(F.col('meanRating').isNull()).count(),
        'avg_countRating': df.agg(F.mean('countRating')).first()[0],
        'var_countRating': df.agg(F.variance('countRating')).first()[0],
        'null_countRating': df.filter(F.col('countRating').isNull()).count()
    }
    data_io.save(res, 'review_statistics')
    return stats  # return DataFrame for downstream use


def extract_category_sales_metrics(data_io, product_df):
    df = product_df.withColumn(
        'primaryCategory',
        F.when(
            (F.size('categories')>0)&(F.size(F.col('categories')[0])>0)&(F.col('categories')[0][0] != ''),
            F.col('categories')[0][0]
        )
    ).withColumn('bestSalesCategory', F.map_keys('salesRank')[0])
     .withColumn('bestSalesRank', F.map_values('salesRank')[0])
    res = {
        'total_products': df.count(),
        'avg_bestSalesRank': df.agg(F.mean('bestSalesRank')).first()[0],
        'var_bestSalesRank': df.agg(F.variance('bestSalesRank')).first()[0],
        'null_primaryCategory': df.filter(F.col('primaryCategory').isNull()).count(),
        'unique_primaryCategory': df.agg(F.countDistinct('primaryCategory')).first()[0],
        'null_bestSalesCategory': df.filter(F.col('bestSalesCategory').isNull()).count()
    }
    data_io.save(res, 'category_sales_metrics')


def extract_related_product_metrics(data_io, product_df):
    exploded = product_df.select('asin', F.explode_outer('related.also_viewed').alias('relatedAsin'))
    price_df = product_df.select(F.col('asin').alias('relatedAsin'), 'price')
    joined = exploded.join(price_df, 'relatedAsin', 'left')
    price_stats = joined.groupBy('asin').agg(F.avg('price').alias('avgRelatedPrice'))
    count_stats = exploded.groupBy('asin').agg(F.count('relatedAsin').alias('relatedCount'))
    df = price_stats.join(count_stats, 'asin', 'full')
    res = {
        'total_products': df.count(),
        'avg_relatedPrice': df.agg(F.mean('avgRelatedPrice')).first()[0],
        'var_relatedPrice': df.agg(F.variance('avgRelatedPrice')).first()[0],
        'avg_relatedCount': df.agg(F.mean('relatedCount')).first()[0],
        'var_relatedCount': df.agg(F.variance('relatedCount')).first()[0]
    }
    data_io.save(res, 'related_product_metrics')


def impute_missing_product_data(data_io, product_df):
    avg_p = product_df.agg(F.mean('price')).first()[0]
    med_p = product_df.approxQuantile('price',[0.5],0)[0]
    df = product_df.withColumn('priceImputedMean', F.coalesce('price', F.lit(avg_p)))
    df = df.withColumn('priceImputedMedian', F.coalesce('price', F.lit(med_p)))
    df = df.withColumn('titleFilled', F.when(F.col('title').isNull()| (F.col('title')==''),'unknown').otherwise(F.col('title')))
    res = {
        'total': df.count(),
        'mean_priceImputedMean': df.agg(F.mean('priceImputedMean')).first()[0],
        'mean_priceImputedMedian': df.agg(F.mean('priceImputedMedian')).first()[0],
        'unknown_titles': df.filter(F.col('titleFilled')=='unknown').count()
    }
    data_io.save(res, 'imputation_metrics')
    return df  # return processed DataFrame


def generate_title_embeddings(data_io, product_df, words):
    df = product_df.withColumn('titleTokens', F.split(F.lower('title'),' '))
    model = Word2Vec(vectorSize=16, minCount=100, inputCol='titleTokens', outputCol='embeddings', seed=SEED)
    fit = model.fit(df)
    res = {'total': df.count(), 'vocab_size': fit.getVectors().count()}
    for i, w in enumerate(words):
        syn = fit.findSynonyms(w, 10).collect()
        res[f'word_{i}_synonyms'] = [(r['word'], r['similarity']) for r in syn]
    data_io.save(res, 'title_embeddings')


def encode_and_reduce_category_features(data_io, product_df):
    pipeline = Pipeline(stages=[
        StringIndexer(inputCol='category', outputCol='catIndex'),
        OneHotEncoder(inputCol='catIndex', outputCol='catOHE', dropLast=False),
        PCA(k=15, inputCol='catOHE', outputCol='catPCA')
    ])
    out = pipeline.fit(product_df).transform(product_df)
    mean_ohe = out.select(Summarizer.mean('catOHE')).first()[0].toArray().tolist()
    mean_pca = out.select(Summarizer.mean('catPCA')).first()[0].toArray().tolist()
    res = {'total': out.count(), 'mean_ohe': mean_ohe, 'mean_pca': mean_pca}
    data_io.save(res, 'category_encoding')


def train_decision_tree_baseline(data_io, train_df, test_df):
    model = DecisionTreeRegressor(labelCol='label', featuresCol='features', maxDepth=5).fit(train_df)
    preds = model.transform(test_df)
    rmse = RegressionEvaluator(metricName='rmse', labelCol='label', predictionCol='prediction').evaluate(preds)
    res = {'rmse_baseline': rmse}
    data_io.save(res, 'dt_baseline')


def tune_decision_tree_model(data_io, train_df, test_df):
    train_s, valid = train_df.randomSplit([0.75,0.25], seed=SEED)
    evaluator = RegressionEvaluator(metricName='rmse', labelCol='label', predictionCol='prediction')
    depths = [5,7,9,12]
    metrics = {}
    best = (float('inf'), None)
    for d in depths:
        rmse_val = evaluator.evaluate(
            DecisionTreeRegressor(labelCol='label', featuresCol='features', maxDepth=d)
            .fit(train_s)
            .transform(valid)
        )
        metrics[f'valid_rmse_{d}'] = rmse_val
        if rmse_val < best[0]: best = (rmse_val, d)
    final_rmse = evaluator.evaluate(
        DecisionTreeRegressor(labelCol='label', featuresCol='features', maxDepth=best[1])
        .fit(train_df)
        .transform(test_df)
    )
    metrics['test_rmse'] = final_rmse
    data_io.save(metrics, 'dt_tuning')

# --------------------------- Entry point ---------------------------
def main():
    parser = argparse.ArgumentParser(description='Amazon Product Analytics')
    parser.add_argument('--meta-path', default='https://snap.stanford.edu/data/amazon-meta.txt.gz')
    parser.add_argument('--reviews-path', default='s3://amazon-reviews-pds/tsv/amazon_reviews_us_ALL_v1_00.tsv.gz')
    parser.add_argument('--output-dir', default='./output')
    args = parser.parse_args()

    spark = SparkSession.builder.appName('AmazonAnalytics').getOrCreate()
    data_io = DataIO(spark, args.output_dir)

    # Load data
    product_df = load_snap_product_metadata(spark, args.meta_path)
    review_df = spark.read.option('sep', '\t').option('header', True).csv(args.reviews_path) \
                     .select('reviewerID', 'asin', F.col('overall').cast('float'))

    # Stage 1-4: Feature engineering
    stats_df = compute_review_statistics(data_io, review_df, product_df)
    extract_category_sales_metrics(data_io, product_df)
    extract_related_product_metrics(data_io, product_df)
    processed_df = impute_missing_product_data(data_io, product_df)

    # Stage 5-6: Advanced features
    generate_title_embeddings(data_io, processed_df, ['piano','rice','laptop'])
    encode_and_reduce_category_features(data_io, processed_df)

    # Stage 7-8: ML pipeline (auto train/test)
    features_df = review_df.join(stats_df, 'asin', 'left')
    assembler = VectorAssembler(inputCols=['meanRating','countRating'], outputCol='features')
    ml_df = assembler.transform(features_df).withColumnRenamed('overall', 'label').select('features','label')
    train_df, test_df = ml_df.randomSplit([0.75, 0.25], seed=SEED)
    train_decision_tree_baseline(data_io, train_df, test_df)
    tune_decision_tree_model(data_io, train_df, test_df)

    spark.stop()

if __name__ == '__main__':
    main()
