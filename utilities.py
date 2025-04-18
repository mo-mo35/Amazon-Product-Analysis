from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import SQLContext
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T
import databricks.koalas as ks
import numpy as np
import os
import traceback
from collections import Mapping
from math import isclose
SEED = 102
TASK_NAMES = ['task_' + str(i) for i in range(1, 9)]
EXT = '.json'
MASTER_IP = 'spark://0.0.0.0:7077'


class data_cat:
    root_folder = '/datasets/courses/dsc102/public/'
    review_filename = root_folder + 'user_reviews_train.csv'
    product_filename = root_folder + 'metadata_header.csv'
    product_processed_filename = root_folder + 'product_processed.csv'
    ml_features_train_filename = root_folder + 'ml_features_train.parquet'
    ml_features_test_filename = root_folder + 'ml_features_test.parquet'
    test_results_root = root_folder + 'test_results'


def quantile(rdd, p, sample=None, seed=SEED):
    """Compute a quantile of order p ∈ [0, 1]
    :rdd a numeric rdd
    :p quantile(between 0 and 1)
    :sample fraction of and rdd to use. If not provided we use a whole dataset
    :seed random number generator seed to be used with sample
    """
    assert 0 <= p <= 1
    assert sample is None or 0 < sample <= 1

    rdd = rdd if sample is None else rdd.sample(False, sample, seed)

    rddSortedWithIndex = (rdd.sortBy(lambda x: x).zipWithIndex().map(
        lambda x: (x[1], x[0])).cache())

    n = rddSortedWithIndex.count()
    h = (n - 1) * p

    rddX, rddXPlusOne = (rddSortedWithIndex.lookup(x)[0][0]
                         for x in int(np.floor(h)) + np.array([0, 1]))
    return rddX + (h - np.floor(h)) * (rddXPlusOne - rddX)


def test_deco(f):
    def f_new(*args, test_dict=None, **kwargs):
        count = test_dict['count']
        failures = test_dict['failures']
        total_count = test_dict['total_count']
        test_name = test_dict['test_name']
        count += 1
        print ('Test {}/{} : {} ... '.format(count, total_count, test_name), end='')  # noqa
        try:
            f(*args, **kwargs)
            print('Pass')
        except Exception as e:
            failures.append(e)
            print('Fail: {}'.format(e))
            traceback.print_exc()
        return count
    return f_new


class PA2Test(object):
    def __init__(self, spark, test_results_root):
        self.spark = spark
        self.test_results_root = test_results_root
        self.dict_res = {}
        for task_name in TASK_NAMES:
            try:
                df = self.spark.read.json(
                    os.path.join(test_results_root, task_name + EXT), )
                self.dict_res[task_name] = df.collect()[0].asDict()
            except Exception as e:
                print(e)
                traceback.print_exc()
        self.dict_res['task_0'] = {
            'count_total': 9430000, 'mean_price': 34.93735609456491}

    def test(self, res, task_name):
        row = 79
        start_msg = 'tests for {} '.format(task_name)
        comp_row = max(0, row - len(start_msg))
        comp_dashes = ''.join(['-' * comp_row])
        print (start_msg + comp_dashes)
        failures = []
        count = 0
        ref_res = self.dict_res[task_name]
        total_count = len(ref_res)
        test_dict = {
            'count': count,
            'failures': failures,
            'total_count': total_count,
            'test_name': None
        }
        if task_name in ['task_1', 'task_2', 'task_3', 'task_4']:
            for k, v in ref_res.items():
                test_name = k
                test_dict['test_name'] = test_name
                test_dict['count'] = count
                count = self.identical_test(
                    test_name, res[k], v, test_dict=test_dict)
        elif task_name in ['task_7', 'task_8']:
            for k, v in ref_res.items():
                test_name = k
                test_dict['test_name'] = test_name
                test_dict['count'] = count
                count = self.identical_test(
                    test_name, res[k], v, rel_tol=0.0, abs_tol=0.1, test_dict=test_dict)
        elif task_name == 'task_5':
            total_length = 10
            test_dict['total_count'] = 8
            at_least = 1
            ref_res_identical = {
                k: v
                for k, v in ref_res.items()
                if k in ['count_total', 'size_vocabulary']
            }

            for k, v in ref_res_identical.items():
                test_name = k
                test_dict['test_name'] = test_name
                test_dict['count'] = count
                count = self.identical_test(
                    test_name, res[k], v, test_dict=test_dict)

            for k in ['word_0_synonyms', 'word_1_synonyms', 'word_2_synonyms']:
                res_v = dict(res[k])
                res_r = dict(map(tuple, ref_res[k]))
                test_name = '{}-length'.format(k)
                test_dict['test_name'] = test_name
                test_dict['count'] = count
                count = self.identical_length_test(
                    k, res_v, list(range(total_length)), test_dict=test_dict)

                test_name = '{}-correctness'.format(k)
                test_dict['test_name'] = test_name
                test_dict['count'] = count
                count = self.synonyms_test(
                    res_v, res_r, total_length, at_least, test_dict=test_dict)

        elif task_name == 'task_6':
            total_count = 9
            test_dict = {
                'count': count,
                'failures': failures,
                'total_count': total_count,
                'test_name': None
            }
            ref_res_identical = {
                k: v
                for k, v in ref_res.items() if k in ['count_total']
            }
            for k, v in ref_res_identical.items():
                test_name = k
                test_dict['test_name'] = test_name
                test_dict['count'] = count
                count = self.identical_test(
                    test_name, res[k], v, test_dict=test_dict)
            for k in ['meanVector_categoryOneHot', 'meanVector_categoryPCA']:
                res_v = np.abs(res[k])
                res_r = np.abs(ref_res[k])
                test_name = '{}-length'.format(k)
                test_dict['test_name'] = test_name
                test_dict['count'] = count
                count = self.identical_length_test(
                    k, res_v, res_r, test_dict=test_dict)

                for fname, fns in zip(('sum', 'mean', 'variance'),
                                      (np.sum, np.mean, np.var)):
                    test_name = '{}-{}'.format(k, fname)
                    test_dict['test_name'] = test_name
                    test_dict['count'] = count
                    vv = fns(res_v)
                    vr = fns(res_r)
                    count = self.identical_test(
                        test_name, vv, vr, test_dict=test_dict)

        print('{}/{} passed'.format(total_count - len(failures), total_count))
        print (''.join(['-' * row]))
        return len(failures) == 0

    @test_deco
    def identical_length_test(self, k, v1, v2):
        size1 = len(v1)
        size2 = len(v2)
        assert size1 == size2, \
            'Length of {} must be {}, but got {} instead'.format(
                k, size2, size1)

    @test_deco
    def identical_test(self, k, v1, v2, rel_tol=1e-2, abs_tol=0.0):
        assert isclose(
            v1, v2, rel_tol=rel_tol, abs_tol=abs_tol), \
            'Value of {} should be close enough to {}, but got {} instead'.format(
            k, v2, v1)

    @test_deco
    def synonyms_test(self, res_v, res_r, total, at_least):
        if sum(res_v.values()) / len(res_v) < 0.9:
            print(
                "WARNING: your top synonyms have an average score less than 0.9, this might indicate errors"
            )
        correct = len(set(res_v.keys()).intersection(set(res_r.keys())))
        assert correct >= at_least, \
            'At least {} synonyms out of {} should overlap with our answer, got only {} instead'. \
            format(at_least, total, correct)

def get_local_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip

def spark_init(pid):
    driver_host = get_local_ip()
    conf = pyspark.SparkConf()
    conf.setAppName(pid)
    conf.setMaster('spark://spark-master-svc:7077') 
    conf.set("spark.blockmanager.port", "50002")
    conf.set("spark.driver.bindAddress", driver_host)
    conf.set("spark.driver.host", driver_host)
    conf.set("spark.driver.port", "50500")
    conf.set("spark.cores.max", "6")
    conf.set("spark.executor.memory", "18G")
    conf.set('spark.authenticate', False)
    conf.set("spark.driver.memory", "8G")
    conf.set("spark.driver.extraLibraryPath", "/opt/hadoop/lib/native")
    conf.set("spark.fileserver.port", "6002")
    conf.set("spark.sql.crossJoin.enabled", "true")
    conf.set("spark.memory.fraction", "0.90")
    conf.set("spark.dynamicAllocation.enabled", 'false')
    conf.set("spark.broadcast.port", "60003")
    conf.set("spark.replClassServer.port", "60004")
    conf.set('spark.authenticate', False)
    
    spark = SparkContext(conf=conf)

    return SQLContext(spark)


class PA2Data(object):
    review_schema = T.StructType([
        T.StructField('reviewerID', T.StringType(), False),
        T.StructField('asin', T.StringType(), False),
        T.StructField('overall', T.FloatType(), False)
    ])
    product_schema = T.StructType([
        T.StructField('asin', T.StringType()),
        T.StructField('salesRank', T.StringType()),
        T.StructField('categories', T.StringType()),
        T.StructField('title', T.StringType()),
        T.StructField('price', T.FloatType()),
        T.StructField('related', T.StringType())
    ])
    product_processed_schema = T.StructType([
        T.StructField('asin', T.StringType()),
        T.StructField('title', T.StringType()),
        T.StructField('category', T.StringType())
    ])
    salesRank_schema = T.MapType(T.StringType(), T.IntegerType())
    categories_schema = T.ArrayType(T.ArrayType(T.StringType()))
    related_schema = T.MapType(T.StringType(), T.ArrayType(T.StringType()))
    schema = {
        'review': review_schema,
        'product': product_schema,
        'product_processed': product_processed_schema
    }
    metadata_schema = {
        'salesRank': salesRank_schema,
        'categories': categories_schema,
        'related': related_schema
    }

    def __init__(self,
                 spark,
                 path_dict,
                 output_root,
                 deploy,
                 input_format='dataframe'
                 ):
        self.spark = spark
        self.path_dict = path_dict
        self.output_root = output_root
        self.deploy = deploy
        self.input_format = input_format

    def load(self, name, path, infer_schema=False):
        if name in ['ml_features_train', 'ml_features_test']:
            data = self.spark.read.parquet(path)
        else:
            schema = self.schema[name] if not infer_schema else None
            data = self.spark.read.csv(
                path,
                schema=schema,
                escape='"',
                quote='"',
                inferSchema=infer_schema,
                header=True
            )
        if name == 'product':
            for column, column_schema in self.metadata_schema.items():
                if column in data.columns:
                    data = data.withColumn(column, F.from_json(
                        F.col(column), column_schema))
        return data

    def load_all(self, input_format='dataframe', no_cache=False):
        self.input_format = input_format
        print ("Loading datasets ...", end='')  # noqa
        data_dict = {}
        count_dict = {}
        for name, path in self.path_dict.items():

            data = self.load(name, path)
            if input_format == 'rdd':
                data = data.rdd
            elif input_format == 'koalas':
                data = data.to_koalas()
            if self.deploy and not no_cache:
                data = data.cache()
            data_dict[name] = data
            count_dict[name] = data.count() if not no_cache else None
        print ("Done")
        return data_dict, count_dict

    def cache_switch(self, data_dict, part):
        count_dict = {}
        if self.input_format == 'koalas':
            print('cache_switch() has no effect on Koalas')
        else:
            part_1_data = ['product', 'review', 'product_processed']
            part_2_data = ['ml_features_train', 'ml_features_test']
            if part == 'part_1':
                data_dict, count_dict = self.switch(data_dict, part_1_data, part_2_data)
            elif part == 'part_2':
                data_dict, count_dict = self.switch(data_dict, part_2_data, part_1_data)
            else:
                raise ValueError
        return data_dict, count_dict

    def switch(self, data_dict, to_persist, to_unpersist):
        count_dict = {}
        for name in to_unpersist:
            try:
                data_dict[name].unpersist()
            except Exception as e:
                pass
        for name in to_persist:
            data_dict[name] = data_dict[name].cache()
            count_dict[name] = data_dict[name].count()
        return data_dict, count_dict

    def save(self, res, task_name, filename=None):
        if task_name in TASK_NAMES or task_name in ['task_0', 'summary']:
            if not filename:
                filename = task_name
            if isinstance(res, Mapping):
                df = self.spark.createDataFrame([res])
            else:
                df = self.spark.createDataFrame(res)
            output_path = 'file://' + os.path.join(
                self.output_root, filename + EXT)
            df.coalesce(1).write.mode('overwrite').json(output_path)
        else:
            raise ValueError
