import ast
import calendar
import statistics as stats
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import re
import cv2
import numpy as np
import pandas as pd
from kafka import KafkaConsumer
import json
from datetime import datetime
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import load_model
import tensorflow as tf
import oci
import io
import boto3

s3 = boto3.resource('s3',
    aws_access_key_id="6669dc2d16b7f3e78f87e3ca1b49a1778a66f0f9",
    aws_secret_access_key="v/pyfJvciV/JsAALQp1e8RxFiQhghpNjVYf3NUCut2g=",
    region_name="us-ashburn-1",  # Region name here that matches the endpoint
    endpoint_url="https://id02wscfvd2x.compat.objectstorage.us-ashburn-1.oraclecloud.com"
    # Include your namespace in the URL)
)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)
config = oci.config.from_file()
object_storage_client = oci.object_storage.ObjectStorageClient(config)
namespace = object_storage_client.get_namespace().data
session = boto3.session.Session()


class EmoAgeGenDetector:
    def __init__(self):
        """Loading the age and gender model into the memory"""
        print("Object created")
        self.load_classification_model_gender()
        self.nyka_load_classification_model_gender()
        self.load_classification_model_emotion()
        self.kid_adult_classification()
        self.kid_age_classification()
        self.adult_age_classification()

    def preprocess_image_emotion(self, img):
        """preprocessing images on which age and gender is trained"""
        try:
            norm_img = np.array(img).astype('float32') / 255.
            norm_img_resized = cv2.resize(norm_img, (150, 150), interpolation=cv2.INTER_AREA)
            pred_img = np.expand_dims(norm_img_resized, axis=0)
        except Exception as e:
            log_string = 'Task Failed - Preprocessing classification images!'
            logger.exception(str('Exception occured in ' + log_string + '\nError message:' + str(e)))
        return pred_img

    def preprocess_image(self, img):
        """preprocessing images on which age and gender is trained"""
        try:
            norm_img = np.array(img).astype('float32') / 255.
            norm_img_resized = cv2.resize(norm_img, (128, 128), interpolation=cv2.INTER_AREA)
            pred_img = np.expand_dims(norm_img_resized, axis=0)
        except Exception as e:
            log_string = 'Task Failed - Preprocessing classification images!'
            logger.exception(str('Exception occured in ' + log_string + '\nError message:' + str(e)))
        return pred_img
    def kid_adult_classification(self):
        filepath = 'age/kid_adult_model_arch.txt'
        self.classifier = load_model(filepath)
        model1 = self.classifier.load_weights('age/kid_adult_weights.h5')

    def adult_age_classification(self):
        arch_path = 'age/adult_model_arch.pb'
        self.adult_classifier = load_model(arch_path)
        model2 = self.adult_classifier.load_weights('age/adult_age_group_classification.h5')

    def kid_age_classification(self):
        arch2_path = 'age/kid_model_arch.pb'
        self.kid_classifier = load_model(arch2_path)
        model3 = self.kid_classifier.load_weights('age/kid_age_group_classification.h5')

    def load_classification_model_gender(self):
        """Loading the Gender weights onto architecture file"""
        try:
            cnn_arch_str = open('gender/gender_emp_cnn.txt','r+')
            self.cnn_model = tf.keras.models.model_from_json((cnn_arch_str.read()))
            self.cnn_model.load_weights( 'gender/gender_emp.h5')

        except Exception as e:
            log_string = 'Task Failed - Loading gender classification model!'
            logger.exception(str('Exception occured in ' + log_string + '\nError message:' + str(e)))

    def gender_prediction(self, image, g_labels):
        """predicting the gender as per our label"""
        color_body = self.preprocess_image(image)
        gender_prediction = self.cnn_model.predict(color_body)
        gender = g_labels[np.argmax(gender_prediction)]
        return gender

    def nyka_load_classification_model_gender(self):
        try:
            nyka_arch_str = open('gender/nyka_modelcnn.txt', 'r+')
            self.nyka_model = tf.keras.models.model_from_json((nyka_arch_str.read()))
            self.nyka_model.load_weights('gender/nyka_model.h5')

        except Exception as e:
            log_string = 'Task Failed - Loading gender nyka classification model!'
            logger.exception(str('Exception occured in ' + log_string + '\nError message:' + str(e)))

    def nyka_gender_prediction(self, image, g_labels):
        nyka_color_body = self.preprocess_image(image)
        nyka_genderprediction = self.nyka_model.predict(nyka_color_body)
        nyka_gender = g_labels[np.argmax(nyka_genderprediction)]
        return nyka_gender
    def load_classification_model_emotion(self):
        """Loading the age weights onto architecture file"""
        try:
            cnn_arch_str_emotion = open('emotion/emotion_150X150resnet_cnn.txt','r+')
            self.cnn_model_emotion = tf.keras.models.model_from_json((cnn_arch_str_emotion.read()))
            self.cnn_model_emotion.load_weights( 'emotion/emotion_150x150resnet.h5')

        except Exception as e:
            log_string = 'Task Failed - Loading age classification model!'
            logger.exception(str('Exception occured in ' + log_string + '\nError message:' + str(e)))

    def emotion_prediction(self, image, e_labels):
        """predicting the sentiment as per our label"""
        color_body = self.preprocess_image_emotion(image)
        emotion_prediction = self.cnn_model_emotion.predict(color_body)
        emotion = e_labels[np.argmax(emotion_prediction)]
        return emotion
    def age_prediction(self, image, a_labels):
        """predicting the age as per our label"""
        color_body = self.preprocess_image(image)
        prediction = np.argmax(self.classifier.predict(color_body), axis=-1)
        if a_labels[prediction[0]] == 'kid':
            kid_prediction = np.argmax(self.kid_classifier.predict(color_body), axis=-1)
            age = kid_labels[kid_prediction[0]]
        elif a_labels[prediction[0]] == 'adult':
            adult_prediction = np.argmax(self.adult_classifier.predict(color_body), axis=-1)
            age = adult_labels[adult_prediction[0]]
        return age

    def Extract(self, lst):
        """function to extract tempid from querry"""
        return sorted(list([item[0] for item in lst]))

    def get_secrets_value(self, secrets_param):

        client = session.client(
            service_name='secretsmanager',
            region_name="us-east-1"
        )
        get_secret_value_response = client.get_secret_value(
            SecretId="oracle_details"
        )
        secrets = json.loads(get_secret_value_response['SecretString'])
        secrets_param_value = secrets[secrets_param]
        return secrets_param_value

    def consumer_message(self, kafka_sasl_plain_username, kafka_age_gender_group_id, kafka_sasl_plain_password):
        print("######INSIDE CONSUMER MESSAGE #########")
        try:
            print('connecting to consumer')
            consumer = KafkaConsumer(
                bootstrap_servers='cell-1.streaming.us-ashburn-1.oci.oraclecloud.com:9092',
                security_protocol='SASL_SSL', sasl_mechanism='PLAIN',
                auto_offset_reset='earliest',
                # group_id="aws_redshift_age_gender",
                group_id=kafka_age_gender_group_id,
                sasl_plain_username=kafka_sasl_plain_username,
                sasl_plain_password=kafka_sasl_plain_password,
                api_version=(0, 10, 1))
            return consumer

        except Exception as err:
            logger.error(err)

    def camera_number(self,camera_ip_num):
        try:
            ip = re.findall(r'[0-9]+(?:\.[0-9]+){3}', camera_ip_num)
            port = re.split("/", camera_ip_num)
            if len(port) != 1:
                port = port[-1:][0]
                temp = re.findall(r'[0-9]', port)
                if len(temp) >= 3:
                    camera_ip = ip[0] + "."
                    for i in range((len(temp) - 2)):
                        camera_ip = camera_ip + str(temp[i])
                    camera_ip_num = camera_ip
                else:
                    camera_ip_num = ip[0] + "." + str(temp[0])
            else:
                camera_ip_num = port[0]

            return camera_ip_num
        except Exception:
            logger.exception("Exception occurred")

    def file_upload_process(self, features_li, file_name, store_id, edge_date_rev, s3_cli, bucket_name):
        try:
            final_df = pd.DataFrame()
            df = pd.DataFrame(features_li)
            final_df = final_df.append(df, ignore_index=True)
            filename = file_name.split(".")
            final_df.to_pickle("{}/{}_{}_{}.zip".format(bucket_name, store_id, edge_date_rev,filename[0]), compression='zip')
            print('{}/{}/{}.zip'.format(store_id, edge_date_rev, filename[0]))
            try:
                s3_cli.upload_file("{}_{}_{}.gzip".format(store_id, edge_date_rev, filename[0]), bucket_name,'{}/{}/{}.gzip'.format(store_id,edge_date_rev, filename[0]))
                os.remove("{}_{}_{}.gzip".format(store_id,edge_date_rev, filename[0]))
            except Exception as err:
                logger.error(err)
        except Exception as err:
            logger.error(err)

    def df_iterate(self,unpickled_df,store_id,client_id,edge_date_rev,cust_beha_li, file_name,s3_cli,redshift_raw_data):
            age = 0
            gender = ''
            sentiment_heatmap_li = []
            for _, row in unpickled_df.iterrows():
                """iterating over each row of dataframe"""
                if row['temp_id'] < 10000:  # to predict only for customers
                    try:
                        row_sentiment = []
                        row_timestamp = []
                        for i in range(len(row['timestamp'])):
                            try:
                                if i <= 2:  # taking only three images for prediction
                                    if store_id[0].startswith("4-"):
                                        gender = self.nyka_gender_prediction(row['reid_img'][i],
                                                                             gender_labels)
                                    else:
                                        gender = self.gender_prediction(row['reid_img'][i],
                                                                        gender_labels)
                                    ag_value = self.age_prediction(row['reid_img'][i], class_labels)

                                    senti = self.emotion_prediction(row['reid_img'][i],
                                                                    emotion_labels)
                                    row_sentiment.append(senti)
                                    row_timestamp.append(int(row['timestamp'][i]))
                                    camera_number = self.camera_number(str(row['camera_num'][i]))
                                    person_str = np.asarray(row['person_coor'][i]).astype(
                                        np.str).tolist()
                                    person_dict = ','.join(map(str, person_str))
                                    sentiment_heatmap_dic = {}
                                    sentiment_heatmap_dic['inserted_at'] = int(row['timestamp'][i])
                                    sentiment_heatmap_dic['store_id'] = store_id
                                    sentiment_heatmap_dic['client_id'] = int(client_id[0])
                                    sentiment_heatmap_dic['store_date'] = str(edge_date_rev)
                                    sentiment_heatmap_dic['sentiment'] = senti
                                    sentiment_heatmap_dic['temp_id'] = row['temp_id']
                                    sentiment_heatmap_dic['person_cood'] = person_dict
                                    sentiment_heatmap_dic['camera_number'] = camera_number
                                    sentiment_heatmap_li.append(sentiment_heatmap_dic)

                                    redhshift_pred_dic = {'age': ag_value, 'gender': gender,
                                                          'temp_id': row['temp_id'],
                                                          'tango_id': 'C',
                                                          'neutral_count': int(
                                                              row_sentiment.count('neutral')),
                                                          'unhappy_count': int(
                                                              row_sentiment.count('unhappy')),
                                                          'happy_count': int(
                                                              row_sentiment.count('happy')),
                                                          'inserted_at': int(row['timestamp'][i])}
                                    print(redhshift_pred_dic)

                                    # raw_data = raw_data.append(pred_dic, ignore_index=True)
                                    redshift_raw_data = redshift_raw_data.append(redhshift_pred_dic,
                                                                                 ignore_index=True)
                                else:
                                    break

                                ####### sentiment prediction

                            except Exception as err:
                                logger.error(err)

                    except Exception as err:
                        logger.error(err)
                else:
                    continue

            redshift_gb = redshift_raw_data.groupby('temp_id')
            for name, group in redshift_gb:
                try:
                    ag = stats.mode(group['age'])
                except Exception as e:
                    print(e)
                    ag = list(group['age'])[0]
                try:
                    gen = stats.mode(group['gender'])
                except Exception as e:
                    print(e)
                    gen = list(group['gender'])[0]
                try:
                    happy_count = stats.mode(group['happy_count'])
                except Exception as e:
                    print(e)
                    happy_count = list(group['happy_count'])[0]
                try:
                    unhappy_count = stats.mode(group['unhappy_count'])
                except Exception as e:
                    print(e)
                    unhappy_count = list(group['unhappy_count'])[0]
                try:
                    neutral_count = stats.mode(group['neutral_count'])
                except Exception as e:
                    print(e)
                    neutral_count = list(group['neutral_count'])[0]

                processed_at = datetime.utcnow()
                processed_at = int(calendar.timegm(processed_at.utctimetuple()))

                cust_beha_di = {}
                cust_beha_di['processed_at'] = int(processed_at)
                cust_beha_di['client_id'] = int(client_id[0])
                cust_beha_di['store_id'] = str(store_id)
                cust_beha_di['partition_date'] = str(edge_date_rev)
                cust_beha_di['happy_count'] = int(happy_count)
                cust_beha_di['unhappy_count'] = int(unhappy_count)
                cust_beha_di['neutral_count'] = int(neutral_count)
                cust_beha_di['age'] = str(ag)
                cust_beha_di['gender'] = str(gen)
                cust_beha_di['tango_id'] = 'C'
                cust_beha_di['temp_id'] = int(name)
                cust_beha_di['inserted_at'] = int(group['inserted_at'].agg(np.min))
                cust_beha_li.append(cust_beha_di)
                print("cust_beha_di")
                print("############")
                print(cust_beha_di)
                print("############")

            self.file_upload_process(cust_beha_li, file_name, store_id, edge_date_rev, s3_cli,'customer_behaviour_data')

    def process(self):

        global newlist
        kafka_fr_topic = self.get_secrets_value('kafka_fr_topic')
        print(kafka_fr_topic)
        kafka_age_gender_group_id = "testing"
        kafka_sasl_plain_username = self.get_secrets_value('kafka_sasl_plain_username')
        kafka_sasl_plain_password = self.get_secrets_value('kafka_sasl_plain_password')
        oracle_access_key = self.get_secrets_value('oracle_access_key')
        oracle_secret_access_key = self.get_secrets_value('oracle_secret_access_key')
        oracle_region = self.get_secrets_value('oracle_region')
        oracle_endpoint_url = self.get_secrets_value('oracle_endpoint_url')
        fr_input_bucket = self.get_secrets_value('fr_input_bucket')

        consumer = self.consumer_message(kafka_sasl_plain_username, kafka_age_gender_group_id,
                                         kafka_sasl_plain_password)
        s3_cli = boto3.client(
            's3',
            aws_access_key_id=oracle_access_key,
            aws_secret_access_key=oracle_secret_access_key,
            region_name=oracle_region,
            endpoint_url=oracle_endpoint_url
        )

        try:
            while True:
                try:
                    consumer.subscribe(['fr_prod'])
                    msg_pack = consumer.poll(max_records=1)
                    consumer.commit()
                except Exception:

                    consumer.subscribe(['fr_prod'])
                    msg_pack = consumer.poll(max_records=1)

                for _, msgs in msg_pack.items():
                    for msg in msgs:
                        print(msg)
                        redshift_raw_data = pd.DataFrame(columns=['temp_id', 'tango_id', 'age', 'gender', 'inserted_at', 'sentiment'])
                        s3_data = ast.literal_eval(msg.value.decode('utf-8'))
                        print(s3_data)
                        file_path = s3_data['data']['resourceId']
                        print(file_path)
                        key_path =  file_path.split("o/")
                        cust_beha_li = []
                        logger.info(f'File {key_path[1]} is processing')
                        file_name = re.split("/", key_path[1])[-1]
                        try:
                            s3_cli.download_file(fr_input_bucket, key_path[1], file_name)
                        except Exception as err:
                            fr_input_bucket = self.get_secrets_value('fr_input_bucket')
                            oracle_access_key = self.get_secrets_value('oracle_access_key')
                            oracle_secret_access_key = self.get_secrets_value('oracle_secret_access_key')
                            oracle_region = self.get_secrets_value('oracle_region')
                            oracle_endpoint_url = self.get_secrets_value('oracle_endpoint_url')
                            s3_cli = boto3.client(
                                's3',
                                aws_access_key_id=oracle_access_key,
                                aws_secret_access_key=oracle_secret_access_key,
                                region_name=oracle_region,
                                endpoint_url=oracle_endpoint_url
                            )
                            s3_cli.download_file(fr_input_bucket, key_path[1], file_name)
                            logger.error("Exception in oracle bucket downloading ", err)

                        unpickled_df = pd.read_pickle(file_name)  # reading dataframe
                        os.remove(file_name)  # removing the dataframe from local disk

                        edge_date = str(re.split("/", key_path[1])[1])
                        store_id = str(re.split("/", key_path[1])[0])
                        client_id = store_id.split("-")
                        store_date_li = edge_date.split("-")
                        edge_date_rev = "{}-{}-{}".format(store_date_li[2],
                                                          store_date_li[1],
                                                          store_date_li[0])
                        try:
                            df = pd.DataFrame()
                            for response in oci.pagination.list_call_get_all_results_generator(
                                    object_storage_client.list_objects,
                                    'response',
                                    namespace, 'customer_behaviour_data',
                                    fields="size,timeCreated,timeModified,md5,etag",
                                    prefix="{}/{}/".format(store_id, edge_date_rev)):
                                for o in response.data.objects:
                                    obj = s3.Object('customer_behaviour_data', o.name)
                                    body = obj.get()['Body'].read()
                                    pq_file = io.BytesIO(body)
                                    data_point_df = pd.read_parquet(pq_file)
                                    df = data_point_df.append(data_point_df, ignore_index=True)
                            if df.empty is False:
                                print("not empty")
                                templist1 = df["temp_id"].tolist()
                                updated_unpickled_df = ~unpickled_df.temp_id.isin(templist1)
                                unpickled_updated_df = unpickled_df[updated_unpickled_df]
                                self.df_iterate(unpickled_updated_df, store_id, client_id, edge_date_rev, cust_beha_li,
                                           file_name, s3_cli,redshift_raw_data)
                            else:
                                print("empty")
                                self.df_iterate(unpickled_df, store_id, client_id, edge_date_rev, cust_beha_li,
                                                file_name, s3_cli,redshift_raw_data)
                        except Exception as err:
                            logger.info(f'File {file_name} completed successfully')
                            pass

        except Exception as err:
            logger.exception("Exception occurred ", err)
            pass


if __name__ == '__main__':
    log_filename = 'AgeGenderTesting.log'
    logging.getLogger().setLevel(logging.ERROR)
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    handler = TimedRotatingFileHandler(log_filename, when="midnight", backupCount=7)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    gender_labels = {0: 'female', 1: 'male'}
    # age_labels = {0: 'age13_19', 1: 'age1_12', 2: 'age20_30', 3: 'age31_45', 4: 'age46_60', 5: 'age60+'}
    #     age_labels = {0: 15, 1: 8, 2: 25, 3: 36, 4: 55, 5: 65}
    kid_labels = ['age1_12', 'age13_19']
    adult_labels = ['age20_30', 'age31_45', 'age46_60', 'age60+']
    class_labels = ['adult', 'kid']
    emotion_labels = {0: 'happy', 1: 'neutral', 2: 'unhappy'}
    detector = EmoAgeGenDetector()
    detector.process()
