# custom_packages

It is a repository with custom packages I have created for further ongoing projects.

1. AwsHandler.py
- It is a custom handler for uploading and downloading from local server to AWS s3.
- example how it is used
```
# create an object
aws_handler = AWSHandler(accessKey, secretKey, region, bucket)

# upload
aws_handler.upload_2S3(s3_path, local_path)

# download
aws_handler.download_fromS3(s3_path, local_path)
```

2. Metric.py
- It is a custom handler for calculating confusion matrix, precision, recall, weighted and macro f1 score for model performance evaluation.
- Mainly using only numpy to increase the speed for metric calculation.
- example how it is used
```
# create an object
metric = Metrics(arr_ytrue, arr_ypred)

# calculate confusion matrix
cm = metric.confusion_matrix

# calculate precision and recall
precision, recall = metric.fit(cm)

# macro f1
macro_f1 = metric.f1(arr_ytrue, precision, recall)

# weighted f1
weighted_f1 = metric.f1(arr_ytrue, precision, recall, "weighted")
```

3. SpamFeaturizer.py
- It is a custom handler to creating the spam features for vectorization and differentiating spam and non-spam posts.
- Multi-processing with 4 cpu are used for creating the features.
- generator is returned in order to prevent large memory space consumption.
- example how it is used
```
# create an object
spamer = Spamer(arr_msg)

# generate spam features
l_spam_feature = next(spamer.fit_transform())
```

4. ZipHandler.py
- It is a custom handler to do zip and upzip of model files
- Switcher is used in order to satisfy O(1) data access and increase speed in zip and unzip process
- On the other hand, not only .tar.gz, zip, gzip, etc can be added to the switcher and user can choose custom method for zipping and unzipping
- example how it is used
```
# create an object
zip_handler = ZipHandler(model_file_name, zip_type)

# zip
zip_helper.compressor(file_path, output_path)

# unzip
zip_helper.decompressor(file_path, output_path)
```
