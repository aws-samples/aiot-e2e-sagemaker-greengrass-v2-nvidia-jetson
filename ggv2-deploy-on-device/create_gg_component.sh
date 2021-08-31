#!/bin/bash

echo ==--------Install Dedendencies---------==
# npm install     
sudo apt install jq
npm --version
jq --version
echo .

echo ==--------Configurations---------==
COMPONENT_NAME=$(cat config.json | jq -r '.Component.ComponentName')
COMPONENT_VERSION=$(cat config.json | jq -r '.Component.ComponentVersion')
ENABLE_SEND_MESSAGE=$(cat config.json | jq -r '.Component.SendMessage')
TOPIC=$(cat config.json | jq -r '.Component.Topic')
TOPIC2=${TOPIC//\//\\/}
PREDICTION_INTERVAL_SECS=$(cat config.json | jq -r '.Component.PredictionIntervalSecs')
TIMEOUT=$(cat config.json | jq -r '.Component.Timeout')

S3_BUCKET=$(cat config.json | jq -r '.Artifacts.S3Bucket')
S3_PREFIX=$(cat config.json | jq -r '.Artifacts.S3Prefix')
S3_PREFIX2=${S3_PREFIX//\//\\/}
ZIP_ARCHIVE_NAME=$(cat config.json | jq -r '.Artifacts.ZipArchiveName')

USE_GPU=$(cat config.json | jq -r '.Parameters.UseGPU')
SCORE_THRESHOLD=$(cat config.json | jq -r '.Parameters.ScoreThreshold')
MAX_NUM_CLASSES=$(cat config.json | jq -r '.Parameters.MaxNumClasses')
MODEL_INPUT_SHAPE=$(cat config.json | jq -r '.Parameters.ModelInputShape')

echo component name: $COMPONENT_NAME
echo component version: $COMPONENT_VERSION
echo topic: $TOPIC
echo zip archive name: $ZIP_ARCHIVE_NAME
echo .

# # json recipe
echo ==--------JSON Recipe---------==
cd recipes
OLD_RECIPE_JSON=$(ls ${COMPONENT_NAME}*.json)
NEW_RECIPE_JSON=${COMPONENT_NAME}-${COMPONENT_VERSION}.json
mv ${OLD_RECIPE_JSON} ${NEW_RECIPE_JSON}

sed -i "s/\(\"UseGPU\"\)\(.*\)/\1: \"${USE_GPU}\",/g" ${NEW_RECIPE_JSON}
# sed -i "s/\[YOUR-BUCKET\]/${S3_BUCKET}/g" ${NEW_RECIPE_JSON}
# sed -i "s/\[YOUR-PREFIX\]/${S3_PREFIX2}/g" ${NEW_RECIPE_JSON} 
sed -i "s/my-model/${ZIP_ARCHIVE_NAME}/g" ${NEW_RECIPE_JSON} 
# sed -i "s/ml\/example\/imgclassification/${TOPIC2}/g" ${NEW_RECIPE_JSON} 
sed -i "s/\(\"ComponentVersion\"\)\(.*\)/\1: \"${COMPONENT_VERSION}\",/g" ${NEW_RECIPE_JSON}

echo old json recipe: $OLD_RECIPE_JSON
echo new json recipe: $NEW_RECIPE_JSON
cd ..
echo .

# config_utils.py
echo ==--------Config Utils---------==
CONFIG_PY="artifacts/config_utils.py"

sed -i "s/\(USE_GPU\)\(.*\)/\1 = ${USE_GPU}/g" ${CONFIG_PY}
sed -i "s/\(SCORE_THRESHOLD\)\(.*\)/\1 = ${SCORE_THRESHOLD}/g" ${CONFIG_PY}
sed -i "s/\(MAX_NO_OF_RESULTS\)\(.*\)/\1 = ${MAX_NUM_CLASSES}/g" ${CONFIG_PY}
sed -i "s/\(SHAPE\)\(.*\)/\1 = ${MODEL_INPUT_SHAPE}/g" ${CONFIG_PY}
sed -i "s/\(TIMEOUT\)\(.*\)/\1 = ${TIMEOUT}/g" ${CONFIG_PY}
sed -i "s/\(DEFAULT_PREDICTION_INTERVAL_SECS\)\(.*\)/\1 = ${PREDICTION_INTERVAL_SECS}/g" ${CONFIG_PY}
sed -i "s/\(ENABLE_SEND_MESSAGE\)\(.*\)/\1 = ${ENABLE_SEND_MESSAGE}/g" ${CONFIG_PY}
# sed -i "s/\(TOPIC\)\(.*\)/\1 = \"${TOPIC2}\"/g" ${CONFIG_PY}

echo .

echo ==--------Compress Artifacts---------==
cd artifacts
zip ${ZIP_ARCHIVE_NAME}.zip -r .
S3_PATH="s3://${S3_BUCKET}/${S3_PREFIX}/${ZIP_ARCHIVE_NAME}.zip"
echo Uploading to $S3_PATH
aws s3 cp ${ZIP_ARCHIVE_NAME}.zip s3://${S3_BUCKET}/${S3_PREFIX}/${ZIP_ARCHIVE_NAME}.zip
rm ${ZIP_ARCHIVE_NAME}.zip
echo .

echo ==--------Creating Component---------==
cd ../recipes 
aws greengrassv2 create-component-version --inline-recipe fileb://${COMPONENT_NAME}-${COMPONENT_VERSION}.json 
echo .