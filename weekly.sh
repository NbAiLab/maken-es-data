#!/bin/bash
# Dumps all records and objects (books and images) since 2000-01-01 by week
# $1 output-dir

if [ "$#" -ne 1 ]; then
  echo "An output-dir must be passed only. Start date is Monday last week."
  exit
else
  OUTPUT=$(realpath ${1})
fi

if [[ $(date +%u) -eq 1 ]]; then
    echo 'Sorry, you cannot run this program on a Monday.'
    exit
fi

echo "---------------"
echo "| Job started |"
echo "---------------"


START_DATE_LAST_WEEK=$(date -d "last monday-7 days" +"%Y%m%d")
END_DATE_LAST_WEEK=$(date -d "last monday-1 day" +"%Y%m%d")


echo "Time period: $START_DATE_LAST_WEEK to $END_DATE_LAST_WEEK"

./download.sh $OUTPUT $START_DATE_LAST_WEEK

./vectorize.sh $OUTPUT $START_DATE_LAST_WEEK

./ingest.sh $OUTPUT $START_DATE_LAST_WEEK

echo "-----------------"
echo "| Job finished |"
echo "----------------"