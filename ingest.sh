#!/bin/bash
# Ingest all (books and images) since 2000-01-01 by week
# $1 output-dir

if [ "$#" -ne 1 ] && [ "$#" -ne 2 ]; then
  echo "An output-dir must be passed. Optionally, a start date in YYYYMMDD format can be passed in."
  exit
else
  OUTPUT=$(realpath ${1})
fi

# User and password can be set in an .env file or pass in as environment variables
if [ -f .env ]; then
  echo "Using a .env file"
  source .env
  export ES_USER=$ES_USER
  export ES_PASSWORD=$ES_PASSWORD
fi

echo "STARTED: $(date)"

INIT_DATE="${2:-20151101}"  # "20000101", there's nothing before 20151101
diff=$((($(date +%s -d "today")-$(date +%s -d ${INIT_DATE}))/$((7*24*3600))))
echo "Ingesting objects for $diff weeks"

for i in $(seq $diff -1 1); do
  start=$(($i*7))
  end=$(($start-6))
  START_DATE=$(date -d "last monday-$start days" +"%Y%m%d")
  END_DATE=$(date -d "last monday-$end day" +"%Y%m%d")

  echo
  echo "====================="
  echo "| $START_DATE $END_DATE |"
  echo "====================="
  echo

  WEEK_FOLDER="$OUTPUT/$START_DATE-$END_DATE"

  mkdir -p $WEEK_FOLDER

  echo
  echo "-----------"
  echo "| IMAGES  |"
  echo "-----------"
  echo

  echo "Ingesting images"
  python -W ignore ingester.py  \
    https://192.168.34.21:9200 \
    images.knn \
    $WEEK_FOLDER/images/records "**/**/*" \
    $WEEK_FOLDER/images/vectors \
    --vectors_format json

  echo
  echo "----------"
  echo "| BOOKS  |"
  echo "----------"
  echo

  echo "Ingesting books"
  python -W ignore ingester.py  \
    https://192.168.34.21:9200 \
    books.knn \
    $WEEK_FOLDER/books/records "**/**/*" \
    $WEEK_FOLDER/books/vectors \
    --vectors_format json

done
echo "ENDED: $(date)"
