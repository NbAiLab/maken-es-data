#!/bin/bash
# Vectorize all (books and images) since 2000-01-01 by week
# $1 output-dir

if [ "$#" -ne 1 ] && [ "$#" -ne 2 ]; then
  echo "An output-dir must be passed. Optionally, a start date in YYYYMMDD format can be passed in."
  exit
else
  OUTPUT=$(realpath ${1})
fi

echo "STARTED: $(date)"

INIT_DATE="${2:-20151101}"  # "20000101", there's nothing before 20151101
diff=$((($(date +%s -d "today")-$(date +%s -d ${INIT_DATE}))/$((7*24*3600))))
echo "Vectorizing objects for $diff weeks"

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

  echo "Vectorizing images"
  mkdir -p $WEEK_FOLDER/images/objects
  python image_vectorizer.py \
    $WEEK_FOLDER/images/records "**/**/*" \
    $WEEK_FOLDER/images/vectors \
    --objects_dir $WEEK_FOLDER/images/objects \
    --vectors_format json \
    --no_download

  echo
  echo "----------"
  echo "| BOOKS  |"
  echo "----------"
  echo

  echo "Vectorizing books"
  mkdir -p $WEEK_FOLDER/books/objects
  python book_vectorizer.py \
    $WEEK_FOLDER/books/records "**/**/*" \
    $WEEK_FOLDER/books/vectors \
    --objects_dir $WEEK_FOLDER/books/objects \
    --vectors_format json \
    --no_download

done
echo "ENDED: $(date)"
