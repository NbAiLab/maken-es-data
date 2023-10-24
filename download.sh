#!/bin/bash
# Dumps all records and objects (books and images) since 2000-01-01 by week
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
echo "Downloading objects for $diff weeks"

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

  echo "Downloading images records"
  mkdir -p $WEEK_FOLDER/images/records
  python downloader.py \
    -f "firstDigitalContentTime:[$START_DATE TO $END_DATE]" \
    -f mediatype:bilder \
    -f contentClasses:JP2 \
    -f digital:Ja \
    --groupby "accessInfo.accessAllowedFrom,metadata.originInfo.firstDigitalContentTime:DATE" \
    --size 100 \
    --scroll \
    --delay 0.0 \
    --log \
    -o $WEEK_FOLDER/images/records

  echo "Downloading images"
  mkdir -p $WEEK_FOLDER/images/objects
  python image_vectorizer.py \
    $WEEK_FOLDER/images/records "**/**/*" \
    $WEEK_FOLDER/images/objects \
    --no_inference \
    --search_local_paths "/nfsmounts/meta2/disk4/folder1/nancy/maken/catalog/images-vectors,/nfsmounts/meta2/disk4/folder1/nancy/maken/vectors_20211011/images"

  echo
  echo "----------"
  echo "| BOOKS  |"
  echo "----------"
  echo

  echo "Downloading books records"
  mkdir -p $WEEK_FOLDER/books/records
  python downloader.py \
    -f "firstDigitalContentTime:[$START_DATE TO $END_DATE]" \
    -f mediatype:bøker \
    --groupby "accessInfo.accessAllowedFrom,metadata.originInfo.firstDigitalContentTime:DATE" \
    --size 100 \
    --scroll \
    --delay 0.0 \
    --log \
    -o $WEEK_FOLDER/books/records

  echo "Downloading books"
  mkdir -p $WEEK_FOLDER/books/objects
  python book_vectorizer.py \
    $WEEK_FOLDER/books/records "**/**/*" \
    $WEEK_FOLDER/books/objects \
    --no_inference \
    --search_local_paths "/nfsmounts/dante/lv_ai_1_dante/tesseractprocessedBooks/,/nfsmounts/meta2/disk1/content/boktekst/"

done
echo "ENDED: $(date)"
